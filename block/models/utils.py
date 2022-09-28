import torch
import torch.nn
import torch.nn.functional as F
from .sh import eval_sh_bases
import numpy as np
import time
import pdb
from torch.utils.cpp_extension import load
g2l = load(name="g2l", sources=["/home/blee/nfs/DCT/BNeRF/cuda/global_to_local.cpp", "/home/blee/nfs/DCT/BNeRF/cuda/global_to_local.cu"])

def positional_encoding(positions, freqs):
    
        freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
        pts = (positions[..., None] * freq_bands).reshape(
            positions.shape[:-1] + (freqs * positions.shape[-1], ))  # (..., DF)
        pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
        return pts

def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1. - torch.exp(-sigma*dist)

    T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)

    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:,-1:]


def SHRender(xyz_sampled, viewdirs, features):
    sh_mult = eval_sh_bases(2, viewdirs)[:, None]
    rgb_sh = features.view(-1, 3, sh_mult.shape[-1])
    rgb = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
    return rgb


def RGBRender(xyz_sampled, viewdirs, features):

    rgb = features
    return rgb

class AlphaGridMask(torch.nn.Module):
    def __init__(self, device, aabb, alpha_volume, block_split, domain_min, domain_max): # issue block_split 
        super(AlphaGridMask, self).__init__()
        self.device = device
        self.aabb=aabb.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0] # 기존 aabb
        self.invgridSize = 1.0/self.aabbSize * 2
        # self.alpha_volume = alpha_volume.view(1,1,*alpha_volume.shape[-3:])
        self.gridSize = torch.LongTensor([alpha_volume.shape[-3],alpha_volume.shape[-2],alpha_volume.shape[-1]]).to(self.device)
        self.block_split = block_split

        x = int(alpha_volume.shape[-3] / block_split[0]) 
        y = int(alpha_volume.shape[-2] / block_split[1])
        z = int(alpha_volume.shape[-1] / block_split[2])

        self.alpha_volume_block = alpha_volume.view(block_split[0] * block_split[1] * block_split[2], 1, x,y,z)
        self.domain_min = domain_min
        self.domain_max = domain_max

    def sample_alpha(self, xyz_sampled, deb=0):
        var = True
        if var:
            # init
            voxel_size = torch.Tensor([(self.aabb[1] - self.aabb[0])[i] / self.block_split[i] for i in range(len(self.block_split))]).to(self.device)
            network_strides = torch.Tensor([self.block_split[2] * self.block_split[1], self.block_split[2], 1]).to(self.device) # 이게 맞지 않나?
            n_block = self.block_split[0] * self.block_split[1] * self.block_split[2]

            # flip!!!
            xyz_sampled = xyz_sampled.view(-1,3)
            # xyz_sampled = xyz_sampled.flip(-1)

            # global points to local points
            # 여기서 넘어가는 애들을 억지로 넣어서 문제?
            points_indices_3d = ((xyz_sampled - self.aabb[0]) / voxel_size).long() # 그냥 얘 기준으로 16넘으면 제끼는게 낫겟다
            for i in range(3):  # x가 15일 때만 문제인건데 일단은 다 고쳐보자.
                idx_edge = (points_indices_3d[...,i] >= self.block_split[i]).nonzero().squeeze(-1)
                if idx_edge.shape[0] > 0:
                    points_indices_3d[idx_edge,i] = self.block_split[i] - 1

                idx_edge = (points_indices_3d[...,i] < 0).nonzero().squeeze(-1)
                if idx_edge.shape[0] > 0:
                    points_indices_3d[idx_edge,i] = 0
                del idx_edge
                # 결국 clamp;;

            filtered_point_indices = (points_indices_3d * network_strides).sum(dim=1).long()

            filtered_point_indices, reorder_indices = torch.sort(filtered_point_indices)
            contained_blocks, batch_size_per_network_incomplete = torch.unique_consecutive(filtered_point_indices, return_counts=True)
            batch_size_per_network = torch.zeros(n_block, device=self.device, dtype=torch.long)

            valid_contained_blocks = contained_blocks[contained_blocks >= 0]
            valid_contained_blocks = valid_contained_blocks[valid_contained_blocks < n_block]
            invalid_contained_blocks = contained_blocks[contained_blocks >= n_block]
            invalid_contained_blocks_neg = contained_blocks[contained_blocks < 0]

            n_valid_block = valid_contained_blocks.shape[0]
            n_invalid_block = invalid_contained_blocks.shape[0]
            n_neg_block = invalid_contained_blocks_neg.shape[0]


            valid_batch_size_per_network_incomplete = batch_size_per_network_incomplete[n_neg_block:n_valid_block + n_neg_block]

            invalid_batch_size_per_network_incomplete = batch_size_per_network_incomplete[n_valid_block + n_neg_block:]
            invalid_batch_size_per_network_incomplete_negative = batch_size_per_network_incomplete[:n_neg_block]

            n_valid_pts = valid_batch_size_per_network_incomplete.sum()
            n_invalid_pts = invalid_batch_size_per_network_incomplete.sum()
            n_invalid_pts_neg = invalid_batch_size_per_network_incomplete_negative.sum()

            pts_book = filtered_point_indices[n_invalid_pts_neg:n_valid_pts + n_invalid_pts_neg].float()   
            
            batch_size_per_network[valid_contained_blocks] = valid_batch_size_per_network_incomplete  
            batch_size_per_network = batch_size_per_network.cpu()
            
            points_reordered = xyz_sampled[reorder_indices] 

            g2l.global_to_local(points_reordered[n_invalid_pts_neg:n_valid_pts + n_invalid_pts_neg], self.domain_min, self.domain_max, batch_size_per_network, 1, 64)

            alpha_vals = []
            st = n_invalid_pts_neg.item()
            points_reordered = points_reordered.view(1,-1,1,1,3)
            for b, batch_size in enumerate(batch_size_per_network):
                if batch_size == 0:
                    continue

                block_coord = points_reordered[None,0, st: st + batch_size, ...]
                alpha_val = F.grid_sample(self.alpha_volume_block[None,b], block_coord, align_corners=True).view(-1)
                alpha_vals.append(alpha_val)
                st += batch_size

            alpha_vals = torch.cat(alpha_vals).to(self.device)
            return alpha_vals

        else:
            # init
            voxel_size = (self.aabb[1] - self.aabb[0]) / self.block_split
            network_strides = torch.Tensor([self.block_split**2, self.block_split, 1]).to(self.device)
            n_block = self.block_split**3

            # flip!!!
            xyz_sampled = xyz_sampled.view(-1,3)
            # xyz_sampled = xyz_sampled.flip(-1)

            # global points to local points
            # 여기서 넘어가는 애들을 억지로 넣어서 문제?
            points_indices_3d = ((xyz_sampled - self.aabb[0]) / voxel_size).long() # 그냥 얘 기준으로 16넘으면 제끼는게 낫겟다
            for i in range(3):  # x가 15일 때만 문제인건데 일단은 다 고쳐보자.
                idx_edge = (points_indices_3d[...,i] >= self.block_split).nonzero().squeeze(-1)
                if idx_edge.shape[0] > 0:
                    points_indices_3d[idx_edge,i] = self.block_split - 1

                idx_edge = (points_indices_3d[...,i] < 0).nonzero().squeeze(-1)
                if idx_edge.shape[0] > 0:
                    points_indices_3d[idx_edge,i] = 0
                del idx_edge
                # 결국 clamp;;

            filtered_point_indices = (points_indices_3d * network_strides).sum(dim=1).long()

            filtered_point_indices, reorder_indices = torch.sort(filtered_point_indices)
            contained_blocks, batch_size_per_network_incomplete = torch.unique_consecutive(filtered_point_indices, return_counts=True)
            batch_size_per_network = torch.zeros(n_block, device=self.device, dtype=torch.long)

            valid_contained_blocks = contained_blocks[contained_blocks >= 0]
            valid_contained_blocks = valid_contained_blocks[valid_contained_blocks < n_block]
            invalid_contained_blocks = contained_blocks[contained_blocks >= n_block]
            invalid_contained_blocks_neg = contained_blocks[contained_blocks < 0]

            n_valid_block = valid_contained_blocks.shape[0]
            n_invalid_block = invalid_contained_blocks.shape[0]
            n_neg_block = invalid_contained_blocks_neg.shape[0]


            valid_batch_size_per_network_incomplete = batch_size_per_network_incomplete[n_neg_block:n_valid_block + n_neg_block]

            invalid_batch_size_per_network_incomplete = batch_size_per_network_incomplete[n_valid_block + n_neg_block:]
            invalid_batch_size_per_network_incomplete_negative = batch_size_per_network_incomplete[:n_neg_block]

            n_valid_pts = valid_batch_size_per_network_incomplete.sum()
            n_invalid_pts = invalid_batch_size_per_network_incomplete.sum()
            n_invalid_pts_neg = invalid_batch_size_per_network_incomplete_negative.sum()

            pts_book = filtered_point_indices[n_invalid_pts_neg:n_valid_pts + n_invalid_pts_neg].float()   
            
            batch_size_per_network[valid_contained_blocks] = valid_batch_size_per_network_incomplete  
            batch_size_per_network = batch_size_per_network.cpu()
            
            points_reordered = xyz_sampled[reorder_indices] 

            g2l.global_to_local(points_reordered[n_invalid_pts_neg:n_valid_pts + n_invalid_pts_neg], self.domain_min, self.domain_max, batch_size_per_network, 1, 64)

            alpha_vals = []
            st = n_invalid_pts_neg.item()
            points_reordered = points_reordered.view(1,-1,1,1,3)
            for b, batch_size in enumerate(batch_size_per_network):
                if batch_size == 0:
                    continue

                block_coord = points_reordered[None,0, st: st + batch_size, ...]
                alpha_val = F.grid_sample(self.alpha_volume_block[None,b], block_coord, align_corners=True).view(-1)
                alpha_vals.append(alpha_val)
                st += batch_size

            alpha_vals = torch.cat(alpha_vals).to(self.device)
            return alpha_vals

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.aabb[0]) * self.invgridSize - 1


class MLPRender_Fea(torch.nn.Module):
    def __init__(self,inChanel, viewpe=6, feape=6, featureC=128):
        super(MLPRender_Fea, self).__init__()

        self.in_mlpC = 2*viewpe*3 + 2*feape*inChanel + 3 + inChanel
        self.viewpe = viewpe
        self.feape = feape
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)


    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb

class MLPRender_PE(torch.nn.Module):
    def __init__(self,inChanel, viewpe=6, pospe=6, featureC=128):
        super(MLPRender_PE, self).__init__()

        self.in_mlpC = (3+2*viewpe*3)+ (3+2*pospe*3)  + inChanel #
        self.viewpe = viewpe
        self.pospe = pospe
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.pospe > 0:
            indata += [positional_encoding(pts, self.pospe)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb

class MLPRender(torch.nn.Module):
    def __init__(self,inChanel, viewpe=6, featureC=128):
        super(MLPRender, self).__init__()

        self.in_mlpC = (3+2*viewpe*3) + inChanel
        self.viewpe = viewpe
        
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb

