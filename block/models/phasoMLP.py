import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import numpy as np

from .cosine_transform import dctn, idctn
from .phasoBase import *
from .utils import positional_encoding
from .utils_fft import (
    getMask_fft, getMask, grid_sample, grid_sample_cmplx, irfft, rfft,
    batch_irfft
)

import itertools
from torch.utils.cpp_extension import load
from cuda.grid import *
from utils import qat

g2l = load(name="g2l", sources=["cuda/global_to_local.cpp", "cuda/global_to_local.cu"])


class CPhasoMLP(PhasorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super().__init__(aabb, gridSize, device, **kargs)

    def init_phasor_volume(self, res, device, block_split, logger, mask_lr, mask):
        """ initialize volume """
        if logger is not None:
            self.logger = logger
        self.axis = [torch.tensor([0.]+[2**i for i in torch.arange(d-1)],
                                  device=self.device)
                     for d in self.app_num_comp]   


        # Block init
        print(block_split)
        self.resolution = res
        self.block_split = block_split
        self.bbox_size = self.aabb[1] - self.aabb[0]  
        self.n_block = self.block_split[0] * self.block_split[1] * self.block_split[2]

        self.n_freq = 8    
        self.block_resolution = [int(res / bs) for res, bs in zip(self.resolution, self.block_split)]
        self.logger.info(f"split per axis: {self.block_split},  #blocks: {self.n_block},  #blk_res: {self.block_resolution}")
        self.den_chan = 1
        self.app_chan = 2
        self.voxel_size = torch.Tensor([self.bbox_size[i] / self.block_split[i] for i in range(len(self.block_split))]).to(device)
        self.domain_min, self.domain_max = self.compute_domain()    
        self.network_strides = torch.Tensor([self.block_split[2] * self.block_split[1], self.block_split[2], 1]).to(device)

        self.den = nn.ParameterList(
            self.init_(self.den_num_comp,
                       self.block_resolution,
                       ksize=self.n_freq * self.den_chan)
        )

        self.app = nn.ParameterList(
            self.init_(self.app_num_comp,
                       self.block_resolution,
                       ksize=self.n_freq * self.app_chan)
        )

        # threshold
        self.target_param = int((sum([d.numel() for d in self.den]) + sum([d.numel() for d in self.app]))* 0.1)
        self.logger.info(f"target parameter to {self.target_param}")
        print(f"target parameter to {self.target_param}")
        self.num_unmasked_den = 0
        self.num_unmasked_app = 0

        self.ktraj_den = self.compute_ktraj(
            self.axis, self.resolution, self.den_scale)
        
        # mask
        self.mask_lr = mask_lr
        self.logger.info(f"mask lr: {self.mask_lr}")
        self.den_mask = nn.ParameterList(
            [nn.Parameter(torch.zeros_like(d)) for d in self.den]) if mask else None
        self.app_mask = nn.ParameterList(
            [nn.Parameter(torch.zeros_like(d)) for d in self.app]) if mask else None

        self.iter = None


        self.concat = True
        den_outdim = self.den_chan * self.n_freq    
        app_outdim = self.app_chan * self.n_freq
        if self.concat:
            den_outdim *= 3
            app_outdim *= 3

        if self.app_aug == 'flip':
            app_outdim = app_outdim * 2
        elif self.app_aug == 'normal':
            app_outdim = app_outdim + 3
        elif self.app_aug == 'flip++':
            app_outdim = app_outdim * 4

        self.basis_mat = nn.Linear(
            app_outdim, self.app_dim, bias=False).to(device)
        self.alpha_params = nn.Parameter(
            torch.tensor([self.alpha_init]).to(device))
        self.beta = nn.Parameter(torch.tensor([self.alpha_init]).to(device))
        self.mlp = nn.Sequential(nn.Linear(den_outdim, 64),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(64, 1)).to(device)
        print(self)

    def compute_domain(self):
        device = torch.device("cuda")
        domain_mins = []
        domain_maxs = []
        res = self.block_split
        for coord in itertools.product(*[range(r) for r in res]):
            coord = torch.tensor(coord, device=device) 
            domain_min = self.aabb[0] + self.voxel_size * coord
            domain_max = domain_min + self.voxel_size
            domain_mins.append(domain_min.tolist())
            domain_maxs.append(domain_max.tolist())
        return torch.tensor(domain_mins, device=device), torch.tensor(domain_maxs, device=device)    



    @torch.no_grad()
    def compute_ktraj(self, axis, res, den_scale): 
        bx, by, bz = self.block_resolution
        ktraj2d = [torch.fft.fftfreq(i, 1/i).to(self.device)
                for i in res]
        ktraj1d = [torch.zeros(r).to(self.device) for r in self.block_split]

        ktrajx = torch.stack(
            torch.meshgrid([ktraj1d[0], ktraj2d[1], ktraj2d[2]]), dim=-1).view(self.n_block, 1, by, bz, 3)

        ktrajy = torch.stack(
            torch.meshgrid([ktraj2d[0], ktraj1d[1], ktraj2d[2]]), dim=-1).view(self.n_block, bx, 1, bz, 3)

        ktrajz = torch.stack(
            torch.meshgrid([ktraj2d[0], ktraj2d[1], ktraj1d[2]]), dim=-1).view(self.n_block, bx, by, 1, 3)
            
        return [ktrajx, ktrajy, ktrajz]



    def init_(self, axis, res, ksize=1, init_scale=1):
        Nx, Ny, Nz = res    
        d1, d2, d3 = axis   

        fx = torch.zeros(self.n_block, 1, ksize, d1, Ny, Nz, dtype=torch.float32,     
                         device=self.device)
        fy = torch.zeros(self.n_block, 1, ksize, Nx, d2, Nz, dtype=torch.float32,
                         device=self.device)
        fz = torch.zeros(self.n_block, 1, ksize, Nx, Ny, d3, dtype=torch.float32,
                         device=self.device)

        return [nn.Parameter(fx), nn.Parameter(fy), nn.Parameter(fz)]

    def compute_densityfeature(self, xyz_sampled):
        sigma_feature = self.compute_fft(self.density, xyz_sampled,
                                         self.den_mask, True)
        return self.mlp(sigma_feature.T).T

    def feature2density(self, density_features):
        if self.fea2denseAct == "softplus":
            return F.softplus(density_features+self.density_shift,
                              beta=self.softplus_beta)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features)

    def compute_appfeature(self, xyz_sampled, viewdirs):
        app_points, points_backordered = self.compute_fft(self.appearance, xyz_sampled,
                                      self.app_mask, False) 
        if self.app_aug == 'flip':
            aug = self.compute_fft(self.appearance, xyz_sampled.flip(-1),
                                   self.app_mask, False, True)     
            app_points = torch.cat([app_points, aug], dim=0)
        elif self.app_aug == 'normal':
            aug = self.compute_normal(xyz_sampled)
            app_points = torch.cat([app_points, aug], dim=0)
        elif self.app_aug == 'flip++':
            aug1 = self.compute_fft(self.appearance, xyz_sampled.flip(-1))
            aug2 = self.compute_fft(self.appearance, -xyz_sampled)
            aug3 = self.compute_fft(self.appearance, -xyz_sampled.flip(-1))
            app_points = torch.cat([app_points, aug1, aug2, aug3], dim=0)
        elif self.app_aug != 'none':
            raise NotImplementedError(f'{self.app_aug} not implemented')

        # print(app_points.shape)
        return self.basis_mat(app_points.T), points_backordered
    
    def L1_loss(self):
        total = 0
        for d in self.den:
            total = total + torch.mean(torch.abs(d))
        return total

    def L2_loss(self):
        total = 0
        rendering_layers_idx = [0,2,4]
        rendering_layers = [self.renderModule.mlp[0].parameters(), self.renderModule.mlp[2].parameters(), self.renderModule.mlp[4].parameters()]
        for l in rendering_layers:
            for p in l:
                total += p.norm(2)
        return total



    def compute_fft(self, features, xyz_sampled, mask=None, is_den=False, aug=False):
        # this is fast because of 2d transform and matrix multiplication.
        # (N*N) logN d + Nsamples * d*d + 3 * Nsamples
        # TODO
        
        is_qat = True  # Implement qat if true
        if is_qat:
            Fx = qat(features[0])
            Fy = qat(features[1])
            Fz = qat(features[2])
        else:
            Fx, Fy, Fz = features

        if mask is not None:
            mx, my, mz = mask
            mx = torch.sigmoid(mx)
            my = torch.sigmoid(my)
            mz = torch.sigmoid(mz)

            mx_unmasked = mx >= self.mask_thres
            my_unmasked = my >= self.mask_thres
            mz_unmasked = mz >= self.mask_thres

            if self.mask_learning:
                if is_den:
                    self.num_unmasked_den = mx_unmasked.count_nonzero() + my_unmasked.count_nonzero() + mz_unmasked.count_nonzero()
                if not is_den and not aug:
                    self.num_unmasked_app = mx_unmasked.count_nonzero() + my_unmasked.count_nonzero() + mz_unmasked.count_nonzero()

            Fx = (Fx * mx_unmasked - Fx * mx).detach() + Fx * mx
            Fy = (Fy * my_unmasked - Fy * my).detach() + Fy * my
            Fz = (Fz * mz_unmasked - Fz * mz).detach() + Fz * mz



            if self.iter is not None and self.iter % 1000 == 1:
                F_live = (mx >= self.mask_thres).count_nonzero()+ (my >= self.mask_thres).count_nonzero()+(mz >= self.mask_thres).count_nonzero()
                F_all = Fx.numel() + Fy.numel() + Fz.numel()
                self.logger.info(f"{F_live} / {F_all} alive. {(F_all - F_live) / F_all * 100}% masked")

        Fx = idctn(Fx, axes=(4, 5))
        Fy = idctn(Fy, axes=(3, 5))
        Fz = idctn(Fz, axes=(3, 4))

        feat_size = self.den_chan * self.n_freq if is_den else self.app_chan * self.n_freq
        feat_size = feat_size * 3 if self.concat else feat_size

        num_rays = xyz_sampled.size(0)
        num_samples = xyz_sampled.size(1)
        global_domain_min, global_domain_max = self.aabb
        points_flat = xyz_sampled.view(-1,3)

        points_indices_3d = ((points_flat - global_domain_min) / self.voxel_size).long()
        normalize = True
        if normalize:
            for i in range(3): 
                idx_edge = (points_indices_3d[...,i] >= self.block_split[i]).nonzero().squeeze(-1)
                if idx_edge.shape[0] > 0:
                    points_indices_3d[idx_edge,i] = self.block_split[i] - 1

                idx_edge = (points_indices_3d[...,i] < 0).nonzero().squeeze(-1)
                if idx_edge.shape[0] > 0:
                    points_indices_3d[idx_edge,i] = 0
                del idx_edge

        point_indices = (points_indices_3d * self.network_strides).sum(dim=1).long() 
        del points_indices_3d

        filtered_point_indices = point_indices
        del point_indices

        filtered_point_indices, reorder_indices = torch.sort(filtered_point_indices)    
        
        contained_blocks, batch_size_per_network_incomplete = torch.unique_consecutive(filtered_point_indices, return_counts=True)
        batch_size_per_network = torch.zeros(self.n_block, device=points_flat.device, dtype=torch.long)

        valid_contained_blocks = contained_blocks[contained_blocks >= 0]
        valid_contained_blocks = valid_contained_blocks[valid_contained_blocks < self.n_block]
        invalid_contained_blocks = contained_blocks[contained_blocks >= self.n_block]
        invalid_contained_blocks_neg = contained_blocks[contained_blocks < 0]

        if len(valid_contained_blocks) == 0:  
            print("No valid pts to compute")
            self.logger.info("No valid pts to compute")
            return torch.zeros([feat_size, filtered_point_indices.shape[0]], device=valid_contained_blocks.device)

        n_valid_block = valid_contained_blocks.shape[0]
        n_invalid_block = invalid_contained_blocks.shape[0]
        n_neg_block = invalid_contained_blocks_neg.shape[0]

        valid_batch_size_per_network_incomplete = batch_size_per_network_incomplete[n_neg_block:n_valid_block + n_neg_block]

        invalid_batch_size_per_network_incomplete = batch_size_per_network_incomplete[n_valid_block + n_neg_block:]
        invalid_batch_size_per_network_incomplete_negative = batch_size_per_network_incomplete[:n_neg_block]

        n_valid_pts = valid_batch_size_per_network_incomplete.sum()
        n_invalid_pts = invalid_batch_size_per_network_incomplete.sum()
        n_invalid_pts_neg = invalid_batch_size_per_network_incomplete_negative.sum()

        pts_book = filtered_point_indices[n_invalid_pts_neg:n_valid_pts + n_invalid_pts_neg].float()  # convert to float b/c of cuda issue.(TODO!!)
        
        batch_size_per_network[valid_contained_blocks] = valid_batch_size_per_network_incomplete  
        batch_size_per_network = batch_size_per_network.cpu()
        
        points_reordered = points_flat[reorder_indices] 
        del points_flat
        
        g2l.global_to_local(points_reordered[n_invalid_pts_neg:n_valid_pts + n_invalid_pts_neg], self.domain_min, self.domain_max, batch_size_per_network, 1, 64)
        
        if points_reordered[n_invalid_pts_neg:n_valid_pts + n_invalid_pts_neg].isnan().count_nonzero() > 0:
            print(self.iter)
            import pdb
            pdb.set_trace()

        coefs = torch.zeros((feat_size, n_invalid_pts_neg)).to(xyz_sampled.device)
        st = n_invalid_pts_neg.item()

        coefs_xyz = GridSample.apply(Fx.transpose(3,3).flatten(2,3), Fy.transpose(3,4).flatten(2,3), Fz.transpose(3,5).flatten(2,3), points_reordered[n_invalid_pts_neg:n_valid_pts + n_invalid_pts_neg], pts_book, self.iter)
        if coefs_xyz.isnan().count_nonzero() > 0:
            print(self.iter)
            import pdb
            pdb.set_trace()
        coefs_xyz = torch.cat([coefs_xyz[0], coefs_xyz[1], coefs_xyz[2]])

        if n_invalid_pts_neg > 0:
            coefs = torch.cat([coefs, coefs_xyz], dim=-1)  
        else:
            coefs = coefs_xyz
        del coefs_xyz
                
        
        invalid_coefs = torch.zeros((feat_size, n_invalid_pts)).to(coefs.device)
        coefs = torch.cat([coefs, invalid_coefs], dim=1) 
        coefs_backordered = torch.zeros_like(coefs, device=coefs.device)
        points_backordered = torch.zeros_like(points_reordered, device=coefs.device)
        
        # backorder
        coefs_backordered[:,reorder_indices] = coefs  
        points_backordered[reorder_indices,:] = points_reordered

        if is_den or aug: 
            return coefs_backordered # fxx+fyy+fzz
        else:
            return coefs_backordered, points_backordered


    def Parseval_Loss(self):
        loss = 0
        app = False
        if app:
            for block_app, block_den, block_ktraj_den in zip(self.app, self.density, self.ktraj_den):
                feat_den = torch.pi * block_den[..., None] * block_ktraj_den.reshape(self.n_block, 1, 1, *block_den.shape[3:], -1)
                feat_app = torch.pi * block_app[..., None] * block_ktraj_den.reshape(self.n_block, 1, 1, *block_app.shape[3:], -1)
                loss = loss + feat_den.square().mean() + feat_app.square().mean()
        else:
            for block_den, block_ktraj_den in zip(self.density, self.ktraj_den):
                feat = torch.pi * block_den[..., None] * block_ktraj_den.reshape(self.n_block, 1, 1, *block_den.shape[3:], -1)
                loss = loss + feat.square().mean()
        

        return loss

    def compute_normal(self, xyz_sampled):  # TODO. need codes for the 1st backward with different class
        with torch.enable_grad():
            xyz_sampled.requires_grad = True
            outs = self.compute_densityfeature(xyz_sampled)
            d_points = torch.ones_like(outs, requires_grad=False,
                                       device=self.device)
            normal = grad(outputs=outs, inputs=xyz_sampled,
                          grad_outputs=d_points, retain_graph=False,
                          only_inputs=True)[0]
            normal = normal.T
            normal = normal / torch.linalg.norm(normal, dim=0, keepdims=True)
            return normal.detach()

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        res_den = [math.ceil(n * self.den_scale) for n in res_target]
        res_app = [math.ceil(n * self.app_scale) for n in res_target]
        
        self.block_resolution = [int(res / bs) for res, bs in zip(res_den, self.block_split)]

        new_den = self.upsample_feats(self.den, res_den)
        self.den = nn.ParameterList([nn.Parameter(den) for den in new_den])

        new_app = self.upsample_feats(self.app, res_app)
        self.app = nn.ParameterList([nn.Parameter(app) for app in new_app])

        # mask
        new_den_mask = self.upsample_feats(self.den_mask, res_den)
        self.den_mask = nn.ParameterList([nn.Parameter(den_mask)
                                          for den_mask in new_den_mask])

        new_app_mask = self.upsample_feats(self.app_mask, res_app)
        self.app_mask = nn.ParameterList([nn.Parameter(app_mask)
                                          for app_mask in new_app_mask])

        self.resolution = res_den
        self.update_stepSize(res_den) 
        print(f'upsamping to {res_den}')
        self.logger.info(self.den)
        print(self.den)

        if self.mask_learning:
            self.target_param = int((sum([d.numel() for d in self.den]) + sum([d.numel() for d in self.app]))* 0.1)
            self.logger.info(f"target parameter to {self.target_param}")
            print(f"target parameter to {self.target_param}")



    def upsample_feats(self, features, res_target, update_dd=False):
        Tx, Ty, Tz = res_target
        Fx, Fy, Fz = features
        d1, d2, d3 = Fx.shape[-3], Fy.shape[-2], Fz.shape[-1]
        Nx, Ny, Nz = Fy.shape[-3], Fz.shape[-2], Fx.shape[-1]

        Tx, Ty, Tz = self.block_resolution


        return F.pad(Fx, (0, Tz-Nz, 0, Ty-Ny, 0, 0)), \
               F.pad(Fy, (0, Tz-Nz, 0, 0, 0, Tx-Nx)), \
               F.pad(Fz, (0, 0, 0, Ty-Ny, 0, Tx-Nx))


    def update_stepSize(self, gridSize):
        self.ktraj_den = self.compute_ktraj(
            self.axis, gridSize, self.den_scale)
        print("dimensions largest ", [torch.max(ax).item() for ax in self.axis])
        return super(CPhasoMLP, self).update_stepSize(gridSize)

    def print_size(self):
        print(self)
        self.logger.info(self)
        print(f' ==> Actual Model Size {np.sum([v.numel() * v.element_size() for k, v in self.named_parameters()])/2**20} MB')
        self.logger.info(f' ==> Actual Model Size {np.sum([v.numel() * v.element_size() for k, v in self.named_parameters()])/2**20} MB')
        for k,v in self.named_parameters():
            print(f'Model Size ({k}) : '
                  f'{v.numel() * v.element_size()/2**20:.4f} MB')
            self.logger.info(f'Model Size ({k}) : '
                  f'{v.numel() * v.element_size()/2**20:.4f} MB')

    def get_optparam_groups(self, lr_init_spatialxyz=0.02,
                            lr_init_network=0.001):
        if self.mask_learning:
            grad_vars = [{'params': self.den, 'lr': lr_init_spatialxyz},
                        {'params': self.app, 'lr': lr_init_spatialxyz},
                        {'params': self.den_mask, 'lr': lr_init_spatialxyz * self.mask_lr},
                        {'params': self.app_mask, 'lr': lr_init_spatialxyz * self.mask_lr},
                        {'params': self.basis_mat.parameters(),
                        'lr':lr_init_network},
                        {'params': self.mlp.parameters(),
                        'lr': lr_init_network},
                        {'params': self.mask_thres,
                        'lr':1e-3}]
        else:
            grad_vars = [{'params': self.den, 'lr': lr_init_spatialxyz},
                        {'params': self.app, 'lr': lr_init_spatialxyz},
                        {'params': self.den_mask, 'lr': lr_init_spatialxyz * self.mask_lr},
                        {'params': self.app_mask, 'lr': lr_init_spatialxyz * self.mask_lr},
                        {'params': self.basis_mat.parameters(),
                        'lr':lr_init_network},
                        {'params': self.mlp.parameters(),
                        'lr': lr_init_network}]

        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(),
                           'lr':lr_init_network}]
        return grad_vars

    @property
    def density(self):
        return [self.alpha * den for den in self.den]

    @property
    def appearance(self):
        return [app * self.beta for app in self.app]

    @property
    def alpha(self):
        return F.softplus(self.alpha_params, beta=10, threshold=1e-4)
