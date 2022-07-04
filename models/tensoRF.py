import torch.nn as nn
from .modules import PosEncoding
from .tensorBase import *
from .voxel_based import FreqGrid


class PREF(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super().__init__(aabb, gridSize, device, **kargs)
        self.density = nn.Sequential(
            # PREF_backbone((256, 256, 256), 2),
            FreqGrid(256, 2),
            nn.Linear(2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1))

        self.appearance = nn.Sequential(
            # PREF_backbone((256, 256, 256), 4),
            FreqGrid(256, 4),
            nn.Linear(4, 27))

    def get_optparam_groups(self, lr_init_spatialxyz=0.02,
                            lr_init_network=0.001):
        return {'params': self.parameters(), 'lr': lr_init_spatialxyz}

    def compute_densityfeature(self, coords):
        return self.density(coords).squeeze(-1)

    def compute_appfeature(self, coords):
        return self.appearance(coords)

    def density_L1(self):
        return torch.zeros(())

    def TV_loss_density(self, reg):
        return self.density[0].compute_tv()

    def TV_loss_app(self, reg):
        return self.appearance[0].compute_tv()

    def bits(self, precision=16):
        return precision * sum([p.numel() for p in self.parameters()])


class TensorVM(TensorBase):
    def __init__(self, aabb, gridSize, device, density_n_comp=8,
                 appearance_n_comp=24, **kargs):
        super().__init__(aabb, gridSize, device, **kargs)
        self.density_n_comp = density_n_comp
        self.appearance_n_comp = appearance_n_comp

        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]
        self.comp_w = [1, 1, 1]

    def init_svd_volume(self, res, device):
        self.plane_coef = torch.nn.Parameter(
            0.1 * torch.randn((3, self.app_n_comp + self.density_n_comp, res, res), device=device))
        self.line_coef = torch.nn.Parameter(
            0.1 * torch.randn((3, self.app_n_comp + self.density_n_comp, res, 1), device=device))
        self.basis_mat = torch.nn.Linear(self.app_n_comp * 3, self.app_dim, bias=False, device=device)

    def get_optparam_groups(self, lr_init_spatialxyz=0.02,
                            lr_init_network=0.001):
        grad_vars = [{'params': self.line_coef, 'lr': lr_init_spatialxyz},
                     {'params': self.plane_coef, 'lr': lr_init_spatialxyz},
                     {'params': self.basis_mat.parameters(),
                      'lr':lr_init_network}]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(),
                           'lr':lr_init_network}]
        return grad_vars

    def compute_densityfeature(self, xyz_sampled):
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]],
                                        xyz_sampled[..., self.matMode[1]],
                                        xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]],
                                       xyz_sampled[..., self.vecMode[1]],
                                       xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line),
                                       coordinate_line), dim=-1)
        coordinate_line = coordinate_line.detach().view(3, -1, 1, 2)

        plane_feats = F.grid_sample(self.plane_coef[:, -self.density_n_comp:],
                                    coordinate_plane,
                                    align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_feats = F.grid_sample(self.line_coef[:, -self.density_n_comp:],
                                   coordinate_line, align_corners=True).view(-1, *xyz_sampled.shape[:1])

        return torch.sum(plane_feats * line_feats, dim=0)

    def compute_appfeature(self, xyz_sampled):
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        plane_feats = F.grid_sample(self.plane_coef[:, :self.app_n_comp], coordinate_plane, align_corners=True).view(3 * self.app_n_comp, -1)
        line_feats = F.grid_sample(self.line_coef[:, :self.app_n_comp], coordinate_line, align_corners=True).view(3 * self.app_n_comp, -1)

        return self.basis_mat((plane_feats * line_feats).T)

    def vectorDiffs(self, vector_comps):
        total = 0

        for idx in range(len(vector_comps)):
            n_comp, n_size = vector_comps[idx].shape[:-1]

            dotp = torch.matmul(vector_comps[idx].view(n_comp,n_size), vector_comps[idx].view(n_comp,n_size).transpose(-1,-2))
            non_diagonal = dotp.view(-1)[1:].view(n_comp-1, n_comp+1)[...,:-1]
            total = total + torch.mean(torch.abs(non_diagonal))

        return total

    def vector_comp_diffs(self):
        return self.vectorDiffs(self.line_coef[:,-self.density_n_comp:]) + self.vectorDiffs(self.line_coef[:,:self.app_n_comp])

    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]

            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(plane_coef[i].data,
                              size=(res_target[mat_id_1], res_target[mat_id_0]),
                              mode='bilinear', align_corners=True))
            line_coef[i] = torch.nn.Parameter(
                F.interpolate(line_coef[i].data, size=(res_target[vec_id], 1),
                              mode='bilinear', align_corners=True))

        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        # assuming xyz have the same scale
        scale = res_target[0]/self.line_coef.shape[2] 
        plane_coef = F.interpolate(self.plane_coef.detach().data, scale_factor=scale, mode='bilinear',align_corners=True)
        line_coef = F.interpolate(self.line_coef.detach().data, size=(res_target[0], 1), mode='bilinear',align_corners=True)
        self.plane_coef, self.line_coef = torch.nn.Parameter(plane_coef), torch.nn.Parameter(line_coef)
        self.compute_stepSize(res_target)
        print(f'upsamping to {res_target}')


class TensorCP(TensorBase):
    def __init__(self, aabb, gridSize, device, density_n_comp=8,
                 appearance_n_comp=24, **kargs):
        super().__init__(aabb, gridSize, device, **kargs)
        self.density_n_comp = density_n_comp
        self.appearance_n_comp = appearance_n_comp

        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]
        self.comp_w = [1, 1, 1]

    def init_svd_volume(self, res, *args, **kwargs):
        self.density_line = self.init_one_svd(
            self.density_n_comp[0], self.gridSize, 0.2) # 1e-3) # scale)
        self.app_line = self.init_one_svd(
            self.app_n_comp[0], self.gridSize, 0.2) # 0.1) # scale)
        self.basis_mat = torch.nn.Linear(
            self.app_n_comp[0], self.app_dim, bias=False)

    def init_one_svd(self, n_component, grid_size, scale):
        return torch.nn.ParameterList([
            torch.nn.Parameter(scale * torch.randn((1, n_component, g, 1)))
            for g in grid_size])

    def get_optparam_groups(self, lr_init_spatialxyz=0.02,
                            lr_init_network=0.001):
        grad_vars = [{'params': self.density_line, 'lr': lr_init_spatialxyz},
                     {'params': self.app_line, 'lr': lr_init_spatialxyz},
                     {'params': self.basis_mat.parameters(),
                      'lr':lr_init_network}]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(),
                           'lr':lr_init_network}]
        return grad_vars

    def preprocess_coords(self, coords):
        coords = torch.stack(tuple(coords[..., ax] for ax in self.vecMode))
        coords = torch.stack((torch.zeros_like(coords), coords), dim=-1)
        return coords.detach().view(3, -1, 1, 2)

    def sample_feats(self, features, coords):
        # ORIGINAL
        sampled = F.grid_sample(features, coords, align_corners=True)
        return sampled.squeeze(0).squeeze(-1)
        '''
        features = features[0, ..., 0] # [C, V]
        coords = (coords[0, ..., -1] + 1) / 2 # [B, 1]
        coords = torch.cos(
            coords * torch.pi
            * (torch.arange(features.size(-1)).to(coords.device) + 0.5))
        return torch.einsum('CV,BV->CB', features, coords)
        '''

    def compute_appfeature(self, coords):
        coords = self.preprocess_coords(coords)
        feats = 1
        for i in range(3):
            feats = feats * self.sample_feats(self.app_line[i], coords[[i]])
        return self.basis_mat(feats.T)

    def compute_densityfeature(self, coords):
        coords = self.preprocess_coords(coords)
        feats = 1
        for i in range(3):
            feats = feats * self.sample_feats(self.density_line[i], coords[[i]])
        return torch.sum(feats, dim=0)

    @torch.no_grad()
    def up_sampling_Vector(self, density_line_coef, app_line_coef, res_target):
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            density_line_coef[i] = torch.nn.Parameter(
                F.interpolate(density_line_coef[i].data,
                              size=(res_target[vec_id], 1), mode='bilinear',
                              align_corners=True))
            app_line_coef[i] = torch.nn.Parameter(
                F.interpolate(app_line_coef[i].data,
                              size=(res_target[vec_id], 1),
                              mode='bilinear', align_corners=True))

        return density_line_coef, app_line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.density_line, self.app_line = self.up_sampling_Vector(
            self.density_line, self.app_line, res_target)
        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')

    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.density_line[i] = torch.nn.Parameter(
                self.density_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            self.app_line[i] = torch.nn.Parameter(
                self.app_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )

        if not torch.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize-1), (b_r-1) / (self.gridSize-1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
            correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))

    def density_L1(self):
        return self.TV_loss_density(lambda x: torch.mean(torch.abs(x)))

    def TV_loss_density(self, reg):
        return sum([reg(l) for l in self.density_line]) * 1e-3

    def TV_loss_app(self, reg):
        return sum([reg(l) for l in self.app_line]) * 1e-3

