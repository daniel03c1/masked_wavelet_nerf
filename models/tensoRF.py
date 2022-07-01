import torch.nn as nn
from .voxel_based import PREF as PREF_backbone
from .tensorBase import *


class TensorVM(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensorVM, self).__init__(aabb, gridSize, device, **kargs)

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


class PREF(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super().__init__(aabb, gridSize, device, **kargs)
        self.density = PREF_backbone((256, 256, 256), 2)
        self.appearance = PREF_backbone((256, 256, 256), 4)
        self.density_net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1))
        self.appearance_net = nn.Linear(4, 27)

    def get_optparam_groups(self, lr_init_spatialxyz=0.02,
                            lr_init_network=0.001):
        return {'params': self.parameters(), 'lr': lr_init_spatialxyz}

    def init_svd_volume(self, res, device):
        pass

    def compute_densityfeature(self, coords):
        return self.density_net(self.density(coords)*2/3).squeeze(-1)

    def compute_appfeature(self, coords):
        return self.appearance_net(self.appearance(coords)*2/3)

    def vectorDiffs(self, vector_comps):
        return torch.zeros(())

    def vector_comp_diffs(self):
        return self.vectorDiffs(self.density_line) \
             + self.vectorDiffs(self.app_line)

    def density_L1(self):
        return torch.zeros(())

    def TV_loss_density(self, reg):
        return self.density.compute_tv()

    def TV_loss_app(self, reg):
        return self.appearance.compute_tv()

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        return

    @torch.no_grad()
    def shrink(self, new_aabb):
        return


class TensorVMSplit(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensorVMSplit, self).__init__(aabb, gridSize, device, **kargs)

    def get_optparam_groups(self, lr_init_spatialxyz=0.02,
                            lr_init_network=0.001):
        grad_vars = [{'params': self.density_line, 'lr': lr_init_spatialxyz},
                     {'params': self.density_plane, 'lr': lr_init_spatialxyz},
                     {'params': self.app_line, 'lr': lr_init_spatialxyz},
                     {'params': self.app_plane, 'lr': lr_init_spatialxyz},
                     {'params': self.basis_mat.parameters(),
                      'lr':lr_init_network}]

        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(),
                           'lr':lr_init_network}]
        return grad_vars

    def init_svd_volume(self, res, device):
        self.density_plane, self.density_line = self.init_one_svd(
            self.density_n_comp, self.gridSize, 1e-5, device)
        self.app_plane, self.app_line = self.init_one_svd(
            self.app_n_comp, self.gridSize, 0.1, device)
        self.basis_mat = torch.nn.Linear(sum(self.app_n_comp), self.app_dim,
                                         bias=False).to(device)

    def init_one_svd(self, n_component, gridSize, scale, device):
        plane_coef = []
        line_coef = []

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component[i], gridSize[mat_id_1],
                                     gridSize[mat_id_0]))))
            line_coef.append(torch.nn.Parameter(
                0.01 * scale * torch.randn((1, n_component[i], gridSize[vec_id], 1))))

        return torch.nn.ParameterList(plane_coef).to(device), \
               torch.nn.ParameterList(line_coef).to(device)

    def to_plane_line(self, coords):
        # coord shape
        coord_plane = torch.stack(tuple(coords[..., ax] for ax in self.matMode))
        coord_plane = coord_plane.detach().view(3, -1, 1, 2)

        coord_line = torch.stack(tuple(coords[..., ax] for ax in self.vecMode))
        coord_line = torch.stack((torch.zeros_like(coord_line),
                                  coord_line), dim=-1)
        coord_line = coord_line.detach().view(3, -1, 1, 2)

        return coord_plane, coord_line

    def vec_sample(self, values, coordinates):
        # self.density_plane: [1, C, V, V] or [1, C, V, 1]
        # coordinate_plane : [1, B, 1, 2]
        # outputs: [1, C, B, 1]
        v_size = values.size(-2)
        device = coordinates.device
        if values.shape[-1] != 1: # plane
            outputs = F.grid_sample(values, coordinates, align_corners=True)
            return outputs.squeeze(0).squeeze(-1)
        else: # line
            outputs = F.grid_sample(values, coordinates, align_corners=True)
            return outputs.squeeze(0).squeeze(-1)

    def compute_densityfeature(self, coords):
        """
        INPUTS
            coords: [n_samples, 3] shaped matrix
        OUTPUTS
            sigma: [n_samples] shaped vector
        """
        # old version
        # plane + line basis
        coordinate_plane, coordinate_line = self.to_plane_line(coords)
        sigma_feature = torch.zeros((coords.shape[0], ), device=coords.device)

        for idx_plane in range(len(self.density_plane)):
            plane_coef_point = self.vec_sample(self.density_plane[idx_plane],
                                               coordinate_plane[[idx_plane]])
            line_coef_point = self.vec_sample(self.density_line[idx_plane],
                                              coordinate_line[[idx_plane]])

            sigma_feature = sigma_feature \
                          + torch.sum(plane_coef_point * line_coef_point, dim=0)

        return sigma_feature
        '''
        # new version
        sigma = 0
        for i, eq in enumerate(['CYX,CZ->ZYX', 'CZX,CY->ZYX', 'CZY,CX->ZYX']):
            sigma += torch.einsum(eq,
                                  self.density_plane[i].squeeze(0),
                                  self.density_line[i].squeeze(0).squeeze(-1))
        sigma = F.grid_sample(sigma.unsqueeze(0).unsqueeze(0), # [1, 1, ...]
                              coords.reshape(1, 1, 1, -1, 3),
                              padding_mode='reflection',
                              align_corners=True)
        return sigma.squeeze(0)
        '''

    def compute_appfeature(self, coords):
        # OLD
        # plane + line basis
        coordinate_plane, coordinate_line = self.to_plane_line(coords)
        plane_coef_point = []
        line_coef_point = []

        for idx_plane in range(len(self.app_plane)):
            plane_coef_point.append(self.vec_sample(
                self.app_plane[idx_plane], coordinate_plane[[idx_plane]]))
            line_coef_point.append(self.vec_sample(
                self.app_line[idx_plane], coordinate_line[[idx_plane]]))

        plane_coef_point = torch.cat(plane_coef_point)
        line_coef_point = torch.cat(line_coef_point)

        return self.basis_mat((plane_coef_point * line_coef_point).T)

    def vectorDiffs(self, vector_comps):
        total = 0

        for idx in range(len(vector_comps)):
            n_comp, n_size = vector_comps[idx].shape[1:-1]

            dotp = torch.matmul(vector_comps[idx].view(n_comp,n_size),
                                vector_comps[idx].view(n_comp,n_size) \
                                                 .transpose(-1,-2))
            non_diagonal = dotp.view(-1)[1:].view(n_comp-1, n_comp+1)[...,:-1]
            total = total + torch.mean(torch.abs(non_diagonal))
        return total

    def vector_comp_diffs(self):
        return self.vectorDiffs(self.density_line) \
             + self.vectorDiffs(self.app_line)

    def density_L1(self):
        return sum([p.abs().mean() for p in self.density_plane]) \
               + sum([l.abs().mean() for l in self.density_line])

    def TV_loss_density(self, reg):
        return sum([reg(p) for p in self.density_plane]) * 1e-2 \
               + sum([reg(l) for l in self.density_line]) * 1e-3

    def TV_loss_app(self, reg):
        return sum([reg(p) for p in self.app_plane]) * 1e-2 \
               + sum([reg(l) for l in self.app_line]) * 1e-3

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
        self.app_plane, self.app_line = self.up_sampling_VM(
            self.app_plane, self.app_line, res_target)
        self.density_plane, self.density_line = self.up_sampling_VM(
            self.density_plane, self.density_line, res_target)
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
            mode0, mode1 = self.matMode[i]
            self.density_plane[i] = torch.nn.Parameter(
                self.density_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )
            self.app_plane[i] = torch.nn.Parameter(
                self.app_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
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


class TensorCP(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensorCP, self).__init__(aabb, gridSize, device, **kargs)

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


class TensorVX(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensorVX, self).__init__(aabb, gridSize, device, **kargs)

    def init_svd_volume(self, res, *args, **kwargs):
        # pass # already initialized
        # density and appearance
        self.density = torch.nn.Parameter(
        # self.register_parameter('density', torch.nn.Parameter(
            1. * torch.randn((1, 1, *self.gridSize)), requires_grad=True)
        self.app = torch.nn.Parameter(
        # self.register_parameter('app', torch.nn.Parameter(
            0.1 * torch.randn((1, self.app_dim, *self.gridSize)),
            requires_grad=True)


    def get_optparam_groups(self, lr_init_spatialxyz=0.02,
                            lr_init_network=0.001):
        grad_vars = [{'params': self.density, 'lr': lr_init_spatialxyz},
                     {'params': self.app, 'lr': lr_init_spatialxyz}]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(),
                           'lr':lr_init_network}]
        return grad_vars

    def sample_feats(self, features, coords):
        '''
        features: [1, C, D, H, W]
        coords: [N, 3]
        '''
        sampled = F.grid_sample(features, coords.reshape(1, -1, 1, 1, 3),
                                align_corners=True)
        return sampled[0, :, 0, 0] # squeeze to [C, N]

    def compute_appfeature(self, coords):
        return self.sample_feats(self.app, coords)

    def compute_densityfeature(self, coords):
        return self.sample_feats(self.density, coords)

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        import pdb; pdb.set_trace() # what is res_target
        app_line_coef[i] = torch.nn.Parameter(
            F.interpolate(app_line_coef[i].data,
                          size=(res_target[vec_id], 1),
                          mode='bilinear', align_corners=True))

        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')

    @torch.no_grad()
    def shrink(self, new_aabb):
        import pdb; pdb.set_trace()
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l = (xyz_min - self.aabb[0]) / self.units
        t_l = torch.round(t_l).long()
        b_r = (xyz_max - self.aabb[0]) / self.units
        b_r = torch.round(b_r).long() + 1
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
        return reg(self.density)

    def TV_loss_app(self, reg):
        return reg(self.app)

