import torch
import torch.nn as nn
import torch.nn.functional as F

from .tensorBase import *
from .dwt import forward, inverse


def min_max_quantize(inputs, bits):
    if bits == 32:
        return inputs

    scale = torch.amax(torch.abs(inputs)).clamp(min=1e-6)
    n = float(2**(bits-1) - 1)
    out = torch.round(torch.abs(inputs / scale) * n) / n * scale
    rounded = out * torch.sign(inputs)

    return (rounded - inputs).detach() + inputs


class TensorVMSplit(TensorBase):
    def __init__(self, aabb, gridSize, device,
                 use_mask=False, use_dwt=False, dwt_level=2, **kargs):
        super(TensorVMSplit, self).__init__(aabb, gridSize, device, **kargs)
        self.use_mask = use_mask
        self.use_dwt = use_dwt
        self.dwt_level = dwt_level

        if use_mask:
            self.init_mask()

    def get_kwargs(self):
        return {
            'aabb': self.aabb,
            'gridSize':self.gridSize.tolist(),
            'density_n_comp': self.density_n_comp,
            'appearance_n_comp': self.app_n_comp,
            'app_dim': self.app_dim,

            'density_shift': self.density_shift,
            'alphaMask_thres': self.alphaMask_thres,
            'distance_scale': self.distance_scale,
            'rayMarch_weight_thres': self.rayMarch_weight_thres,
            'fea2denseAct': self.fea2denseAct,

            'near_far': self.near_far,
            'step_ratio': self.step_ratio,

            'shadingMode': self.shadingMode,
            'pos_pe': self.pos_pe,
            'view_pe': self.view_pe,
            'fea_pe': self.fea_pe,
            'featureC': self.featureC,

            'grid_bit': self.grid_bit,
            'use_mask': self.use_mask,
            'use_dwt': self.use_dwt,
            'dwt_level': self.dwt_level
        }

    def init_svd_volume(self, res, device):
        self.density_plane, self.density_line = self.init_one_svd(
            self.density_n_comp, self.gridSize, 0.1, device)
        self.app_plane, self.app_line = self.init_one_svd(
            self.app_n_comp, self.gridSize, 0.1, device)
        self.basis_mat = nn.Linear(
            sum(self.app_n_comp), self.app_dim, bias=False).to(device)

    @torch.no_grad()
    def init_mask(self):
        self.density_plane_mask = nn.ParameterList(
            [nn.Parameter(torch.ones_like(self.density_plane[i]))
             for i in range(3)])
        self.density_line_mask = nn.ParameterList(
            [nn.Parameter(torch.ones_like(self.density_line[i]))
             for i in range(3)])
        self.app_plane_mask = nn.ParameterList(
            [nn.Parameter(torch.ones_like(self.app_plane[i]))
             for i in range(3)])
        self.app_line_mask = nn.ParameterList(
            [nn.Parameter(torch.ones_like(self.app_line[i]))
             for i in range(3)])

    def init_one_svd(self, n_component, gridSize, scale, device):
        plane_coef, line_coef = [], []

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(nn.Parameter(
                scale * torch.randn((1, n_component[i], gridSize[mat_id_1],
                                     gridSize[mat_id_0]))))
            line_coef.append(nn.Parameter(
                scale * torch.randn((1, n_component[i], gridSize[vec_id], 1))))

        return (nn.ParameterList(plane_coef).to(device),
                nn.ParameterList(line_coef).to(device))

    def get_optparam_groups(self, lr0=0.02, lr1=0.001):
        grad_vars = [{'params': self.density_line, 'lr': lr0},
                     {'params': self.density_plane, 'lr': lr0},
                     {'params': self.app_line, 'lr': lr0},
                     {'params': self.app_plane, 'lr': lr0},
                     {'params': self.basis_mat.parameters(), 'lr':lr1}]

        if isinstance(self.renderModule, nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr1}]

        if self.use_mask:
            grad_vars += [{'params': self.density_plane_mask, 'lr': lr0},
                          {'params': self.density_line_mask, 'lr': lr0},
                          {'params': self.app_plane_mask, 'lr': lr0},
                          {'params': self.app_line_mask, 'lr': lr0}]

        return grad_vars

    def compute_densityfeature(self, points):
        # plane + line basis
        # [3, B, 1, 2]
        coordinate_plane = points[..., self.matMode].transpose(0, -2) \
                                                    .view(3, -1, 1, 2)
        coordinate_line = points[..., self.vecMode, None].transpose(0, -2)
        coordinate_line = F.pad(coordinate_line, (1, 0)).reshape(3, -1, 1, 2)

        sigma_feature = torch.zeros((points.shape[0],), device=points.device)

        for idx in range(len(self.density_plane)):
            plane = min_max_quantize(self.density_plane[idx], self.grid_bit)
            line = min_max_quantize(self.density_line[idx], self.grid_bit)

            if self.use_mask:
                mask = torch.sigmoid(self.density_plane_mask[idx])
                plane = (plane * (mask >= 0.5) - plane * mask).detach() \
                      + plane * mask
                mask = torch.sigmoid(self.density_line_mask[idx])
                line = (line * (mask >= 0.5) - line * mask).detach() \
                     + line * mask

            if self.use_dwt:
                plane = inverse(plane, self.dwt_level)

            plane_coef_point = F.grid_sample(
                plane, coordinate_plane[[idx]],
                align_corners=True).view(-1, *points.shape[:1])

            line_coef_point = F.grid_sample(
                line, coordinate_line[[idx]],
                align_corners=True).view(-1, *points.shape[:1])

            sigma_feature += torch.sum(plane_coef_point*line_coef_point, dim=0)

        return sigma_feature

    def compute_appfeature(self, points):
        # plane + line basis
        # [3, B, 1, 2]
        coordinate_plane = points[..., self.matMode].transpose(0, -2) \
                                                    .view(3, -1, 1, 2)
        coordinate_line = points[..., self.vecMode, None].transpose(0, -2)
        coordinate_line = F.pad(coordinate_line, (1, 0)).reshape(3, -1, 1, 2)

        plane_coef_point, line_coef_point = [], []
        for idx in range(len(self.app_plane)):
            plane = min_max_quantize(self.app_plane[idx], self.grid_bit)
            line = min_max_quantize(self.app_line[idx], self.grid_bit)

            if self.use_mask:
                mask = torch.sigmoid(self.app_plane_mask[idx])
                plane = (plane * (mask >= 0.5) - plane * mask).detach() \
                      + plane * mask
                mask = torch.sigmoid(self.app_line_mask[idx])
                line = (line * (mask >= 0.5) - line * mask).detach() \
                     + line * mask

            if self.use_dwt:
                plane = inverse(plane, self.dwt_level)

            plane_coef_point.append(F.grid_sample(
                plane, coordinate_plane[[idx]],
                align_corners=True).view(-1, *points.shape[:1]))
            line_coef_point.append(F.grid_sample(
                line, coordinate_line[[idx]],
                align_corners=True).view(-1, *points.shape[:1]))

        plane_coef_point = torch.cat(plane_coef_point)
        line_coef_point = torch.cat(line_coef_point)

        return self.basis_mat((plane_coef_point * line_coef_point).T)

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.app_plane, self.app_line = self.up_sampling_VM(
            self.app_plane, self.app_line, res_target)
        self.density_plane, self.density_line = self.up_sampling_VM(
            self.density_plane, self.density_line, res_target)

        if self.use_mask:
            self.app_plane_mask, self.app_line_mask = self.up_sampling_VM(
                self.app_plane_mask, self.app_line_mask, res_target)
            self.density_plane_mask, self.density_line_mask = \
                self.up_sampling_VM(self.density_plane_mask,
                                    self.density_line_mask, res_target)

        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')

    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]

            if self.use_dwt:
                plane_coef[i].set_(inverse(plane_coef[i], self.dwt_level))

            plane_coef[i] = nn.Parameter(
                F.interpolate(plane_coef[i].data,
                              size=(res_target[mat_id_1], res_target[mat_id_0]),
                              mode='bilinear', align_corners=True))
            line_coef[i] = nn.Parameter(
                F.interpolate(line_coef[i].data, size=(res_target[vec_id], 1),
                mode='bilinear', align_corners=True))

            if self.use_dwt:
                plane_coef[i].set_(forward(plane_coef[i], self.dwt_level))

        return plane_coef, line_coef

    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        unit = 16 # unit for DWT

        for i in range(len(self.vecMode)):
            # Lines
            mode0 = self.vecMode[i]
            steps = (new_aabb[1][mode0]-new_aabb[0][mode0]) / self.units[mode0]
            steps = int(steps / unit) * unit

            grid = torch.linspace(new_aabb[0][mode0], new_aabb[1][mode0],
                                  steps).to(self.density_line[i].device)
            grid = F.pad(grid.reshape(1, -1, 1, 1), (0, 1))

            self.density_line[i] = nn.Parameter(
                F.grid_sample(self.density_line[i], grid, align_corners=True))
            self.app_line[i] = nn.Parameter(
                F.grid_sample(self.app_line[i], grid, align_corners=True))

            # Planes
            mode0, mode1 = self.matMode[i]
            if self.use_dwt:
                self.density_plane[i].set_(inverse(self.density_plane[i],
                                                   self.dwt_level))
                self.app_plane[i].set_(inverse(self.app_plane[i],
                                               self.dwt_level))

            steps = (new_aabb[1][mode0]-new_aabb[0][mode0]) / self.units[mode0]
            steps = int(steps / unit) * unit
            grid0 = torch.linspace(new_aabb[0][mode0], new_aabb[1][mode0],
                                   steps).to(self.density_line[i].device)

            steps = (new_aabb[1][mode1]-new_aabb[0][mode1]) / self.units[mode1]
            steps = int(steps / unit) * unit
            grid1 = torch.linspace(new_aabb[0][mode1], new_aabb[1][mode1],
                                   steps).to(self.density_line[i].device)
            grid = torch.stack(torch.meshgrid(grid0, grid1), -1).unsqueeze(0)

            self.density_plane[i] = nn.Parameter(
                F.grid_sample(self.density_plane[i], grid, align_corners=True))
            self.app_plane[i] = nn.Parameter(
                F.grid_sample(self.app_plane[i], grid, align_corners=True))

            if self.use_dwt:
                self.density_plane[i].set_(forward(self.density_plane[i],
                                                   self.dwt_level))
                self.app_plane[i].set_(forward(self.app_plane[i],
                                               self.dwt_level))

        self.aabb = new_aabb
        self.update_stepSize(
            tuple(reversed([p.shape[-2] for p in self.density_line])))

        if self.use_mask:
            self.init_mask()

    def vectorDiffs(self, vector_comps):
        breakpoint()
        total = 0
        for idx in range(len(vector_comps)):
            n_comp, n_size = vector_comps[idx].shape[1:-1]

            dotp = torch.matmul(
                vector_comps[idx].view(n_comp, n_size),
                vector_comps[idx].view(n_comp ,n_size).transpose(-1, -2))
            non_diagonal = dotp.view(-1)[1:].view(n_comp-1, n_comp+1)[...,:-1]
            total = total + torch.mean(torch.abs(non_diagonal))
        return total

    def vector_comp_diffs(self):
        return (self.vectorDiffs(self.density_line)
                + self.vectorDiffs(self.app_line))

    def density_L1(self):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + torch.mean(torch.abs(self.density_plane[idx])) \
                  + torch.mean(torch.abs(self.density_line[idx]))
        return total

    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + reg(self.density_plane[idx]) * 1e-2
        return total

    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.app_plane)):
            total = total + reg(self.app_plane[idx]) * 1e-2
        return total


class TensorCP(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensorCP, self).__init__(aabb, gridSize, device, **kargs)

    def init_svd_volume(self, res, device):
        self.density_line = self.init_one_svd(self.density_n_comp[0], self.gridSize, 0.2, device)
        self.app_line = self.init_one_svd(self.app_n_comp[0], self.gridSize, 0.2, device)
        self.basis_mat = nn.Linear(self.app_n_comp[0], self.app_dim, bias=False).to(device)

    def init_one_svd(self, n_component, gridSize, scale, device):
        line_coef = []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            line_coef.append(
                nn.Parameter(scale * torch.randn((1, n_component, gridSize[vec_id], 1))))
        return nn.ParameterList(line_coef).to(device)

    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [{'params': self.density_line, 'lr': lr_init_spatialxyz},
                     {'params': self.app_line, 'lr': lr_init_spatialxyz},
                     {'params': self.basis_mat.parameters(), 'lr':lr_init_network}]
        if isinstance(self.renderModule, nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network}]
        return grad_vars

    def compute_densityfeature(self, xyz_sampled):
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        shape = xyz_sampled.shape[:1]
        line_coef_point = 1
        for i in range(3):
            line_coef_point = line_coef_point * F.grid_sample(
                min_max_quantize(self.density_line[i], self.grid_bit),
                coordinate_line[[i]], align_corners=True).view(-1, *shape)

        sigma_feature = torch.sum(line_coef_point, dim=0)
        return sigma_feature

    def compute_appfeature(self, xyz_sampled):
        coordinate_line = torch.stack(
            (xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        shape = xyz_sampled.shape[:1]
        line_coef_point = 1
        for i in range(3):
            line_coef_point = line_coef_point * F.grid_sample(
                min_max_quantize(self.app_line[i], self.grid_bit),
                coordinate_line[[i]], align_corners=True).view(-1, *shape)

        return self.basis_mat(line_coef_point.T)

    @torch.no_grad()
    def up_sampling_Vector(self, density_line_coef, app_line_coef, res_target):
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            density_line_coef[i] = nn.Parameter(
                F.interpolate(density_line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))
            app_line_coef[i] = nn.Parameter(
                F.interpolate(app_line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))

        return density_line_coef, app_line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.density_line, self.app_line = self.up_sampling_Vector(self.density_line, self.app_line, res_target)

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
            self.density_line[i] = nn.Parameter(
                self.density_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            self.app_line[i] = nn.Parameter(
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
        total = 0
        for idx in range(len(self.density_line)):
            total = total + torch.mean(torch.abs(self.density_line[idx]))
        return total

    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.density_line)):
            total = total + reg(self.density_line[idx]) * 1e-3
        return total

    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.app_line)):
            total = total + reg(self.app_line[idx]) * 1e-3
        return total

