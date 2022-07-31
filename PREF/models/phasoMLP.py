import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

from .cosine_transform import dctn, idctn
from .phasoBase import *
from .utils import positional_encoding
from .utils_fft import (
    getMask_fft, getMask, grid_sample, grid_sample_cmplx, irfft, rfft,
    batch_irfft
)


class CPhasoMLP(PhasorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super().__init__(aabb, gridSize, device, **kargs)

    def init_phasor_volume(self, res, device):
        """ initialize volume """
        self.axis = [torch.tensor([0.]+[2**i for i in torch.arange(d-1)],
                                  device=self.device)
                     for d in self.app_num_comp]

        self.ktraj = self.compute_ktraj(self.axis, self.gridSize)
        self.ktraj_den = self.compute_ktraj(
            self.axis, (self.gridSize * self.den_scale).long())

        self.den = nn.ParameterList(
            self.init_(self.den_num_comp,
                       (self.gridSize * self.den_scale).long(),
                       ksize=self.den_ksize))
        self.app = nn.ParameterList(
            self.init_(self.app_num_comp,
                       (self.gridSize * self.app_scale).long(),
                       ksize=self.app_ksize))

        # mask
        self.den_mask = nn.ParameterList(
            # [nn.Parameter(torch.zeros_like(d[0, 0])) for d in self.den])
            [nn.Parameter(torch.zeros_like(d)) for d in self.den])
        self.app_mask = nn.ParameterList(
            # [nn.Parameter(torch.zeros_like(d[0, 0])) for d in self.app])
            [nn.Parameter(torch.zeros_like(d)) for d in self.app])

        den_outdim = self.den_ksize * 3
        app_outdim = self.app_ksize * 3

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

    @torch.no_grad()
    def compute_ktraj(self, axis, res):
        ktraj2d = [torch.arange(i, dtype=torch.float32, device=self.device)
                   for i in res]
        ktraj1d = [torch.arange(ax, dtype=torch.float32, device=self.device)
                   if type(ax) == int else ax for ax in axis]

        ktrajx = torch.stack(
            torch.meshgrid([ktraj1d[0], ktraj2d[1], ktraj2d[2]]), dim=-1)
        ktrajy = torch.stack(
            torch.meshgrid([ktraj2d[0], ktraj1d[1], ktraj2d[2]]), dim=-1)
        ktrajz = torch.stack(
            torch.meshgrid([ktraj2d[0], ktraj2d[1], ktraj1d[2]]), dim=-1)

        return [ktrajx, ktrajy, ktrajz]

    def init_(self, axis, res, ksize=1, init_scale=1):
        Nx, Ny, Nz = res
        d1, d2, d3 = axis

        fx = torch.zeros(1, ksize, d1, Ny, Nz, dtype=torch.float32,
                         device=self.device)
        fy = torch.zeros(1, ksize, Nx, d2, Nz, dtype=torch.float32,
                         device=self.device)
        fz = torch.zeros(1, ksize, Nx, Ny, d3, dtype=torch.float32,
                         device=self.device)

        return [nn.Parameter(fx), nn.Parameter(fy), nn.Parameter(fz)]

    def compute_densityfeature(self, xyz_sampled):
        # sigma_feature = self.compute_fft(self.density, xyz_sampled)
        sigma_feature = self.compute_fft(self.density, xyz_sampled,
                                         self.den_mask)
        return self.mlp(sigma_feature.T).T

    def feature2density(self, density_features):
        if self.fea2denseAct == "softplus":
            return F.softplus(density_features+self.density_shift,
                              beta=self.softplus_beta)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features)

    def compute_appfeature(self, xyz_sampled):
        # app_points = self.compute_fft(self.appearance, xyz_sampled)
        app_points = self.compute_fft(self.appearance, xyz_sampled,
                                      self.app_mask)
        if self.app_aug == 'flip':
            aug = self.compute_fft(self.appearance, xyz_sampled.flip(-1))
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

        return self.basis_mat(app_points.T)

    def compute_fft(self, features, xyz_sampled, mask=None):
        # this is fast because of 2d transform and matrix multiplication.
        # (N*N) logN d + Nsamples * d*d + 3 * Nsamples
        Fx, Fy, Fz = features
        d1, d2, d3 = Fx.shape[2], Fy.shape[3], Fz.shape[4]
        Nx, Ny, Nz = Fy.shape[2], Fz.shape[3], Fx.shape[4]
        kx, ky, kz = self.axis
        kx, ky, kz = kx[:d1], ky[:d2], kz[:d3]
        xs, ys, zs = xyz_sampled.chunk(3, dim=-1)

        '''
        # [1, F, N, N, N]
        pad_size = [(8 - x%8) % 8 for x in [Nx, Ny, Nz]]

        Fx = F.pad(Fx, (0, pad_size[-1], 0, pad_size[-2]))
        Fx = Fx.reshape(*Fx.shape[:-2], Fx.shape[-2]//8, 8,
                        Fx.shape[-1]//8, 8)
        Fx = idctn(Fx, axes=(-3, -1))
        Fx = Fx.flatten(-2, -1).flatten(-3, -2)[..., :Ny, :Nz]

        Fy = F.pad(Fy, (0, pad_size[-1], 0, 0, 0, pad_size[-3]))
        Fy = Fy.reshape(*Fy.shape[:-3], Fy.shape[-3]//8, 8, Fy.shape[-2],
                        Fy.shape[-1]//8, 8)
        Fy = idctn(Fy, axes=(-4, -1))
        Fy = Fy.flatten(-2, -1).flatten(-4, -3)[..., :Nx, :, :Nz]

        Fz = F.pad(Fz, (0, 0, 0, pad_size[-2], 0, pad_size[-3]))
        Fz = Fz.reshape(*Fz.shape[:-3], Fz.shape[-3]//8, 8,
                        Fz.shape[-2]//8, 8, Fz.shape[-1])
        Fz = idctn(Fz, axes=(-4, -2))
        Fz = Fz.flatten(-3, -2).flatten(-4, -3)[..., :Nx, :Ny, :]

        assert tuple(Fx.shape[-3:]) == (d1, Ny, Nz)
        assert tuple(Fy.shape[-3:]) == (Nx, d2, Nz)
        assert tuple(Fz.shape[-3:]) == (Nx, Ny, d3)
        '''

        if mask is not None:
            mx, my, mz = mask
            mx = torch.sigmoid(mx)
            my = torch.sigmoid(my)
            mz = torch.sigmoid(mz)
            Fx = (Fx * (mx >= 0.5) - Fx * mx).detach() + Fx * mx
            Fy = (Fy * (my >= 0.5) - Fy * my).detach() + Fy * my
            Fz = (Fz * (mz >= 0.5) - Fz * mz).detach() + Fz * mz
            '''
            # Fx = Fx * (((mx >= 0.5).float() - mx).detach() + mx)
            # Fy = Fy * (((my >= 0.5).float() - my).detach() + my)
            # Fz = Fz * (((mz >= 0.5).float() - mz).detach() + mz)
            Fx = (Fx * (mx >= 0.5) - Fx - Fx * mx).detach() \
               + Fx + Fx.detach() * mx
            Fy = (Fy * (my >= 0.5) - Fy - Fy * my).detach() \
               + Fy + Fy.detach() * my
            Fz = (Fz * (mz >= 0.5) - Fz - Fz * mz).detach() \
               + Fz + Fz.detach() * mz
            '''

        Fx = idctn(Fx, axes=(3, 4))
        Fy = idctn(Fy, axes=(2, 4))
        Fz = idctn(Fz, axes=(2, 3))

        fx = grid_sample(
            Fx.transpose(3, 3).flatten(1,2),
            torch.stack([zs, ys], dim=-1)[None]).reshape(Fx.shape[1], Fx.shape[2], -1)
        fy = grid_sample(
            Fy.transpose(2, 3).flatten(1,2),
            torch.stack([zs, xs], dim=-1)[None]).reshape(Fy.shape[1], Fy.shape[3], -1)
        fz = grid_sample(
            Fz.transpose(2, 4).flatten(1,2),
            torch.stack([xs, ys], dim=-1)[None]).reshape(Fz.shape[1], Fz.shape[4], -1)

        fx = fx.reshape(-1, xyz_sampled.shape[0])
        fy = fy.reshape(-1, xyz_sampled.shape[0])
        fz = fz.reshape(-1, xyz_sampled.shape[0])

        return torch.cat([fx, fy, fz], 0) # fxx+fyy+fzz

    def Parseval_Loss(self):
        loss = 0
        for f, w in zip(self.density, self.ktraj_den):
            feat = torch.pi * f[..., None] * w.reshape(1, 1, *f.shape[2:], -1)
            loss = loss + feat.square().mean()

        # for f, w in zip(self.appearance, self.ktraj):
        #     feat = torch.pi * f[..., None] * w.reshape(1, 1, *f.shape[2:], -1)
        #     loss += feat.square().mean()
        return loss

    def compute_normal(self, xyz_sampled):
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

    # used in train.py
    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        res_den = [math.ceil(n * self.den_scale) for n in res_target]
        res_app = [math.ceil(n * self.app_scale) for n in res_target]

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

        self.print_size()
        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')

    def upsample_feats(self, features, res_target, update_dd=False):
        Tx, Ty, Tz = res_target
        Fx, Fy, Fz = features
        d1, d2, d3 = Fx.shape[-3], Fy.shape[-2], Fz.shape[-1]
        Nx, Ny, Nz = Fy.shape[-3], Fz.shape[-2], Fx.shape[-1]

        return F.pad(Fx, (0, Tz-Nz, 0, Ty-Ny, 0, 0)), \
               F.pad(Fy, (0, Tz-Nz, 0, 0, 0, Tx-Nx)), \
               F.pad(Fz, (0, 0, 0, Ty-Ny, 0, Tx-Nx))
        '''

        pad_size = [(8 - x%8) % 8 for x in [Nx, Ny, Nz]]

        Fx = F.pad(Fx, (0, pad_size[-1], 0, pad_size[-2]))
        Fx = Fx.reshape(*Fx.shape[:-2], Fx.shape[-2]//8, 8,
                        Fx.shape[-1]//8, 8)
        Fx = idctn(Fx, axes=(-3, -1))
        Fx = Fx.flatten(-2, -1).flatten(-3, -2)[..., :Ny, :Nz]

        Fy = F.pad(Fy, (0, pad_size[-1], 0, 0, 0, pad_size[-3]))
        Fy = Fy.reshape(*Fy.shape[:-3], Fy.shape[-3]//8, 8, Fy.shape[-2],
                        Fy.shape[-1]//8, 8)
        Fy = idctn(Fy, axes=(-4, -1))
        Fy = Fy.flatten(-2, -1).flatten(-4, -3)[..., :Nx, :, :Nz]

        Fz = F.pad(Fz, (0, 0, 0, pad_size[-2], 0, pad_size[-3]))
        Fz = Fz.reshape(*Fz.shape[:-3], Fz.shape[-3]//8, 8,
                        Fz.shape[-2]//8, 8, Fz.shape[-1])
        Fz = idctn(Fz, axes=(-4, -2))
        Fz = Fz.flatten(-3, -2).flatten(-4, -3)[..., :Nx, :Ny, :]

        # interpolate
        Fx = F.interpolate(Fx, (d1, Ty, Tz), mode='trilinear',
                           align_corners=True)
        Fy = F.interpolate(Fy, (Tx, d2, Tz), mode='trilinear',
                           align_corners=True)
        Fz = F.interpolate(Fz, (Tx, Ty, d3), mode='trilinear',
                           align_corners=True)

        # dctn
        pad_size = [(8 - x%8) % 8 for x in [Tx, Ty, Tz]]

        Fx = F.pad(Fx, (0, pad_size[-1], 0, pad_size[-2]))
        Fx = Fx.reshape(*Fx.shape[:-2], Fx.shape[-2]//8, 8,
                        Fx.shape[-1]//8, 8)
        Fx = dctn(Fx, axes=(-3, -1))
        Fx = Fx.flatten(-2, -1).flatten(-3, -2)[..., :Ty, :Tz]

        Fy = F.pad(Fy, (0, pad_size[-1], 0, 0, 0, pad_size[-3]))
        Fy = Fy.reshape(*Fy.shape[:-3], Fy.shape[-3]//8, 8, Fy.shape[-2],
                        Fy.shape[-1]//8, 8)
        Fy = dctn(Fy, axes=(-4, -1))
        Fy = Fy.flatten(-2, -1).flatten(-4, -3)[..., :Tx, :, :Tz]

        Fz = F.pad(Fz, (0, 0, 0, pad_size[-2], 0, pad_size[-3]))
        Fz = Fz.reshape(*Fz.shape[:-3], Fz.shape[-3]//8, 8,
                        Fz.shape[-2]//8, 8, Fz.shape[-1])
        Fz = dctn(Fz, axes=(-4, -2))
        Fz = Fz.flatten(-3, -2).flatten(-4, -3)[..., :Tx, :Ty, :]

        return Fx, Fy, Fz
        '''

    def update_stepSize(self, gridSize):
        self.ktraj = self.compute_ktraj(self.axis, gridSize)
        self.ktraj_den = self.compute_ktraj(
            self.axis, [math.ceil(n * self.den_scale) for n in gridSize])
        print("dimensions largest ", [torch.max(ax).item() for ax in self.axis])
        return super(CPhasoMLP, self).update_stepSize(gridSize)

    def print_size(self):
        print(self)
        print(f' ==> Actual Model Size {np.sum([v.numel() * v.element_size() for k, v in self.named_parameters()])/2**20} MB')
        for k,v in self.named_parameters():
            print(f'Model Size ({k}) : '
                  f'{v.numel() * v.element_size()/2**20:.4f} MB')

    def get_optparam_groups(self, lr_init_spatialxyz=0.02,
                            lr_init_network=0.001):
        grad_vars = [{'params': self.den, 'lr': lr_init_spatialxyz},
                     {'params': self.app, 'lr': lr_init_spatialxyz},
                     {'params': self.den_mask, 'lr': lr_init_spatialxyz},
                     {'params': self.app_mask, 'lr': lr_init_spatialxyz},
                     {'params': self.basis_mat.parameters(),
                      'lr':lr_init_network},
                     {'params': self.mlp.parameters(),
                      'lr': lr_init_network}]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(),
                           'lr':lr_init_network}]
        return grad_vars

    @property
    def alpha(self):
        # avoid negative value
        return F.softplus(self.alpha_params, beta=10, threshold=1e-4)

    @property
    def density(self):
        return [self.alpha * den for den in self.den]

    @property
    def appearance(self):
        return [app * self.beta for app in self.app]

    def compute_gaussian(self, variance, mode, ktraj=None):
        breakpoint()
        if mode== 'none':
            return [1., 1., 1.]
        if mode == 'prod':
            if ktraj is None:
                ktraj = self.ktraj
            ktraj = [k / g for k,g in zip(ktraj, self.gridSize)]
            gauss = [torch.exp((-2*(np.pi*kk)**2*variance[None]).sum(-1)).reshape(1,1,*kk.shape[:-1]) for kk in ktraj]
        else:
            raise ValueError(f'no mode named {mode}')
        return gauss


class PhasoMLP(PhasorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super().__init__(aabb, gridSize, device, **kargs)

    def init_phasor_volume(self, res, device):
        """ initialize volume """
        # using a dilated phasor volume 
        self.axis = [
            torch.tensor([0.]+[2**i for i in torch.arange(d-1)],
                         device=self.device)
            for d in self.app_num_comp]

        self.ktraj = self.compute_ktraj(self.axis, self.gridSize)
        self.ktraj_den = self.compute_ktraj(
            self.axis, (self.gridSize * self.den_scale).long())

        self.den = torch.nn.ParameterList(
            self.init_(self.den_num_comp,
                       (self.gridSize * self.den_scale).long(),
                       ksize=self.den_ksize))
        self.app = torch.nn.ParameterList(
            self.init_(self.app_num_comp,
                       (self.gridSize * self.app_scale).long(),
                       ksize=self.app_ksize))

        den_outdim = self.den_ksize
        app_outdim = self.app_ksize

        # duplicate real and image parts
        if self.den_num_comp == [1, 1, 1]:
            den_outdim = den_outdim * 2
        if self.app_num_comp == [1, 1, 1]:
            app_outdim = app_outdim * 2

        if self.app_aug == 'flip':
            app_outdim = app_outdim * 2
        elif self.app_aug == 'normal':
            app_outdim = app_outdim + 3
        elif self.app_aug == 'flip++':
            app_outdim = app_outdim * 4

        self.basis_mat = torch.nn.Linear(
            app_outdim, self.app_dim, bias=False).to(device)
        self.alpha_params = torch.nn.Parameter(
            torch.tensor([self.alpha_init]).to(device))
        self.beta = torch.nn.Parameter(
            torch.tensor([self.alpha_init]).to(device))
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(den_outdim, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(64, 1)
        ).to(device)
        print(self)

    @torch.no_grad()
    def compute_ktraj(self, axis, res):
        ktraj2d = [torch.fft.fftfreq(i, 1/i).to(self.device)
                   for i in res]
        ktraj1d = [torch.arange(ax).to(torch.float).to(self.device)
                   if type(ax) == int else ax for ax in axis]

        ktrajx = torch.stack(
            torch.meshgrid([ktraj1d[0], ktraj2d[1], ktraj2d[2]]),
            dim=-1)
        ktrajy = torch.stack(
            torch.meshgrid([ktraj2d[0], ktraj1d[1], ktraj2d[2]]),
            dim=-1)
        ktrajz = torch.stack(
            torch.meshgrid([ktraj2d[0], ktraj2d[1], ktraj1d[2]]),
            dim=-1)

        return [ktrajx, ktrajy, ktrajz]

    def init_(self, axis, res, ksize=1, init_scale=1):
        Nx, Ny, Nz = res
        d1, d2, d3 = axis

        fx = torch.zeros(1, ksize, d1, Ny, Nz, dtype=torch.complex64,
                         device=self.device)
        fy = torch.zeros(1, ksize, Nx, d2, Nz, dtype=torch.complex64,
                         device=self.device)
        fz = torch.zeros(1, ksize, Nx, Ny, d3, dtype=torch.complex64,
                         device=self.device)

        return [torch.nn.Parameter(fx), torch.nn.Parameter(fy),
                torch.nn.Parameter(fz)]

    def compute_densityfeature(self, xyz_sampled):
        sigma_feature = self.compute_fft(self.density, xyz_sampled,
                                         interp=False)
        return self.mlp(sigma_feature.T).T

    def feature2density(self, density_features):
        if self.fea2denseAct == "softplus":
            return F.softplus(density_features+self.density_shift,
                              beta=self.softplus_beta)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features)

    def compute_appfeature(self, xyz_sampled):
        app_points = self.compute_fft(self.appearance, xyz_sampled,
                                      interp=False)
        if self.app_aug == 'flip':
            aug = self.compute_fft(self.appearance,
                                   xyz_sampled.flip(-1), interp=False)
            app_points = torch.cat([app_points, aug], dim=0)
        elif self.app_aug == 'normal':
            aug = self.compute_normal(xyz_sampled)
            app_points = torch.cat([app_points, aug], dim=0)
        elif self.app_aug == 'flip++':
            aug1 = self.compute_fft(self.appearance, xyz_sampled.flip(-1), interp=False)
            aug2 = self.compute_fft(self.appearance, -xyz_sampled, interp=False)
            aug3 = self.compute_fft(self.appearance, -xyz_sampled.flip(-1), interp=False)
            app_points = torch.cat([app_points, aug1, aug2, aug3], dim=0)
        elif self.app_aug != 'none':
            raise NotImplementedError(f'{self.app_aug} not implemented')

        return self.basis_mat(app_points.T)

    def compute_fft(self, features, xyz_sampled, interp=True):
        if interp:
            # Nx: num of samples
            # using interpolation to compute fft = (N*N) log (N) d  + (N*N*d*d) + Nsamples 
            fx, fy, fz = self.compute_spatial_volume(features)
            volume = fx+fy+fz
            points = F.grid_sample(volume, xyz_sampled[None, None, None].flip(-1), align_corners=True).view(-1, *xyz_sampled.shape[:1],)
            # this is somewhat expensive when the xyz_samples is few and a 3D volume stills need computed
        else:
            # this is fast because we did 2d transform and matrix multiplication . (N*N) logN d + Nsamples * d*d + 3 * Nsamples 
            Fx, Fy, Fz = features
            d1, d2, d3 = Fx.shape[2], Fy.shape[3], Fz.shape[4]
            Nx, Ny, Nz = Fy.shape[2], Fz.shape[3], Fx.shape[4]
            kx, ky, kz = self.axis
            kx, ky, kz = kx[:d1], ky[:d2], kz[:d3]
            xs, ys, zs = xyz_sampled.chunk(3, dim=-1)

            Fx = torch.fft.ifftn(Fx, dim=(3,4), norm='forward')
            Fy = torch.fft.ifftn(Fy, dim=(2,4), norm='forward')
            Fz = torch.fft.ifftn(Fz, dim=(2,3), norm='forward')

            fx = grid_sample_cmplx(Fx.transpose(3,3).flatten(1,2), torch.stack([zs, ys], dim=-1)[None]).reshape(Fx.shape[1], Fx.shape[2], -1)
            fy = grid_sample_cmplx(Fy.transpose(2,3).flatten(1,2), torch.stack([zs, xs], dim=-1)[None]).reshape(Fy.shape[1], Fy.shape[3], -1)
            fz = grid_sample_cmplx(Fz.transpose(2,4).flatten(1,2), torch.stack([xs, ys], dim=-1)[None]).reshape(Fz.shape[1], Fz.shape[4], -1)

            if d1 == 1:
                # when d==1 do not need transform and split complex into two channel
                fxx = torch.concat([fx.real, fx.imag]).reshape(-1, xyz_sampled.shape[0])
                fyy = torch.concat([fy.real, fy.imag]).reshape(-1, xyz_sampled.shape[0])
                fzz = torch.concat([fz.real, fz.imag]).reshape(-1, xyz_sampled.shape[0])
            else:
                fxx = batch_irfft(fx, xs, kx, Nx)
                fyy = batch_irfft(fy, ys, ky, Ny)
                fzz = batch_irfft(fz, zs, kz, Nz)

            return fxx+fyy+fzz

        return points

    def compute_spatial_volume(self, features):
        Fx, Fy, Fz = features
        Nx, Ny, Nz = Fy.shape[2], Fz.shape[3], Fx.shape[4]
        xx, yy, zz = [torch.linspace(0, 1, N).to(self.device) for N in [Nx, Ny, Nz]]
        d1, d2, d3 = Fx.shape[2], Fy.shape[3], Fz.shape[4]
        kx, ky, kz = self.axis
        kx, ky, kz = kx[:d1], ky[:d2], kz[:d3]
        fx = irfft(torch.fft.ifftn(Fx, dim=(3,4), norm='forward'), xx, ff=kx, T=Nx, dim=2)
        fy = irfft(torch.fft.ifftn(Fy, dim=(2,4), norm='forward'), yy, ff=ky, T=Ny, dim=3)
        fz = irfft(torch.fft.ifftn(Fz, dim=(2,3), norm='forward'), zz, ff=kz, T=Nz, dim=4)
        return (fx, fy, fz)

    def Parseval_Loss(self):
        # Parseval Loss i.e., suppressing higher frequencies
        # avoid higher freqeuncies explaining everything
        new_feat = [Fk.unsqueeze(-1) * 1j * np.pi * wk.reshape(1, 1, *Fk.shape[2:], -1) 
            for Fk, wk in zip(self.density, self.ktraj_den)]
        loss = sum([feat.abs().square().mean() for feat in new_feat])
        return loss

    def compute_normal(self, xyz_sampled):
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

        new_den = self.upsample_fft(self.den, res_den)
        self.den = torch.nn.ParameterList([torch.nn.Parameter(den) for  den in new_den])
        new_app = self.upsample_fft(self.app, res_app)
        self.app = torch.nn.ParameterList([torch.nn.Parameter(app) for  app in new_app])

        self.print_size()
        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')

    def upsample_fft(self, features, res_target, update_dd=False):
        Tx, Ty, Tz = res_target
        Fkx, Fky, Fkz = features
        d1, d2, d3 = Fkx.shape[2], Fky.shape[3], Fkz.shape[4]
        Nx, Ny, Nz = Fky.shape[2], Fkz.shape[3], Fkx.shape[4]

        if update_dd:
            t1, t2, t3 = d1, d2, d3
            d1, d2, d3 = [int(np.log2(d))+1+2 for d in res_target]
            self.den_num_comp = [d1, d2, d3]
            self.axis = [torch.tensor([0.]+[2**i for i in torch.arange(d-1)]).to(self.device) for d in self.den_num_comp]

        maskx = getMask_fft([Ny, Nz], [Ty, Tz]).unsqueeze(0).repeat(d1,1,1)
        masky = getMask_fft([Nx, Nz], [Tx, Tz]).unsqueeze(1).repeat(1,d2,1)
        maskz = getMask_fft([Nx, Ny], [Tx, Ty]).unsqueeze(2).repeat(1,1,d3)

        if update_dd:
            maskx[t1:, :, :] = False
            masky[:, t2:, :] = False
            maskz[:, :, t3:] = False

        new_Fkx = torch.zeros(*Fkx.shape[:2], d1, Ty, Tz).to(Fkx)
        new_Fky = torch.zeros(*Fky.shape[:2], Tx, d2, Tz).to(Fky)
        new_Fkz = torch.zeros(*Fkz.shape[:2], Tx, Ty, d3).to(Fkz)
        
        try:
            new_Fkx[..., maskx] = Fkx[:, :, :d1, :, :].flatten(2)
            new_Fky[..., masky] = Fky[:, :, :, :d2, :].flatten(2)
            new_Fkz[..., maskz] = Fkz[:, :, :, :, :d3].flatten(2)
        except:
            raise ValueError("Error")
        return new_Fkx, new_Fky, new_Fkz

    def shrink_fft(self, features, t_l, b_r):
        # transform the fourier domain to spatial domain
        Fx, Fy, Fz = features
        res = torch.tensor([Fy.shape[2], Fz.shape[3], Fx.shape[4]]).to(b_r.device)
        b_r = torch.max(b_r, t_l+1)
        b_r = torch.min(b_r, res)
        Nx, Ny, Nz = res
        d1, d2, d3 = Fx.shape[2], Fy.shape[3], Fz.shape[4]
        xx, yy, zz = [torch.linspace(0, 1, N).to(self.device) for N in b_r-t_l]
        kx, ky, kz = self.axis
        kx, ky, kz = kx[:d1], ky[:d2], kz[:d3]
        fx, fy, fz = self.compute_spatial_volume(features)
        # shrink in the spatial domain
        fx = fx.data[..., t_l[0]:b_r[0], t_l[1]:b_r[1], t_l[2]:b_r[2]]
        fy = fy.data[..., t_l[0]:b_r[0], t_l[1]:b_r[1], t_l[2]:b_r[2]]
        fz = fz.data[..., t_l[0]:b_r[0], t_l[1]:b_r[1], t_l[2]:b_r[2]]
        try:
            # transform back to the fourier domain
            fx = rfft(torch.fft.fftn(fx.transpose(2,4), dim=(2,3), norm='forward'),xx, ff=kx, T=Nx).transpose(2,4)
            fy = rfft(torch.fft.fftn(fy.transpose(3,4), dim=(2,3), norm='forward'),yy, ff=ky, T=Ny).transpose(3,4)
            fz = rfft(torch.fft.fftn(fz.transpose(4,4), dim=(2,3), norm='forward'),zz, ff=kz, T=Nz).transpose(4,4)
        except:
            raise ValueError("Error in shrink FFT")
        return (fx, fy, fz)

    @torch.no_grad()
    def shrink(self, new_aabb):
        # return
        # since we have used tight bounding box,  no need shrinkage
        # you can turn this to opimize from a large bounding box in the first 2k iterations like TensoRF
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        den_tl, den_br = torch.floor(self.den_scale * t_l).long(), torch.ceil(self.den_scale * b_r).long()
        app_tl, app_br = torch.floor(self.app_scale * t_l).long(), torch.ceil(self.app_scale * b_r).long()

        
        old_den = self.shrink_fft(self.density, den_tl, den_br)
        # ugly here, how to compute the inverse of density
        new_den = tuple(den / self.alpha for den in old_den)

        
        new_app = self.shrink_fft(self.appearance, app_tl, app_br)
        new_app = tuple(app/self.beta for app in new_app)

        self.den = torch.nn.ParameterList([torch.nn.Parameter(den) for den in new_den])
        self.app = torch.nn.ParameterList([torch.nn.Parameter(app) for app in new_app])
        
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
       
    def update_stepSize(self, gridSize):
        self.ktraj = self.compute_ktraj(self.axis, gridSize)
        self.ktraj_den = self.compute_ktraj(self.axis, [math.ceil(n * self.den_scale) for n in gridSize])
        print("dimensions largest ",  [torch.max(ax).item() for ax in self.axis])
        return super(PhasoMLP, self).update_stepSize(gridSize)

    def print_size(self):
        print(self)
        print(f' ==> Actual Model Size {np.sum([v.numel() * v.element_size() for k, v in self.named_parameters()])/2**20} MB')
        for k,v in self.named_parameters():
            print(f'Model Size ({k}) : {v.numel() * v.element_size()/2**20:.4f} MB')

    def get_optparam_groups(self, lr_init_spatialxyz=0.02,
                            lr_init_network=0.001):
        grad_vars = [{'params': self.den, 'lr': lr_init_spatialxyz},
                     {'params': self.app, 'lr': lr_init_spatialxyz},
                     {'params': self.basis_mat.parameters(),
                      'lr':lr_init_network},
                     {'params': self.mlp.parameters(),
                      'lr': lr_init_network}]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(),
                           'lr':lr_init_network}]
        return grad_vars

    @property
    def alpha(self):
        return F.softplus(self.alpha_params, beta=10, threshold=1e-4) # avoid negative value

    @property
    def density(self):
        return [self.alpha * den for den in self.den]

    @property
    def appearance(self):
        return [app * self.beta for app in self.app]

    def compute_gaussian(self, variance, mode, ktraj=None):
        if mode== 'none':
            return [1., 1., 1.]
        if mode == 'prod':
            if ktraj is None:
                ktraj = self.ktraj
            ktraj = [k / g for k,g in zip(ktraj, self.gridSize)]
            gauss = [torch.exp((-2*(np.pi*kk)**2*variance[None]).sum(-1)).reshape(1,1,*kk.shape[:-1]) for kk in ktraj]
        else:
            raise ValueError(f'no mode named {mode}')
        return gauss

