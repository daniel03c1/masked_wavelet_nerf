import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from .modules import get_activation


class TwoStageNeuralField(nn.Module):
    def __init__(self, first_stage, second_stage):
        super().__init__()
        self.first_stage = first_stage
        self.second_stage = second_stage

    def forward(self, coords, viewdirs=None):
        latent_features = self.first_stage(coords, viewdirs)
        return self.second_stage(coords, viewdirs, latent_features)

    def compute_tv(self):
        # compute total variation
        tv = 0
        for net in [self.first_stage, self.second_stage]:
            if isinstance(net, nn.Sequential):
                tv += sum([m.compute_tv() if hasattr(m, 'compute_tv') else 0
                           for m in net.modules()])
            else: # assume it's nn.Module
                if hasattr(net, 'compute_tv'):
                    tv += net.compute_tv()
        return tv

    def compute_bits(self):
        def default_fn(module):
            return sum([p.numel() for p in module.parameters()]) * 32

        bits = 0
        for net in [self.first_stage, self.second_stage]:
            if isinstance(net, nn.Sequential):
                bits += sum([m.compute_bits()
                             if hasattr(m, 'compute_bits') else default_fn(m)
                             for m in net.modules()])
            else: # assume it's nn.Module
                if hasattr(net, 'compute_bits'):
                    bits += net.compute_bits()
                else:
                    bits += default_fn(net)
        return bits


class Renderer(nn.Module):
    def __init__(self, main_net: nn.Module, appearance_net=None,
                 n_samples_per_ray: int = 1024, bounding_box=None,
                 near=2, far=7, white_bg=True,
                 use_alpha=True, min_alpha_requirement=1e-4,
                 normalize_coords=True,
                 density_scale=1, # 25,
                 density_activation=None, appearance_activation=None):
        super().__init__()
        """
        INPUTS
            main_net: a network to output both density and color
                      if appearance_net is given, the main_net refers to
                      a density net.
            appearance_net:
            n_samples_per_ray:
            bounding_box: None or [2, n_dim] shaped tensor
            near:
            far:
            white_bg:
            use_alpha:
            min_alpha_requirement:
            normalize_coords:
        """
        self.main_net = main_net
        self.appearance_net = appearance_net
        self.n_samples_per_ray = n_samples_per_ray
        self.bounding_box = bounding_box
        self.near = near
        self.far = far
        self.white_bg = white_bg
        self.use_alpha = use_alpha # TODO(daniel): need to implement alpha
        self.min_alpha_requirement = min_alpha_requirement
        self.normalize_coords = normalize_coords
        self.density_scale = density_scale
        self.density_activation = get_activation(density_activation)
        self.appearance_activation = get_activation(appearance_activation)

    def forward(self, rays, batch_size=None):
        if batch_size is None:
            return self.render(rays)

        outputs = []
        for rays_minibatch in torch.split(rays, batch_size):
            outputs.append(self.render(rays_minibatch))

        return list(map(torch.cat, zip(*outputs)))

    def render(self, rays):
        """
        RENDERING PROCESS

        INPUTS:
            rays: [..., 6] shaped tensor (3 for origin, 3 for direction)
        """
        rays_o, rays_d = rays[..., :3], rays[..., 3:] # origins, viewdirs
        z_vals = torch.linspace(self.near, self.far,
                                self.n_samples_per_ray).unsqueeze(0).to(rays_o)

        if self.training:
            mids = (z_vals[..., 1:] + z_vals[..., :-1]) / 2
            lower = torch.cat([z_vals[..., :1], mids], -1)
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            z_vals = lower + (upper - lower) * torch.rand_like(z_vals)

        # [B, n_samples, 3]
        pts = rays_d[..., None, :] * z_vals[..., None] + rays_o[..., None, :]
        viewdirs = F.normalize(rays_d, dim=-1).view(-1, 1, 3).expand(pts.shape)

        if self.bounding_box is not None:
            bbox_low = self.bounding_box.amin(0)
            bbox_high = self.bounding_box.amax(0)
            bbox_size = bbox_high - bbox_low

            valid_rays = ((bbox_low <= pts) & (pts <= bbox_high)).all(dim=-1)
        else:
            valid_rays = torch.ones_like(pts[..., 0])

        if self.normalize_coords and self.bounding_box is not None:
            pts = 2 * (pts - bbox_low) / bbox_size - 1

        dists = F.pad(z_vals[..., 1:]-z_vals[..., :-1], [0, 1], value=1e10)
        dists = dists * torch.norm(rays_d, dim=-1, keepdim=True)

        if self.appearance_net is not None:
            # 1. when using separate density and appearance nets
            sigma = torch.zeros_like(pts[..., 0])
            rgb = torch.zeros((*pts.shape[:2], 3), device=pts.device)

            if valid_rays.any():
                sigma[valid_rays] = self.density_activation(
                    self.main_net(pts[valid_rays]) * self.density_scale)

            # alpha & weights
            alpha = 1. - torch.exp(-sigma * dists)
            weights = alpha * torch.cumprod(F.pad(1 - alpha + 1e-10, [1, 0],
                                                  value=1),
                                            -1)[..., :-1] # exclusive cumprod

            # appearance
            app_mask = weights > self.min_alpha_requirement
            if app_mask.any():
                rgb[app_mask] += self.appearance_activation(
                    self.appearance_net(pts[app_mask], viewdirs[app_mask]))
        else:
            outs = self.main_net(pts[valid_rays], viewdirs[valid_rays])
            sigma = self.density_activation(outs[..., 0] * self.density_scale)
            rgb = self.appearance_activation(outs[..., 1:])

            # alpha & weights
            alpha = 1. - torch.exp(-sigma * dists)
            weights = alpha * torch.cumprod(F.pad(1 - alpha + 1e-10, [1, 0],
                                                  value=1),
                                            -1)[..., :-1] # exclusive cumprod

        acc_map = torch.sum(weights, -1)
        rgb_map = torch.sum(weights[..., None] * rgb, -2)

        if self.white_bg:
            rgb_map = rgb_map + (1. - acc_map[..., None])
        rgb_map = rgb_map.clamp(min=0, max=1)

        depth_map = torch.sum(weights * z_vals, -1)

        return rgb_map, depth_map

    def evaluation(self, test_dataset, batch_size=4096, save_path=None,
                   compute_extra_metrics=True):
        self.eval()

        PSNRs, rgb_maps, depth_maps = [], [], []

        if save_path is not None:
            import imageio
            os.makedirs(save_path, exist_ok=True)
            os.makedirs(os.path.join(save_path, "rgbd"), exist_ok=True)

        if compute_extra_metrics:
            import lpips
            from pytorch_msssim import ssim

            ssims, l_alex, l_vgg = [], [], []
            lpips_alex = lpips.LPIPS(net='alex', version='0.1').eval().cuda()
            lpips_vgg = lpips.LPIPS(net='vgg', version='0.1').eval().cuda()

        try:
            gt_exist = len(test_dataset.all_rgbs) > 0
        except Exception:
            gt_exist = False

        with torch.no_grad():
            for idx in tqdm.tqdm(range(len(test_dataset))):
                W, H = test_dataset.img_wh

                rays = test_dataset.all_rays[idx]
                rays = rays.view(-1, rays.shape[-1]).cuda(non_blocking=True)

                rgb_map, depth_map = self.forward(rays, batch_size=batch_size)

                rgb_map = rgb_map.reshape(H, W, 3)
                depth_map = depth_map.reshape(H, W)

                if gt_exist:
                    gt_rgb = test_dataset.all_rgbs[idx].view(H, W, 3) \
                                                       .cuda(non_blocking=True)

                    loss = F.mse_loss(rgb_map, gt_rgb)
                    PSNRs.append(-10.0 * torch.log(loss) / np.log(10.0))

                    if compute_extra_metrics:
                        gt = gt_rgb.permute([2, 0, 1]).contiguous()
                        im = rgb_map.permute([2, 0, 1]).contiguous()

                        ssims.append(ssim(im[None], gt[None], data_range=1))
                        l_alex.append(lpips_alex(gt, im, normalize=True))
                        l_vgg.append(lpips_vgg(gt, im, normalize=True))

                    del gt_rgb

                rgb_map = (rgb_map * 255).int()
                rgb_maps.append(rgb_map.cpu())
                depth_maps.append(depth_map.cpu())

                if save_path is not None:
                    imageio.imwrite(os.path.join(save_path, f'{idx:03d}.png'),
                                    rgb_map.cpu().numpy())
                    rgb_map = torch.concat((rgb_map, depth_map), axis=1)
                    imageio.imwrite(os.path.join(save_path,
                                                 f'rgbd/{idx:03d}.png'),
                                    rgb_map.cpu().numpy())

                del rays, rgb_map, depth_map

        if save_path is not None:
            imageio.mimwrite(os.path.join(save_path, f'video.mp4'),
                             torch.stack(rgb_maps).numpy(),
                             fps=30, quality=10)
            imageio.mimwrite(os.path.join(save_path, f'depthvideo.mp4'),
                             torch.stack(depth_maps).numpy(),
                             fps=30, quality=10)

        if gt_exist:
            psnr = torch.stack(PSNRs).mean().cpu().numpy()

            if compute_extra_metrics:
                avg_ssim = torch.stack(ssims).mean().cpu().numpy()
                avg_l_a = torch.stack(l_alex).mean().cpu().numpy()
                avg_l_v = torch.stack(l_vgg).mean().cpu().numpy()
                print(f'ssim: {avg_ssim:.4f}, LPIPS(alexnet): {avg_l_a:.4f}, '
                      f'LPIPS(vgg): {avg_l_v:.4f}')
                if save_path is not None:
                    np.savetxt(os.path.join(save_path, 'mean.txt'),
                               np.asarray([psnr, avg_ssim, avg_l_a, avg_l_v]))
            elif save_path is not None:
                np.savetxt(os.path.join(save_path, 'mean.txt'),
                           np.asarray([psnr]))

        self.train()

        return PSNRs

    def compute_tv(self):
        bits = self.main_net.compute_tv()
        if self.appearance_net is not None:
            bits = bits + self.appearance_net.compute_tv()
        return bits

    def compute_bits(self):
        bits = self.main_net.compute_bits()
        if self.appearance_net is not None:
            bits = bits + self.appearance_net.compute_bits()
        return bits

