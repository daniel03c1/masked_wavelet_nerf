import imageio
import os
import sys
import torch
import torch.nn.functional as F
from dataLoader.ray_utils import get_rays, ndc_rays_blender
from pytorch_msssim import ssim
from tqdm.auto import tqdm

from utils import *
from vis_utils import *


def render_rays_split(density_net, appearance_net, rays, chunk,
                      *args, **kwargs):
    outputs = []

    for rays_minibatch in torch.split(rays, chunk):
        outputs.append(render_rays(density_net, appearance_net, rays_minibatch,
                                   *args, **kwargs))

    return list(map(torch.cat, zip(*outputs)))


def render_rays(density_net, appearance_net, rays, n_samples,
                is_train=False, white_bg=True, near=2, far=7,
                bounding_box=None, min_alpha_requirement=1e-4,
                normalize=True):
    rays_o, rays_d = rays[..., :3], rays[..., 3:] # origins, viewdirs

    # z_vals
    z_vals = torch.linspace(near, far, n_samples).unsqueeze(0).to(rays_o)

    if is_train:
        mids = (z_vals[..., 1:] + z_vals[..., :-1]) / 2
        lower = torch.cat([z_vals[..., :1], mids], -1)
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        z_vals = lower + (upper - lower) * torch.rand_like(z_vals)

    # [B, n_samples, 3]
    pts = rays_d[..., None, :] * z_vals[..., None] + rays_o[..., None, :]

    dists = F.pad(z_vals[:, 1:] - z_vals[:, :-1], [0, 1], value=1e10)
    dists = dists * torch.norm(rays_d, dim=-1, keepdim=True)

    viewdirs = F.normalize(rays_d, dim=-1).view(-1, 1, 3).expand(pts.shape)

    # TODO: alphamask

    sigma = torch.zeros(pts.shape[:-1], device=pts.device)
    rgb = torch.zeros((*pts.shape[:2], 3), device=pts.device)

    if bounding_box is not None:
        low_bbox = bounding_box.amin(0)
        high_bbox = bounding_box.amax(0)
        bbox_size = high_bbox - low_bbox

        valid_rays = ((low_bbox <= pts) & (pts <= high_bbox)).all(dim=-1)

        if normalize:
            pts = 2 * (pts - low_bbox) / bbox_size - 1
            rays_d *= 2 / bbox_size

        if valid_rays.any():
            sigma[valid_rays] = density_net(pts[valid_rays]).squeeze(-1)

    # alpha & weights
    alpha = 1. - torch.exp(-sigma * dists)
    weights = alpha * torch.cumprod(F.pad(1 - alpha + 1e-10, [1, 0], value=1),
                                    -1)[..., :-1] # exclusive cumprod

    # appearance
    app_mask = weights > min_alpha_requirement
    if app_mask.any():
        rgb[app_mask] += appearance_net(pts[app_mask], viewdirs[app_mask])

    rgb_map = torch.sum(weights[..., None] * rgb, -2)
    acc_map = torch.sum(weights, -1)

    if white_bg:
        rgb_map = rgb_map + (1. - acc_map[..., None])
    rgb_map = rgb_map.clamp(min=0, max=1)

    depth_map = torch.sum(weights * z_vals, -1)

    return rgb_map, depth_map


@torch.no_grad()
def evaluation(density_net, appearance_net, test_dataset, save_path,
               n_vis=5, prtx='', n_samples=-1, white_bg=False,
               compute_extra_metrics=True):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims, l_alex, l_vgg = [], [], []
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, "rgbd"), exist_ok=True)

    near_far = test_dataset.near_far
    if n_vis < 0:
        img_interval = 1
    else:
        img_interval = max(test_dataset.all_rays.shape[0] // n_vis, 1)
    idxs = list(range(0, test_dataset.all_rays.shape[0], img_interval))

    if compute_extra_metrics:
        import lpips
        lpips_alex = lpips.LPIPS(net='alex', version='0.1').eval().cuda()
        lpips_vgg = lpips.LPIPS(net='vgg', version='0.1').eval().cuda()

    for idx, samples in tqdm(enumerate(test_dataset.all_rays[::img_interval])):
        W, H = test_dataset.img_wh
        rays = samples.view(-1, samples.shape[-1]).cuda(non_blocking=True)

        with torch.no_grad(): 
            near, far = near_far
            rgb_map, depth_map = render_rays_split(
                density_net, appearance_net, rays, 4096, n_samples=n_samples,
                white_bg=white_bg, near=near, far=far,
                bounding_box=test_dataset.scene_bbox.cuda())

        rgb_map = rgb_map.reshape(H, W, 3)
        depth_map = depth_map.reshape(H, W)

        depth_map, _ = visualize_depth_numpy(depth_map.cpu().numpy(), near_far)

        if len(test_dataset.all_rgbs):
            gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3).cuda()
            loss = F.mse_loss(rgb_map, gt_rgb).cpu()
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                gt = gt_rgb.permute([2, 0, 1]).contiguous()
                im = rgb_map.permute([2, 0, 1]).contiguous()

                ssims.append(ssim(im[None], gt[None], data_range=1).cpu().item())
                l_alex.append(lpips_alex(gt, im, normalize=True).item())
                l_vgg.append(lpips_vgg(gt, im, normalize=True).item())

        rgb_map = (rgb_map.cpu().numpy() * 255).astype('uint8')
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)

        if save_path is not None:
            imageio.imwrite(f'{save_path}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{save_path}/rgbd/{prtx}{idx:03d}.png', rgb_map)

    imageio.mimwrite(f'{save_path}/{prtx}video.mp4', np.stack(rgb_maps),
                     fps=30, quality=10)
    imageio.mimwrite(f'{save_path}/{prtx}depthvideo.mp4', np.stack(depth_maps),
                     fps=30, quality=10)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            avg_ssim = np.mean(np.asarray(ssims))
            avg_l_a = np.mean(np.asarray(l_alex))
            avg_l_v = np.mean(np.asarray(l_vgg))
            print(f'ssim: {avg_ssim}, LPIPS(alexnet): {avg_l_a}, '
                  f'LPIPS(vgg): {avg_l_v}')
            np.savetxt(f'{save_path}/{prtx}mean.txt',
                       np.asarray([psnr, avg_ssim, avg_l_a, avg_l_v]))
        else:
            np.savetxt(f'{save_path}/{prtx}mean.txt', np.asarray([psnr]))

    return PSNRs


@torch.no_grad()
def evaluation_path(test_dataset,tensorf, c2ws, save_path=None,
                    n_vis=5, prtx='', n_samples=-1, white_bg=False,
                    ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    for idx, c2w in tqdm(enumerate(c2ws)):
        W, H = test_dataset.img_wh

        c2w = torch.FloatTensor(c2w)
        rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        if ndc_ray:
            rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

        rgb_map, depth_map = render_rays(rays, tensorf, chunk=8192,
                                         n_samples=n_samples, ndc_ray=ndc_ray,
                                         white_bg=white_bg, device=device)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if save_path is not None:
            imageio.imwrite(f'{save_path}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{save_path}/rgbd/{prtx}{idx:03d}.png', rgb_map)

    imageio.mimwrite(f'{save_path}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=8)
    imageio.mimwrite(f'{save_path}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=8)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{save_path}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{save_path}/{prtx}mean.txt', np.asarray([psnr]))

    return PSNRs

