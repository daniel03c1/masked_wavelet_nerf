import datetime
import json, random
import os
import sys
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from dataLoader import dataset_dict
from models.tensoRF import AlphaGridMask, TensorCP, TensorVM, raw2alpha, PREF
from opt import config_parser
from renderer import *
from utils import *


device = torch.device("cuda") #  if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def export_mesh(args):
    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    alpha, _ = tensorf.getDenseAlpha()
    convert_sdf_samples_to_ply(alpha.cpu(), f'{args.ckpt[:-3]}.ply',
                               bbox=tensorf.aabb.cpu(), level=0.005)


@torch.no_grad()
def render_test(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(args.datadir, split='test',
                           downsample=args.downsample_train, is_stack=True)
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    logfolder = os.path.dirname(args.ckpt)
    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train',
                                downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset, tensorf, args, 
                                f'{logfolder}/imgs_train_all/', n_vis=-1,
                                n_samples=-1, white_bg=white_bg,
                                ndc_ray=ndc_ray, device=device)
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_test)} '
              f'<========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/{args.expname}/imgs_test_all', exist_ok=True)
        evaluation(test_dataset, tensorf, args,
                   f'{logfolder}/{args.expname}/imgs_test_all/', n_vis=-1,
                   n_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray,
                   device=device)

    if args.render_path:
        c2ws = test_dataset.render_path
        os.makedirs(f'{logfolder}/{args.expname}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset, tensorf, c2ws,
                        f'{logfolder}/{args.expname}/imgs_path_all/', n_vis=-1,
                        n_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray,
                        device=device)


def reconstruction(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train',
                            downsample=args.downsample_train, is_stack=False)
    test_dataset = dataset(args.datadir, split='test',
                           downsample=args.downsample_train, is_stack=True)

    allrays = train_dataset.all_rays.cuda(non_blocking=True)
    allrgbs = train_dataset.all_rgbs.cuda(non_blocking=True)

    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray

    logfolder = f'{args.basedir}/{args.expname}'
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_rgba', exist_ok=True)
    os.makedirs(f'{logfolder}/rgba', exist_ok=True)

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device': device})
        tensorf = eval(args.model_name)(**kwargs)
        tensorf.load(ckpt)
    else:
        tensorf = eval(args.model_name)(
            train_dataset.scene_bbox.cuda(), (256, 256, 256), device,
            app_dim=args.data_dim_color,
            alphaMask_thres=args.alpha_mask_thre,
            density_shift=args.density_shift,
            distance_scale=args.distance_scale,
            fea2denseAct=args.fea2denseAct,
            featureC=args.featureC, step_ratio=args.step_ratio,
            near_far=near_far, shadingMode=args.shadingMode,
            pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe)

    tensorf = tensorf.cuda()
    print(tensorf)
    print(4 * sum([p.numel() for p in tensorf.parameters()]))

    lr = 0.005 # 2
    print(lr)
    optimizer = torch.optim.Adam(tensorf.parameters(),
                                 lr=lr, betas=(0.9, 0.99))
    scheduler = get_cos_warmup_scheduler(optimizer, args.n_iters, 0)
    scaler = torch.cuda.amp.GradScaler()

    PSNRs, PSNRs_test = [], [0]

    TV_weight_density = 0. # 001 # args.TV_weight_density
    TV_weight_app = 0. # 001 # args.TV_weight_app
    print(f"initial TV_weight density: {TV_weight_density} "
          f"appearance: {TV_weight_app}")

    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate)
    n_samples = 1024 # 512

    for iteration in pbar:
        indices = torch.randint(len(allrays), (args.batch_size,)).cuda()
        rays_train = torch.index_select(allrays, 0, indices)
        rgb_train = torch.index_select(allrgbs, 0, indices)

        optimizer.zero_grad()

        # rgb_map, alphas_map, depth_map, weights, uncertainty
        with torch.cuda.amp.autocast(enabled=True):
            rgb_map, depth_map = render_rays(
                rays_train, tensorf, chunk=args.batch_size, n_samples=n_samples,
                white_bg=white_bg, ndc_ray=ndc_ray, is_train=True)

            loss = torch.mean((rgb_map - rgb_train) ** 2)

            # loss
            total_loss = loss
            if TV_weight_density > 0:
                total_loss += tensorf.TV_loss_density(None) * TV_weight_density
            if TV_weight_app > 0:
                total_loss += tensorf.TV_loss_app(None) * TV_weight_app

            scaler.scale(total_loss).backward(retain_graph=True)

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(tensorf.parameters(), 1)
        scaler.step(optimizer)
        scaler.update()

        '''
        total_loss.backward()
        optimizer.step()
        '''
        scheduler.step()

        PSNRs.append(mse2psnr_np(loss.detach().item()))

        if args.n_vis > 0 and (iteration+1) % args.vis_every == args.vis_every:
            PSNRs_test = evaluation(test_dataset, tensorf, args, renderer,
                                    f'{logfolder}/imgs_vis/', n_vis=args.n_vis,
                                    prtx=f'{iteration:06d}_',
                                    n_samples=n_samples, white_bg=white_bg,
                                    ndc_ray=ndc_ray,
                                    compute_extra_metrics=False)

        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f'Iter {iteration:05d}: train={float(np.mean(PSNRs)):.3f}'
                f' test={float(np.mean(PSNRs_test)):.3f}')
            PSNRs = []

    tensorf.save(f'{logfolder}/{args.expname}.th')

    import pdb; pdb.set_trace()

    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train',
                                downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset, tensorf, args,
                                f'{logfolder}/imgs_train_all/', n_vis=-1,
                                n_samples=n_samples, white_bg=white_bg,
                                ndc_ray=ndc_ray, device=device)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        PSNRs_test = evaluation(test_dataset, tensorf, args,
                                f'{logfolder}/imgs_test_all/', n_vis=-1,
                                n_samples=n_samples, white_bg=white_bg,
                                ndc_ray=ndc_ray, device=device)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_path:
        c2ws = test_dataset.render_path
        # c2ws = test_dataset.poses
        print('========>',c2ws.shape)
        os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset, tensorf, c2ws,
                        f'{logfolder}/imgs_path_all/', n_vis=-1,
                        n_samples=n_samples,
                        white_bg=white_bg, ndc_ray=ndc_ray,device=device)


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()
    print(args)

    if args.export_mesh:
        export_mesh(args)

    if args.render_only and (args.render_test or args.render_path):
        render_test(args)
    else:
        reconstruction(args)

