import datetime
import json
import os
import random
import sys
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from dataLoader import dataset_dict
from models import AppearanceNet, DensityNet
from models.grid_based import FreqGrid
from models.modules import MLP, Softplus
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
    near, far = train_dataset.near_far
    bbox = train_dataset.scene_bbox.cuda()
    ndc_ray = args.ndc_ray

    logfolder = f'{args.basedir}/{args.expname}'
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_rgba', exist_ok=True)
    os.makedirs(f'{logfolder}/rgba', exist_ok=True)

    # TODO: loading weights
    '''
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device': device})
        tensorf = eval(args.model_name)(**kwargs)
        tensorf.load(ckpt)
    '''
    # defining networks
    density_net = nn.Sequential(FreqGrid(256, 2),
                                nn.Linear(48, 64),
                                nn.ReLU(inplace=True),
                                nn.Linear(64, 64),
                                nn.ReLU(inplace=True),
                                nn.Linear(64, 1),
                                Softplus(shift=-10))
    density_net = DensityNet(density_net).cuda()

    appearance_net = AppearanceNet(
        FreqGrid(256, 4),
        MLP(96, include_view=True, feat_n_freq=0, pos_n_freq=args.pos_pe,
            view_n_freq=args.view_pe, hidden_dim=128, out_activation='sigmoid'))
    appearance_net = appearance_net.cuda()

    print(density_net)
    print(appearance_net)
    n_params = sum([p.numel() for p in density_net.parameters()]) \
             + sum([p.numel() for p in appearance_net.parameters()])
    print(f'SIZE: {16 * n_params / 8_388_608 :.2f} MB')

    print(f'lr_init: {args.lr_init}')
    optimizer = torch.optim.Adam([{'params': density_net.parameters()},
                                  {'params': appearance_net.parameters()}],
                                 lr=args.lr_init, betas=(0.9, 0.99))
    scheduler = get_cos_warmup_scheduler(optimizer, args.n_iters, 0,
                                         min_ratio=0.1)
    scaler = torch.cuda.amp.GradScaler()

    PSNRs, PSNRs_test = [], [0]

    TV_weight_density = args.TV_weight_density
    TV_weight_app = args.TV_weight_app
    print(f"TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")

    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate)
    n_samples = args.n_samples

    for iteration in pbar:
        indices = torch.randint(len(allrays), (args.batch_size,)).cuda()
        rays_train = torch.index_select(allrays, 0, indices)
        rgb_train = torch.index_select(allrgbs, 0, indices)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=True):
            rgb_map, depth_map = render_rays(
                density_net, appearance_net, rays_train, n_samples,
                is_train=True, white_bg=white_bg, near=near,
                far=far, bounding_box=bbox)

            loss = F.mse_loss(rgb_map, rgb_train)

            # loss
            total_loss = loss
            if TV_weight_density > 0:
                total_loss += density_net.compute_tv() * TV_weight_density
            if TV_weight_app > 0:
                total_loss += appearance_net.compute_tv() * TV_weight_app

            assert not torch.isnan(loss)
            scaler.scale(total_loss).backward(retain_graph=True)

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in density_net.parameters()]
            + [p for p in appearance_net.parameters()], 1)
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()

        PSNRs.append(mse2psnr_np(loss.detach().item()))

        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f'Iter {iteration:05d}: train={float(np.mean(PSNRs)):.3f}')
            PSNRs = []

        if iteration+1 in [500, 1000, 2500, 5000, 10000, 20000]:
            print()

    # TODO: save the model
    # tensorf.save(f'{logfolder}/{args.expname}.th')

    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train',
                                downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset, tensorf, args,
                                f'{logfolder}/imgs_train_all/', n_vis=-1,
                                n_samples=n_samples, white_bg=white_bg,
                                ndc_ray=ndc_ray, device=device)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} '
              f'<========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        PSNRs_test = evaluation(density_net, appearance_net, test_dataset,
                                f'{logfolder}/imgs_test_all/', n_vis=-1,
                                n_samples=n_samples, white_bg=white_bg)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} '
              f'<========================')

    if args.render_path:
        c2ws = test_dataset.render_path
        # c2ws = test_dataset.poses
        print('========>', c2ws.shape)
        os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset, tensorf, c2ws,
                        f'{logfolder}/imgs_path_all/', n_vis=-1,
                        n_samples=n_samples,
                        white_bg=white_bg, ndc_ray=ndc_ray, device=device)


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

