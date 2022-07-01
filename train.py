import datetime
import json, random
import os
import sys
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from dataLoader import dataset_dict
from models.tensoRF import (
    AlphaGridMask, TensorCP, TensorVM, TensorVMSplit, TensorVX, raw2alpha, PREF
)
from opt import config_parser
from renderer import *
from renderer import OctreeRender_trilinear_fast as renderer
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
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset, tensorf, args, renderer,
                                f'{logfolder}/imgs_train_all/', n_vis=-1,
                                n_samples=-1, white_bg=white_bg,
                                ndc_ray=ndc_ray, device=device)
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/{args.expname}/imgs_test_all', exist_ok=True)
        evaluation(test_dataset, tensorf, args, renderer, f'{logfolder}/{args.expname}/imgs_test_all/',
                                n_vis=-1, n_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray, device=device)

    if args.render_path:
        c2ws = test_dataset.render_path
        os.makedirs(f'{logfolder}/{args.expname}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset, tensorf, c2ws, renderer, f'{logfolder}/{args.expname}/imgs_path_all/',
                                n_vis=-1, n_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray, device=device)


def reconstruction(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train',
                            downsample=args.downsample_train, is_stack=False)
    test_dataset = dataset(args.datadir, split='test',
                           downsample=args.downsample_train, is_stack=True)
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray

    # init resolution
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh
 
    logfolder = f'{args.basedir}/{args.expname}'
    if args.add_timestamp:
        logfolder += f'{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'

    # init parameters
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    n_samples = 512 # min(args.n_samples, cal_n_samples(reso_cur, args.step_ratio))

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device': device})
        tensorf = eval(args.model_name)(**kwargs)
        tensorf.load(ckpt)
    else:
        tensorf = eval(args.model_name)(
            aabb, reso_cur, device,
            alphaMask_thres=args.alpha_mask_thre,
            app_dim=args.data_dim_color,
            appearance_n_comp=n_lamb_sh,
            density_n_comp=n_lamb_sigma,
            density_shift=args.density_shift,
            distance_scale=args.distance_scale,
            fea2denseAct=args.fea2denseAct,
            featureC=args.featureC, step_ratio=args.step_ratio,
            near_far=near_far, shadingMode=args.shadingMode,
            pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe)

    tensorf = tensorf.cuda()
    print(tensorf)
    print(4 * sum([p.numel() for p in tensorf.parameters()]))

    lr = 0.002
    print(lr)
    optimizer = torch.optim.Adam(tensorf.parameters(),
                                 lr=lr, betas=(0.9, 0.99))
    scheduler = get_cos_warmup_scheduler(optimizer, args.n_iters, 0)

    PSNRs, PSNRs_test = [], [0]

    allrays = train_dataset.all_rays.cuda(non_blocking=True)
    allrgbs = train_dataset.all_rgbs.cuda(non_blocking=True)

    Ortho_reg_weight = args.Ortho_weight
    print("initial Ortho_reg_weight", Ortho_reg_weight)

    L1_reg_weight = args.L1_weight_inital
    print("initial L1_reg_weight", L1_reg_weight)

    TV_weight_density = 0.001 # args.TV_weight_density
    TV_weight_app = 0.001 # args.TV_weight_app
    tvreg = TVLoss()
    print(f"initial TV_weight density: {TV_weight_density} "
          f"appearance: {TV_weight_app}")

    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate)

    for iteration in pbar:
        indices = torch.randint(len(allrays), (args.batch_size,)).cuda()
        rays_train = torch.index_select(allrays, 0, indices)
        rgb_train = torch.index_select(allrgbs, 0, indices)

        # rgb_map, alphas_map, depth_map, weights, uncertainty
        rgb_map, alphas_map, depth_map, weights, uncertainty = renderer(
            rays_train, tensorf, chunk=args.batch_size, n_samples=n_samples,
            white_bg=white_bg, ndc_ray=ndc_ray, is_train=True)

        loss = torch.mean((rgb_map - rgb_train) ** 2)

        # loss
        total_loss = loss
        if Ortho_reg_weight > 0:
            total_loss += Ortho_reg_weight * tensorf.vector_comp_diffs()
        if L1_reg_weight > 0:
            total_loss += L1_reg_weight * tensorf.density_L1()
        if TV_weight_density > 0:
            loss_tv = tensorf.TV_loss_density(tvreg) * TV_weight_density
            total_loss = total_loss + loss_tv
        if TV_weight_app > 0:
            loss_tv = loss_tv + tensorf.TV_loss_app(tvreg)*TV_weight_app
            total_loss = total_loss + loss_tv

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
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
                f'Iter {iteration:05d}:'
                f' train={float(np.mean(PSNRs)):.3f}'
                f' test={float(np.mean(PSNRs_test)):.3f}')
            PSNRs = []

        if iteration in upsamp_list:
            pass # upsampling

    tensorf.save(f'{logfolder}/{args.expname}.th')

    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train',
                                downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset, tensorf, args, renderer,
                                f'{logfolder}/imgs_train_all/', n_vis=-1,
                                n_samples=-1, white_bg=white_bg,
                                ndc_ray=ndc_ray, device=device)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        PSNRs_test = evaluation(test_dataset, tensorf, args, renderer, f'{logfolder}/imgs_test_all/',
                                n_vis=-1, n_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_path:
        c2ws = test_dataset.render_path
        # c2ws = test_dataset.poses
        print('========>',c2ws.shape)
        os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset, tensorf, c2ws, renderer, f'{logfolder}/imgs_path_all/',
                                n_vis=-1, n_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)


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

