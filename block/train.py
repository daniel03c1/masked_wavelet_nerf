import datetime
import json
import os
import random
import sys
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from dataLoader import dataset_dict
from opt import config_parser
from renderer import *
from utils import *
from huffman import *

#test!!
import imageio_ffmpeg


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda" if torch.cuda.is_available() else "cpu"

renderer = OctreeRender_trilinear_fast


class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr += self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.randperm(self.total).to("cuda")    # issue
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]


@torch.no_grad()
def export_mesh(args):
    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    phasorf = eval(args.model_name)(**kwargs)
    phasorf.load(ckpt)

    alpha,_ = phasorf.getDenseAlpha()
    convert_sdf_samples_to_ply(alpha.cpu(), f'{args.ckpt[:-3]}.ply',
                               bbox=phasorf.aabb.cpu(), level=0.005)


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
    phasorf = eval(args.model_name)(**kwargs)
    phasorf.load(ckpt)

    logfolder = os.path.dirname(args.ckpt)
    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train',
                                downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset, phasorf, args, renderer,
                                f'{logfolder}/imgs_train_all/', N_vis=-1,
                                N_samples=-1, white_bg=white_bg,
                                ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_test)} '
              f'<========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/{args.expname}/imgs_test_all', exist_ok=True)
        PSNRs_test = evaluation(test_dataset, phasorf, args, renderer,
                                f'{logfolder}/{args.expname}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg=white_bg,
                                ndc_ray=ndc_ray, device=device)
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_test)} '
              f'<========================')

    if args.render_path:
        c2ws = test_dataset.render_path
        os.makedirs(f'{logfolder}/{args.expname}/imgs_path_all', exist_ok=True)
        PSNRs_test = evaluation_path(test_dataset, phasorf, c2ws, renderer,
                                     f'{logfolder}/{args.expname}/imgs_path_all/',
                                     N_vis=-1, N_samples=-1, white_bg=white_bg,
                                     ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_test)} '
              f'<========================')


def rle_test(args):
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logfolder = f'{args.basedir}/{args.expname}'    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(f'{logfolder}/{args.expname}.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device':device})
    kwargs.update({'logger':logger})
    phasorf = eval(args.model_name)(**kwargs)
    phasorf.load(ckpt)

    phasorf.mask_thres = 0.51
    phasorf.mask_learning = False
    phasorf.iter = None

    phasorf.mask_thres = ckpt['mask_thres']

    mask_thres_tensor = torch.Tensor([phasorf.mask_thres]).to(device)
    if phasorf.mask_thres == 0.5:
        m_thres = 0
    else:
        m_thres = torch.log((1 - mask_thres_tensor) / mask_thres_tensor) * (-1)

    from rle.np_impl import dense_to_rle, rle_length, rle_to_dense
    
    for i in range(3):
        phasorf.den[i].requires_grad=False
        phasorf.app[i].requires_grad=False

        masked = phasorf.den_mask[i] < m_thres
        phasorf.den[i][masked] = 0 
        masked = phasorf.app_mask[i] < m_thres
        phasorf.app[i][masked] = 0 

    den_compressed = app_compressed = []
    den_perm_compressed = app_perm_compressed = []
    den_zig_compressed = app_zig_compressed = []

    for i in range(3):
        den_zig = phasorf.den[i].squeeze().permute(0,2,3,1).cpu().detach().numpy() 
        app_zig = phasorf.app[i].squeeze().permute(0,2,3,1).cpu().detach().numpy() 
        
        den_perm = phasorf.den[i].squeeze().permute(0,2,3,1).flatten().cpu().detach().numpy() 
        app_perm = phasorf.app[i].squeeze().permute(0,2,3,1).flatten().cpu().detach().numpy()  
        
        den_zig = zigzag(den_zig).flatten()
        app_zig = zigzag(app_zig).flatten()
        
        den = phasorf.den[i].flatten().cpu().detach().numpy()  
        app = phasorf.app[i].flatten().cpu().detach().numpy()  
        rle_den_perm = dense_to_rle(den_perm)
        rle_app_perm = dense_to_rle(app_perm)
        rle_den_zig = dense_to_rle(den_zig)
        rle_app_zig = dense_to_rle(app_zig)
        rle_den = dense_to_rle(den)
        rle_app = dense_to_rle(app)

        den_compressed.append(rle_den)
        app_compressed.append(rle_app)
        den_perm_compressed.append(rle_den_perm)
        app_perm_compressed.append(rle_app_perm)
        den_zig_compressed.append(rle_den_zig)
        app_zig_compressed.append(rle_app_zig)
    
    uncomp = 0
    comp = perm = zig = 0
    for i in range(3):
        print(den_compressed[i].shape[0] / den_perm_compressed[i].shape[0], app_compressed[i].shape[0] / app_perm_compressed[i].shape[0])
        uncomp = uncomp + phasorf.den[i].count_nonzero() + phasorf.app[i].count_nonzero()
        comp = den_compressed[i].shape[0] + app_compressed[i].shape[0]
        perm = den_perm_compressed[i].shape[0] + app_perm_compressed[i].shape[0]
        zig = den_zig_compressed[i].shape[0] + app_zig_compressed[i].shape[0]

    print(args.ckpt)
    print(f'uncomp size: {uncomp*4/1_048_576:.4f}MB')
    print(f'comp size: {comp*4/1_048_576:.4f}MB')
    print(f'perm size: {perm*4/1_048_576:.4f}MB')
    print(f'zig size: {zig*4/1_048_576:.4f}MB')

    zig_size = zig*4/1_048_576
    net_size = 0
    net_size  += phasorf.basis_mat.weight.shape[0] * phasorf.basis_mat.weight.shape[1] * phasorf.basis_mat.weight.element_size() 
    
    idx = [0,2]
    
    for i in idx:
        net_size  += phasorf.mlp[i].weight.shape[0] * phasorf.mlp[i].weight.shape[1] * phasorf.mlp[i].weight.element_size()
        net_size  += phasorf.mlp[i].bias.shape[0] * phasorf.mlp[i].bias.element_size()


    idx = [0,2,4]
    for i in idx:
        net_size  += phasorf.renderModule.mlp[i].weight.shape[0] * phasorf.renderModule.mlp[i].weight.shape[1] * phasorf.renderModule.mlp[i].weight.element_size()
        net_size  += phasorf.renderModule.mlp[i].bias.shape[0] * phasorf.renderModule.mlp[i].bias.element_size()

    net_size = net_size / 1_048_576.0
    print(f"{zig_size} + {net_size} = {zig_size + net_size} ")


def reconstruction(args, return_bbox=False, return_memory=False,
                   bbox_only=False):

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

    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}' \
                    f'{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}'

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_rgba', exist_ok=True)
    os.makedirs(f'{logfolder}/rgba', exist_ok=True)
    summary_writer = SummaryWriter(logfolder)
    # save config files
    json.dump(args.__dict__, open(f'{logfolder}/config.json',mode='w'),indent=2)

    import logging
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(f'{logfolder}/{args.expname}.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    print(args.expname)


    # init parameters
    if not bbox_only and args.dataset_name == 'blender':   
        # use tight bbox pre-extracted and stored in misc.py,
        # which takes 2k iters
        data = args.datadir.split('/')[-1]
        from misc import blender_aabb
        aabb = torch.tensor(blender_aabb[data]).reshape(2,3).to(device)
    else:
        # run bbox from scratch
        aabb = train_dataset.scene_bbox.to(device)

    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    
    import math
    var_split = True if type(args.block_split) is list else False



    adaptive_block = True 
    mask_learning = False    
    mask_schedule = True    


    if mask_learning:
        mask_thres = torch.ones(1).to(device)
        mask_thres.requires_grad=True
    else:
        mask_thres = 0 # 0

    if mask_schedule:
        mask_iter = args.mask_iter          
        mask_thres_list = args.mask_thres_list
        assert len(mask_iter) == len(mask_thres_list)
        logger.info(f"masking schedule: {mask_iter},   {mask_thres_list}")

    if adaptive_block:
        ratio = True
        if ratio:
            bbox_size = aabb[1] - aabb[0]   
            bbox_ind = torch.sort(bbox_size)[1] 
            _blk_split = [args.block_split[0]]*3   
            print(_blk_split)
            smallest_bbox_size = bbox_size[bbox_ind[0]]
            print(bbox_size, smallest_bbox_size)
            
            for i in range(len(bbox_ind)):
                _blk_split[bbox_ind[i]] = int(_blk_split[bbox_ind[i]] * (bbox_size[bbox_ind[i]] / smallest_bbox_size))

            args.block_split = _blk_split
            print(args.block_split)
            logger.info(f"block split: {args.block_split}")
            
    

    if var_split:
        reso_cur = [math.ceil(reso_cur[i] / args.block_split[i]) * args.block_split[i] for i in range(len(reso_cur))]
    else:
        reso_cur = [math.ceil(reso / args.block_split) * args.block_split for reso in reso_cur]

    logger.info(f"resolution: {reso_cur}")
    nSamples = min(args.nSamples,
                   cal_n_samples(reso_cur,args.step_ratio))

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device':device})
        kwargs.update({'logger':logger})
        phasorf = eval(args.model_name)(**kwargs)
        phasorf.load(ckpt)

    else:
        phasorf = eval(args.model_name)(aabb, reso_cur, device,
                    # modeling
                    den_num_comp=args.den_num_comp, 
                    app_num_comp=args.app_num_comp, 
                    app_dim=args.app_dim, 
                    softplus_beta=args.softplus_beta,
                    app_aug=args.app_aug,
                    app_ksize = args.app_ksize,
                    den_ksize = args.den_ksize,
                    alpha_init=args.alpha_init,
                    den_scale=args.den_scale,
                    app_scale=args.app_scale,
                    update_dd=args.update_dd, 
                    # rendering 
                    near_far=near_far,
                    shadingMode=args.shadingMode, 
                    alphaMask_thres=args.alpha_mask_thre, 
                    density_shift=args.density_shift, 
                    distance_scale=args.distance_scale,
                    pos_pe=args.pos_pe, 
                    view_pe=args.view_pe, 
                    fea_pe=args.fea_pe, 
                    featureC=args.featureC, 
                    step_ratio=args.step_ratio, 
                    fea2denseAct=args.fea2denseAct,
                    block_split=args.block_split,
                    logger=logger, mask_lr=args.mask_lr)
        phasorf.mask_thres = mask_thres
        logger.info(args)

    phasorf.logger = logger
    eps = 1e-4
    phasorf.mask_learning = mask_learning
    
    mask_thres_tensor = torch.Tensor([phasorf.mask_thres]).to(device) # threshold for sigmoided value, not pure x

    if phasorf.mask_thres == 0.5:
        m_thres = 0
    else:
        m_thres = torch.log((1 - mask_thres_tensor) / mask_thres_tensor) * (-1)
    logger.info(f"mask threshold = {phasorf.mask_thres}, m_thres = {m_thres}")
    logger.info(f"split {phasorf.resolution} with {phasorf.block_split} blocks. Block res = {phasorf.block_resolution} and freq is {phasorf.n_freq}. net stride: {phasorf.network_strides[1]}")

    L1_weight = 8e-5
    L2_weight = 1e-6

    grad_vars = phasorf.get_optparam_groups(args.lr_init, args.lr_basis)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)
    logger.info(f"lr decay  {args.lr_decay_target_ratio}  {args.lr_decay_iters}")

    optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))
        

    # linear in logrithmic space
    if upsamp_list:
        N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(args.N_voxel_init), 
            np.log(args.N_voxel_final), len(upsamp_list)+1))).long()).tolist()[1:]

    torch.cuda.empty_cache()
    PSNRs, PSNRs_test = [], [0]

    allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
    if not args.ndc_ray:
        allrays, allrgbs = phasorf.filtering_rays(allrays, allrgbs,
                                                  bbox_only=True)
    allrays = allrays.to(device)
    allrgbs = allrgbs.to(device)
    trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)

    TV_weight_density = args.TV_weight_density
    TV_weight_app = args.TV_weight_app
    print(f"initial TV_weight density: {TV_weight_density} "
          f"appearance: {TV_weight_app}")
    logger.info(f"initial TV_weight density: {TV_weight_density}  appearance: {TV_weight_app}")

    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate)

    phasorf.print_size()

    for iteration in pbar:
        torch.cuda.empty_cache()
        phasorf.iter = iteration
        ray_idx = trainingSampler.nextids()
        rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)

        rgb_map, alphas_map, depth_map, weights, uncertainty = renderer(
            rays_train, phasorf, chunk=args.batch_size, N_samples=nSamples,
            white_bg=white_bg, ndc_ray=ndc_ray, device=device, is_train=True)

        loss = torch.mean((rgb_map - rgb_train) ** 2)
        # loss
        torch.set_printoptions(precision=10)
        total_loss = loss
        
        if TV_weight_density > 0 and (iteration % args.TV_step == 0):
            TV_weight_density *= lr_factor
            reg = phasorf.Parseval_Loss() * TV_weight_density

            total_loss = total_loss + reg
            summary_writer.add_scalar('train/reg_tv_density',
                                      reg.detach().item(),
                                      global_step=iteration)

        if TV_weight_app > 0:
            TV_weight_app *= lr_factor
            raise NotImplementedError('not implemented')

        if mask_learning:
            mask_loss = ((phasorf.num_unmasked_den + phasorf.num_unmasked_app - phasorf.target_param) ** 2)**(0.5)
            total_loss + mask_loss * 1e-9

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss = loss.detach().item()

        PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))
        summary_writer.add_scalar('train/PSNR', PSNRs[-1],
                                  global_step=iteration)
        summary_writer.add_scalar('train/mse', loss, global_step=iteration)

        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f'Iteration {iteration:05d}:'
                f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'
                f' mse = {loss:.6f} tv_loss = {reg.detach().item():.10f}')

            if iteration % (args.progress_refresh_rate * 10) == 0:
                logger.info(f"Iter {iteration}: {float(np.mean(PSNRs))}   tv_loss = {reg.detach().item():.10f}")
            PSNRs = []

        if iteration % args.vis_every == args.vis_every - 1 and args.N_vis!=0:
            PSNRs_test = evaluation(test_dataset,phasorf, args, renderer, 
                                    f'{logfolder}/imgs_vis/',
                                    prtx=f'{iteration:06d}_',
                                    N_samples=nSamples, N_vis=args.N_vis,
                                    white_bg = white_bg, ndc_ray=ndc_ray, 
                                    compute_extra_metrics=args.compute_extra_metric)
            print(np.mean(PSNRs_test))
            summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test),
                                      global_step=iteration)
            logger.info(f"Iter {iteration}, test psnr: {np.mean(PSNRs_test)}")


        # # TODO: to accelerate 
        if update_AlphaMask_list is not None \
                and iteration in update_AlphaMask_list:

            # update volume resolution
            # if reso_cur[0] * reso_cur[1] * reso_cur[2] < 256**3:
            #     reso_mask = reso_cur

            # new_aabb = phasorf.updateAlphaMask(phasorf.domain_min, phasorf.domain_max, tuple(reso_mask))
            # print("al mask!!")
            # print(reso_mask, new_aabb)
            # logger.info(f"reso_mask: {reso_mask}")

            # if bbox_only:
            #     return new_aabb

            # if return_bbox:
            #     return (new_aabb[1]-new_aabb[0]).prod().cpu().numpy()

            if iteration == update_AlphaMask_list[0]:
                # use tight aabb already
                # phasorf.shrink(new_aabb)
                if args.TV_weight_density_reset >= 0:
                    TV_weight_density = args.TV_weight_density_reset
                    print(f'TV weight density reset to '
                          f'{args.TV_weight_density_reset}')
                    L1_weight = 4e-5

            # if not args.ndc_ray and iteration == update_AlphaMask_list[1]:
            #     # filter rays outside the bbox
            #     # allrays,allrgbs = phasorf.filtering_rays(allrays,allrgbs, bbox_only=True)   # 여기서 alphamask 사용.
            #     allrays,allrgbs = phasorf.filtering_rays(allrays,allrgbs)   # 여기서 alphamask 사용.
            #     trainingSampler = SimpleSampler(allrgbs.shape[0],
            #                                     args.batch_size)
            #     allrays = allrays.cuda()
            #     allrgbs = allrgbs.cuda()

        if upsamp_list is not None and iteration in upsamp_list:
            n_voxels = N_voxel_list.pop(0)
            irregular_reso = N_to_reso(n_voxels, phasorf.aabb)
            if var_split:
                reso_cur = [math.ceil(irregular_reso[i] / args.block_split[i]) * args.block_split[i] for i in range(len(irregular_reso))]
            else:
                reso_cur = [math.ceil(reso / args.block_split) * args.block_split for reso in irregular_reso]

            print(f"calculated: {irregular_reso} ---> {reso_cur}")
            nSamples = min(args.nSamples,
                           cal_n_samples(reso_cur, args.step_ratio))
            phasorf.upsample_volume_grid(reso_cur)
            logger.info(f"upsample. reso to {reso_cur}, sample to {nSamples}")

            if args.lr_upsample_reset:  
                print("reset lr to initial")
                logger.info("reset lr to initial")
                lr_scale = 1 #0.1 ** (iteration / args.n_iters)
            else:
                lr_scale = args.lr_decay_target_ratio**(iteration/args.n_iters)
                print(f'lr set {lr_scale}')
                logger.info(f'lr set {lr_scale}')
            grad_vars = phasorf.get_optparam_groups(args.lr_init*lr_scale,
                                                    args.lr_basis*lr_scale)
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))


        # print size
        if iteration % 1000 == 0:
            if mask_learning:
                mask_thres_tensor[0] = phasorf.mask_thres
                if phasorf.mask_thres == 0.5:
                    m_thres = 0
                else:
                    m_thres = torch.log((1 - mask_thres_tensor) / mask_thres_tensor) * (-1)

            numel = sum([p.numel() for p in phasorf.parameters()])
            if hasattr(phasorf, 'den_mask'):
                numel -= sum([m.numel() for m in phasorf.den_mask])
                numel -= sum([m.numel() for m in phasorf.app_mask])

            print(f'Total size: {numel*4/1_048_576:.4f}MB')
            logger.info(f'Total size: {numel*4/1_048_576:.4f}MB')
            if hasattr(phasorf, 'den_mask'):
                reduced = sum([d.numel() * (m < m_thres).float().mean()
                            for d, m in zip(phasorf.den, phasorf.den_mask)]) \
                        + sum([d.numel() * (m < m_thres).float().mean()
                            for d, m in zip(phasorf.app, phasorf.app_mask)])
                print(f'reduced size: {(numel - reduced)*4/1_048_576:.4f}MB')
                logger.info(f'reduced size: {(numel - reduced)*4/1_048_576:.4f}MB')
        
        if mask_schedule:
            if iteration in mask_iter:
                phasorf.mask_thres =  mask_thres_list.pop(0)
                mask_thres_tensor[0] = phasorf.mask_thres
                if phasorf.mask_thres == 0.5:
                    m_thres = 0
                else:
                    m_thres = torch.log((1 - mask_thres_tensor) / mask_thres_tensor) * (-1)
                logger.info(f"mask threshold = {phasorf.mask_thres}, m_thres = {m_thres}")
                logger.info(f"split {phasorf.resolution} with {phasorf.block_split} blocks. Block res = {phasorf.block_resolution} and freq is {phasorf.n_freq}. net stride: {phasorf.network_strides[1]}")


    phasorf.save(f'{logfolder}/{args.expname}.th')
    # test
    if mask_learning:
        mask_thres_tensor[0] = phasorf.mask_thres
        if phasorf.mask_thres == 0.5:
            m_thres = 0
        else:
            m_thres = torch.log((1 - mask_thres_tensor) / mask_thres_tensor) * (-1)
    numel = sum([p.numel() for p in phasorf.parameters()])
    if hasattr(phasorf, 'den_mask'):
        numel -= sum([m.numel() for m in phasorf.den_mask])
        numel -= sum([m.numel() for m in phasorf.app_mask])

    print(f'Total size: {numel*4/1_048_576:.4f}MB')
    logger.info(f'Total size: {numel*4/1_048_576:.4f}MB')
    if hasattr(phasorf, 'den_mask'):
        reduced = sum([d.numel() * (m < m_thres).float().mean()
                       for d, m in zip(phasorf.den, phasorf.den_mask)]) \
                + sum([d.numel() * (m < m_thres).float().mean()
                       for d, m in zip(phasorf.app, phasorf.app_mask)])
        print(f'reduced size: {(numel - reduced)*4/1_048_576:.4f}MB')
        logger.info(f'reduced size: {(numel - reduced)*4/1_048_576:.4f}MB')
        logger.info(f'used m_thres as {m_thres}')

    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train',
                                downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset,phasorf, args, renderer,
                                f'{logfolder}/imgs_train_all/', N_vis=-1,
                                N_samples=-1, white_bg=white_bg,
                                ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} '
              f'<========================')
        logger.info(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} '
              f'<========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        PSNRs_test = evaluation(test_dataset, phasorf, args, renderer,
                                f'{logfolder}/imgs_test_all/', N_vis=-1,
                                N_samples=-1, white_bg=white_bg,
                                ndc_ray=ndc_ray,device=device)
        summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test),
                                  global_step=iteration)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} '
              f'<========================')
        logger.info(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} '
              f'<========================')

        if return_memory:
            memory = np.sum([v.numel() * v.element_size()
                             for k, v in phasorf.named_parameters()]) / 2**20
            return np.mean(PSNRs_test), memory

        return np.mean(PSNRs_test)

    if args.render_path:
        c2ws = test_dataset.render_path
        print('========>',c2ws.shape)
        os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset, phasorf, c2ws, renderer,
                        f'{logfolder}/imgs_path_all/', N_vis=-1, N_samples=-1,
                        white_bg=white_bg, ndc_ray=ndc_ray,device=device)
    
    if not args.render_test:
        PSNRs_test = evaluation(test_dataset,phasorf, args, renderer, 
                                f'{logfolder}/imgs_vis_all/', N_vis=10,
                                N_samples=nSamples, white_bg=white_bg,
                                ndc_ray=ndc_ray,
                                compute_extra_metrics=args.compute_extra_metric)
        if return_memory:
            memory = np.sum([v.numel() * v.element_size()
                             for k, v in phasorf.named_parameters()]) / 2**20
            return np.mean(PSNRs_test), memory

        return np.mean(PSNRs_test)



if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    seed = 2020233254
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    args = config_parser()
    print(args)

    rle = False
    if rle:
        if args.ckpt is None:
            print("requires ckpt")
            raise NotImplementedError
        else:
            rle_test(args)

    else :
        if args.export_mesh:
            export_mesh(args)

        if args.render_only and (args.render_test or args.render_path):
            render_test(args)
        else:
            reconstruction(args)

