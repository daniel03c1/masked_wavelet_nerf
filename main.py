import configargparse
import datetime
import json
import os
import random
import sys
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from dataLoader import dataset_dict
from models import TwoStageNeuralField, Renderer
from models.grid_based import TensoRF_VM
from models.modules import MLP, Softplus, EmptyMLP, LRAmplifier
from utils import *


parser = configargparse.ArgumentParser()
parser.add_argument('--config', is_config_file=True,
                    help='config file path')
parser.add_argument("--expname", type=str, help='experiment name')
parser.add_argument("--save_path", type=str, default='./log',
                    help='where to store ckpts and logs')
parser.add_argument("--datadir", type=str, default='../nerf_synthetic/chair',
                    help='input data directory')

# dataset
parser.add_argument('--downsample_train', type=float, default=1.0)
parser.add_argument('--downsample_test', type=float, default=1.0)
parser.add_argument('--dataset_name', type=str, default='blender',
                    choices=['blender', 'llff', 'nsvf', 'dtu',
                             'tankstemple', 'own_data'])

# network decoder
parser.add_argument("--pos_pe", type=int, default=0,
                    help='number of pe for pos')
parser.add_argument("--view_pe", type=int, default=2,
                    help='number of pe for view')
parser.add_argument("--feat_pe", type=int, default=2,
                    help='number of pe for features')
parser.add_argument("--hidden_dim", type=int, default=128,
                    help='hidden feature channel in MLP')

# training hyperparameters
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--n_iters", type=int, default=30000)
parser.add_argument("--lr_init", type=float, default=0.02)
parser.add_argument("--tv_weight", type=float, default=0.0)

# rendering options
parser.add_argument('--ndc_ray', action='store_true')
parser.add_argument('--n_samples', type=int, default=1024,
                    help='sample point each ray, pass 1e6 if automatic adjust')


def main(args):
    n_samples = args.n_samples
    ndc_ray = args.ndc_ray

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

    # TODO(daniel): loading weights

    # defining networks
    density_net = TwoStageNeuralField(
        TensoRF_VM(300, 16, 1), EmptyMLP()).cuda()
    appearance_net = TwoStageNeuralField(
        TensoRF_VM(300, 48, 27),
        MLP(27, include_pos=False, include_view=True,
            feat_n_freq=args.feat_pe, pos_n_freq=args.pos_pe,
            view_n_freq=args.view_pe,
            hidden_dim=args.hidden_dim, out_activation='sigmoid')).cuda()

    renderer = Renderer(density_net, appearance_net,
                        n_samples_per_ray=n_samples,
                        bounding_box=bbox,
                        near=near, far=far, white_bg=white_bg,
                        density_activation='softplus')
    print(renderer)
    print(f'SIZE: {renderer.compute_bits() / 8_388_608 :.2f} MB')

    optimizer = torch.optim.Adam(
        [{'params': density_net.parameters(), 'lr': 0.02},
         {'params': (appearance_net.first_stage.planes,
                     appearance_net.first_stage.vectors),'lr': 0.02},
         {'params': appearance_net.first_stage.basis_mat.parameters(),
          'lr': 1e-3},
         {'params': appearance_net.second_stage.parameters(), 'lr': 1e-3}],
        betas=(0.9, 0.99))
    scheduler = get_cos_warmup_scheduler(optimizer, args.n_iters, 0,
                                         min_ratio=0.1)
    # scaler = torch.cuda.amp.GradScaler()

    pbar = tqdm(range(args.n_iters))
    for i in pbar:
        indices = torch.randint(len(allrays), (args.batch_size,)).cuda()
        rays_train = torch.index_select(allrays, 0, indices)
        rgb_train = torch.index_select(allrgbs, 0, indices)

        optimizer.zero_grad()

        # with torch.cuda.amp.autocast(enabled=True):
        rgb_map, depth_map = renderer(rays_train)

        loss = F.mse_loss(rgb_map, rgb_train)

        # loss
        total_loss = loss
        if args.tv_weight > 0:
            total_loss += renderer.compute_tv() * args.tv_weight

        assert not torch.isnan(loss)
        # scaler.scale(total_loss).backward(retain_graph=True)

        total_loss.backward()
        optimizer.step()
        # scaler.unscale_(optimizer)
        # scaler.step(optimizer)
        # scaler.update()

        scheduler.step()

        psnr = mse2psnr_np(loss.detach().item())
        pbar.set_description(f'Iter {i:05d}: train={psnr:.3f}')

        if i + 1 in [500, 1000, 2500, 5000, 10000, 20000]:
            print()

    # Evaluation
    PSNRs_test = renderer.evaluation(test_dataset)
    print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} '
          f'<========================')


if __name__ == '__main__':
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = parser.parse_args()
    print(args)

    main(args)

