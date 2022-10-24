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
from models.modules import MLP, Softplus, EmptyMLP
from utils import *

# for search
import ray
from ray import tune
from ray.air import session
from ray.tune.search.optuna import OptunaSearch
from sympy import Symbol
from sympy.solvers import solve


parser = configargparse.ArgumentParser()
parser.add_argument('--config', is_config_file=True,
                    help='config file path')
parser.add_argument("--expname", type=str, help='experiment name')
parser.add_argument("--save_path", type=str, default='./log',
                    help='where to store ckpts and logs')
parser.add_argument("--datadir", type=str,
                    default='/codes/nerf_synthetic/chair',
                    help='input data directory')

# dataset
parser.add_argument('--downsample_train', type=float, default=1.0)
parser.add_argument('--downsample_test', type=float, default=1.0)
parser.add_argument('--dataset_name', type=str, default='blender',
                    choices=['blender', 'llff', 'nsvf', 'dtu',
                             'tankstemple', 'own_data'])

# network decoder
parser.add_argument("--den_res", type=int, default=300)
# parser.add_argument("--den_chan", type=int, default=16)
parser.add_argument("--app_res", type=int, default=300)
# parser.add_argument("--app_chan", type=int, default=48)
parser.add_argument("--feat_dim", type=int, default=27)

parser.add_argument("--include_feat", action='store_true')
parser.add_argument("--include_pos", action='store_true')
parser.add_argument("--include_view", action='store_true')
parser.add_argument("--pos_pe", type=int, default=0,
                    help='number of pe for pos')
parser.add_argument("--view_pe", type=int, default=2,
                    help='number of pe for view')
parser.add_argument("--feat_pe", type=int, default=2,
                    help='number of pe for features')
parser.add_argument("--n_layers", type=int, default=3)
parser.add_argument("--hidden_dim", type=int, default=128,
                    help='hidden feature channel in MLP')

# training hyperparameters
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--n_iters", type=int, default=500) # 30000)
parser.add_argument("--lr_init", type=float, default=0.02)
parser.add_argument("--tv_weight", type=float, default=0.0)

# rendering options
parser.add_argument('--ndc_ray', action='store_true')
parser.add_argument('--n_samples', type=int, default=1036,
                    help='sample point each ray, pass 1e6 if automatic adjust')

# search hyperparameters
parser.add_argument('--den_ratio', type=float, default=0.3,
                    help='the ratio of number of parameters of den_ratio '
                         'compared to the total number of parameters')
parser.add_argument('--target_sz', type=float, default=5.,
                    help='target neural network size (MB)')
parser.add_argument('--num_samples', type=int, default=128,
                    help='the total number of samples used for searching')
args = parser.parse_args()


def main(config):
    for k, v in config.items():
        setattr(args, k, v)

    n_samples = args.n_samples
    ndc_ray = args.ndc_ray

    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train',
                            downsample=args.downsample_train, is_stack=False,
                            verbose=False)
    test_dataset = dataset(args.datadir, split='test',
                           downsample=args.downsample_train, is_stack=True,
                           verbose=False)

    allrays = train_dataset.all_rays.cuda(non_blocking=True)
    allrgbs = train_dataset.all_rgbs.cuda(non_blocking=True)

    white_bg = train_dataset.white_bg
    near, far = train_dataset.near_far
    bbox = train_dataset.scene_bbox.cuda()

    # TODO(daniel): loading weights

    total_numel = args.target_sz * 1_048_576 * 8 / 32
    den_numel = args.den_ratio * total_numel
    x = Symbol('x')
    eq = 3 * x * (args.den_res + 1) * args.den_res - den_numel
    args.den_chan = int(max(solve(eq)))
    assert args.den_chan > 0

    app_numel = total_numel - den_numel
    x = Symbol('x')
    in_size = args.feat_dim * (args.include_feat + 2 * args.feat_pe) \
            + 3 * (args.include_pos + 2 * args.pos_pe) \
            + 3 * (args.include_view + 2 * args.view_pe)
    eq = 3 * x * (args.app_res + 1) * args.app_res + 3 * x * args.feat_dim \
       + in_size * ((args.n_layers == 1) * 3 + (args.n_layers > 1) * args.hidden_dim) + (args.n_layers - 1) * args.hidden_dim ** 2 + (args.n_layers > 1) * args.hidden_dim * 3 \
       - app_numel
    args.app_chan = int(max(solve(eq)))
    assert args.app_chan > 0

    # defining networks
    density_net = TwoStageNeuralField(
        TensoRF_VM(int(args.den_res), int(args.den_chan), 1), EmptyMLP()).cuda()
    appearance_net = TwoStageNeuralField(
        TensoRF_VM(int(args.app_res), int(args.app_chan), int(args.feat_dim)),
        MLP(int(args.feat_dim), include_feat=args.include_feat,
            include_pos=args.include_pos, include_view=args.include_view,
            feat_n_freq=int(args.feat_pe), pos_n_freq=int(args.pos_pe),
            view_n_freq=int(args.view_pe),
            hidden_dim=int(args.hidden_dim), n_layers=int(args.n_layers),
            out_activation='sigmoid')).cuda()

    renderer = Renderer(density_net, appearance_net,
                        n_samples_per_ray=n_samples,
                        bounding_box=bbox,
                        near=near, far=far, white_bg=white_bg,
                        density_activation='softplus')

    optimizer = torch.optim.Adam(
        [{'params': density_net.parameters(), 'lr': args.lr_init},
         {'params': (appearance_net.first_stage.planes,
                     appearance_net.first_stage.vectors),'lr': args.lr_init},
         {'params': appearance_net.first_stage.basis_mat.parameters(),
          'lr': 1e-3},
         {'params': appearance_net.second_stage.parameters(), 'lr': 1e-3}],
        betas=(0.9, 0.99))
    scheduler = get_cos_warmup_scheduler(optimizer, args.n_iters, 0,
                                         min_ratio=0.1)
    # scaler = torch.cuda.amp.GradScaler()

    psnr = 0
    for i in range(args.n_iters):
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

        psnr = 0.9 * psnr + 0.1 * mse2psnr_np(loss.detach().item())

    # Evaluation
    # PSNRs_test = [p.cpu() for p in renderer.evaluation(test_dataset)]
    session.report({'psnr': psnr,
                    'size': renderer.compute_bits() / 8_388_608,
                    'config/den_chan': args.den_chan,
                    'config/app_chan': args.app_chan})


if __name__ == '__main__':
    search_space = {
        'lr_init': tune.qloguniform(1e-4, 1e-1, 1e-4),
        'den_ratio': tune.uniform(0.1, 0.5),
        'den_res': tune.quniform(100, 300, 10),
        'app_res': tune.quniform(100, 300, 10),
        'feat_dim': tune.quniform(1, 64, 1),
        'include_feat': tune.choice([True, False]),
        'feat_pe': tune.quniform(0, 6, 1),
        'include_pos': tune.choice([True, False]),
        'pos_pe': tune.quniform(0, 6, 1),
        'include_view': tune.choice([True, False]),
        'view_pe': tune.quniform(0, 6, 1),
        'n_layers': tune.quniform(1, 5, 1),
        'hidden_dim': tune.quniform(8, 128, 4),
    }

    algo = OptunaSearch()

    tuner = tune.Tuner(
        tune.with_resources(tune.with_parameters(main),
                            resources={'cpu': 4, 'gpu': 1}),
        tune_config=tune.TuneConfig(metric='psnr', mode='max', search_alg=algo,
                                    num_samples=args.num_samples),
        param_space=search_space)

    results = tuner.fit()
    results.get_dataframe().to_csv(f'results({args.target_sz}).csv')

