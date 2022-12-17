import argparse
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', required=True, type=str)
parser.add_argument('--dataset', required=True, type=str)
parser.add_argument('--dwt_level', type=int, default=4)
parser.add_argument('--trans_func', type=str, default='bior4.4')


if __name__ == '__main__':
    args = parser.parse_args()
    print(str(vars(args)))

    datadirs = {'nerf': ['../nerf_synthetic/chair',
                         '../nerf_synthetic/drums',
                         '../nerf_synthetic/ficus',
                         '../nerf_synthetic/hotdog',
                         '../nerf_synthetic/lego',
                         '../nerf_synthetic/materials',
                         '../nerf_synthetic/mic',
                         '../nerf_synthetic/ship'],
                'nsvf': ['../Synthetic_NSVF/Bike',
                         '../Synthetic_NSVF/Lifestyle',
                         '../Synthetic_NSVF/Palace',
                         '../Synthetic_NSVF/Robot',
                         '../Synthetic_NSVF/Spaceship',
                         '../Synthetic_NSVF/Steamtrain',
                         '../Synthetic_NSVF/Toad',
                         '../Synthetic_NSVF/Wineholder'],
                'tank': ['../TanksAndTemple/Barn',
                         '../TanksAndTemple/Caterpillar',
                         '../TanksAndTemple/Family',
                         '../TanksAndTemple/Ignatius',
                         '../TanksAndTemple/Truck']}
    dataset_names = {'nerf': 'blender',
                     'nsvf': 'nsvf',
                     'tank': 'tankstemple'}

    args.dataset = args.dataset.lower()

    for datadir in datadirs[args.dataset]:
        for mw in [1e-10]: # , 5e-11]:
            name = f'{args.dataset}_{datadir.split("/")[-1]}_{mw}' \
                   f'_{args.dwt_level}_{args.trans_func}'
            os.system(f'CUDA_VISIBLE_DEVICES={args.gpu} python3 train.py '
                      f'--use_mask --grid_bit=8 --mask_weight={mw} '
                      f'--config=configs/default.txt '
                      f'--use_dwt --dwt_level={args.dwt_level} '
                      f'--trans_func={args.trans_func} '
                      f'--expname={name} '
                      f'--datadir={datadir} '
                      f'--dataset_name={dataset_names[args.dataset]}')

