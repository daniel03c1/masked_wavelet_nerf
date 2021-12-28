import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import argparse
import imageio
import numpy as np
import tensorflow as tf
import time
import tqdm

from load_llff import load_llff_data
from models import generate_nerf_model, Embedder
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument("--expname", type=str, help='experiment name')
parser.add_argument("--basedir", type=str, default='./logs/',
                    help='where to store ckpts and logs')
parser.add_argument("--datadir", type=str,
                    default='./data/llff/fern', help='input data directory')

# training options
parser.add_argument("--n_iters", type=int, default=10000,
                    help='the number of train iterations (steps)')
parser.add_argument("--n_rays", type=int, default=4096,
                    help='batch size (number of random rays per step)')
parser.add_argument("--lr", type=float,
                    default=5e-4, help='learning rate')
parser.add_argument("--chunk", type=int, default=1024*128,
                    help='number of rays processed in parallel')

# rendering options (model)
parser.add_argument("--n_samples", type=int, default=64,
                    help='number of coarse samples per ray')
parser.add_argument("--n_stages", type=int, default=2)
parser.add_argument("--perturb", type=float, default=1.,
                    help='set to 0. for no jitter, 1. for jitter')
parser.add_argument("--use_viewdirs", action='store_true',
                    help='use full 5D input instead of 3D')
parser.add_argument("--multires", type=int, default=10,
                    help='log2 of max freq for positional encoding '
                         '(3D location)')
parser.add_argument("--multires_views", type=int, default=4,
                    help='log2 of max freq for positional encoding '
                         '(2D direction)')

# dataset options
parser.add_argument("--factor", type=int, default=8,
                    help='downsample factor for LLFF images')
parser.add_argument("--lindisp", action='store_true',
                    help='sampling linearly in disparity rather than depth')
parser.add_argument("--llffhold", type=int, default=8,
                    help='will take every 1/N images as LLFF test set, '
                         'paper uses 8')


def apply_embedder(points, views, points_emb, views_emb):
    return tf.concat([points_emb(points), views_emb(views)], -1)


if __name__ == '__main__':
    args = parser.parse_args()

    # loading dataset
    images, poses, bds = load_llff_data(args.datadir, bd_factor=0.75)
    height, width, focal = poses[0, :3, -1]
    height = int(height / args.factor)
    width = int(width / args.factor)
    focal = focal / args.factor
    images = tf.nn.avg_pool2d(images, args.factor, args.factor, 'VALID')
    poses = tf.convert_to_tensor(poses[..., :3, :4])

    train_indices = tf.convert_to_tensor(
        [i for i in range(len(images)) if i % args.llffhold])
    test_indices = tf.convert_to_tensor(
        [i for i in range(len(images)) if i not in train_indices])

    # no ndc
    near = tf.reduce_min(bds) * .9
    far = tf.reduce_max(bds) * 1.
    print(f'NEAR: {near:.5f}, FAR: {far:.5f}')

    pos_embedder = Embedder(args.multires)
    views_embedder = Embedder(args.multires_views)

    model = generate_nerf_model((args.multires*2+1)*3,
                                (args.multires_views*2+1)*3, use_views=True)
    optimizer = tf.keras.optimizers.Adam(args.lr)

    grids = [get_grid(height, width, focal, 2**i)
             for i in reversed(range(args.n_stages))]

    with tqdm.tqdm(range(args.n_iters)) as loop:
        for i in loop:
            # random ray sampling
            j = tf.random.uniform(
                [args.n_rays], maxval=len(train_indices), dtype=tf.int32)
            x = tf.random.uniform([args.n_rays], maxval=width, dtype=tf.int32)
            y = tf.random.uniform([args.n_rays], maxval=width, dtype=tf.int32)

            true_rgb = tf.gather_nd(
                images,
                tf.stack([tf.gather(train_indices, j), y, x], -1))

            rays_o, rays_d = get_rays(
                tf.gather_nd(grids[-1], tf.stack([y, x], -1)),
                tf.gather(poses, j))
            points, z_vals = rays_to_points(rays_o, rays_d, near, far,
                                            args.n_samples)
            viewdirs = tf.broadcast_to(rays_d[..., None, :],
                                       points.shape)
            viewdirs = tf.math.l2_normalize(viewdirs, axis=-1)
            points = apply_embedder(points, viewdirs,
                                    pos_embedder, views_embedder)

            with tf.GradientTape() as tape:
                shape = points.shape
                points = tf.reshape(points, [-1, shape[-1]])
                raw = batchify(model, chunk=args.chunk)(points)
                raw = tf.reshape(raw, [*shape[:-1], -1])
                pred_rgb, _ = render_raw_outputs(raw, z_vals, rays_d)

                img_loss = img_to_mse(pred_rgb, true_rgb)
                loss = img_loss
                psnr = mse_to_psnr(img_loss)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables))

            loop.set_postfix({"psnr": psnr.numpy()})

            # TODO: model evaluation
