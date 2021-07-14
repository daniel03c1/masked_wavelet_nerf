import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import configargparse
import imageio
import numpy as np
import tensorflow as tf
import time
import tqdm

from load_llff import load_llff_data
from models import generate_nerf_model, Embedder
from utils import *


parser = configargparse.ArgumentParser()
parser.add_argument('--config', is_config_file=True,
                    help='config file path')
parser.add_argument("--expname", type=str, help='experiment name')
parser.add_argument("--basedir", type=str, default='./logs/',
                    help='where to store ckpts and logs')
parser.add_argument("--datadir", type=str,
                    default='./data/llff/fern', help='input data directory')

# training options
parser.add_argument("--n_rand", type=int, default=32*32*4,
                    help='batch size (number of random rays per step)')
parser.add_argument("--lr", type=float,
                    default=5e-4, help='learning rate')
parser.add_argument("--lr_decay", type=int, default=250,
                    help='exponential learning rate decay (in 1000s)')
parser.add_argument("--chunk", type=int, default=1024*32,
                    help='number of rays processed in parallel')
parser.add_argument("--netchunk", type=int, default=1024*64,
                    help='number of pts sent through network in parallel')
parser.add_argument("--random_seed", type=int, default=None,
                    help='fix random seed for repeatability')

# pre-crop options
parser.add_argument("--precrop_iters", type=int, default=0,
                    help='number of steps to train on central crops')
parser.add_argument("--precrop_frac", type=float, default=.5,
                    help='fraction of img taken for central crops')    

# rendering options
parser.add_argument("--n_samples", type=int, default=64,
                    help='number of coarse samples per ray')
parser.add_argument("--n_importance", type=int, default=0,
                    help='number of additional fine samples per ray')
parser.add_argument("--perturb", type=float, default=1.,
                    help='set to 0. for no jitter, 1. for jitter')
parser.add_argument("--use_viewdirs", action='store_true',
                    help='use full 5D input instead of 3D')
parser.add_argument("--i_embed", type=int, default=0,
                    help='set 0 for default positional encoding, '
                         '-1 for none')
parser.add_argument("--multires", type=int, default=10,
                    help='log2 of max freq for positional encoding '
                         '(3D location)')
parser.add_argument("--multires_views", type=int, default=4,
                    help='log2 of max freq for positional encoding '
                         '(2D direction)')

parser.add_argument("--render_factor", type=int, default=0,
                    help='downsampling factor to speed up rendering, '
                         'set 4 or 8 for fast preview')

# dataset options
parser.add_argument("--testskip", type=int, default=8,
                    help='will load 1/N images from test/val sets, '
                         'useful for large datasets like deepvoxels')

parser.add_argument("--factor", type=int, default=8,
                    help='downsample factor for LLFF images')
parser.add_argument("--no_ndc", action='store_true',
                    help='do not use normalized device coordinates '
                         '(set for non-forward facing scenes)')
parser.add_argument("--lindisp", action='store_true',
                    help='sampling linearly in disparity rather than depth')
parser.add_argument("--spherify", action='store_true',
                    help='set for spherical 360 scenes')
parser.add_argument("--llffhold", type=int, default=8,
                    help='will take every 1/N images as LLFF test set, '
                         'paper uses 8')

# logging/saving options
parser.add_argument("--i_print",   type=int, default=100,
                    help='frequency of console printout and metric loggin')
parser.add_argument("--i_img",     type=int, default=500,
                    help='frequency of tensorboard image logging')
parser.add_argument("--i_weights", type=int, default=10000,
                    help='frequency of weight ckpt saving')
parser.add_argument("--i_testset", type=int, default=50000,
                    help='frequency of testset saving')
parser.add_argument("--i_video",   type=int, default=50000,
                    help='frequency of render_poses video saving')


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'."""
    inputs_flat = tf.reshape(inputs, [-1, inputs.shape[-1]])

    embedded = embed_fn(inputs_flat)
    if viewdirs is not None:
        input_dirs = tf.broadcast_to(viewdirs[:, None], inputs.shape)
        input_dirs_flat = tf.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = tf.concat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = tf.reshape(outputs_flat,
                         [*inputs.shape[:-1], outputs_flat.shape[-1]])
    return outputs


def apply_embedder(points, views, points_emb, views_emb):
    return tf.concat([points_emb(points), views_emb(views)], -1)


if __name__ == '__main__':
    args = parser.parse_args()

    n_iters = 1000000
    
    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        tf.compat.v1.set_random_seed(args.random_seed)

    # loading dataset
    images, poses, bds, pts_arr, pts_rgb_arr, vis_arr, z_vals = load_llff_data(
        args.datadir, bd_factor=0.75)

    height, width, focal = poses[0, :3, -1]
    height = int(height)
    width = int(width)
    hwf = [height, width, focal]
    poses = poses[:, :3, :4]

    # no ndc
    near = tf.reduce_min(bds) * .9
    far = tf.reduce_max(bds) * 1.
    print(f'NEAR: {near:.5f}, FAR: {far:.5f}')

    pos_embedder = Embedder(args.multires)
    views_embedder = Embedder(args.multires_views)
    model = generate_nerf_model((args.multires*2+1)*3,
                                (args.multires_views*2+1)*3, use_views=True)

    # defining an optimizer
    lr = args.lr
    if args.lr_decay > 0:
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            lr, decay_steps=args.lr_decay * 1000, decay_rate=0.1)
    optimizer = tf.keras.optimizers.Adam(lr)

    xs = apply_embedder(pts_arr, tf.zeros_like(pts_arr),
                        pos_embedder, views_embedder)
    ys = tf.concat([pts_rgb_arr, tf.ones([*xs.shape[:-1], 1])], -1)

    dataset = tf.data.Dataset.from_tensor_slices((xs, ys))
    dataset = dataset.cache().repeat(64).shuffle(4096).batch(1024)

    std = tf.convert_to_tensor(pts_arr.std(axis=0, keepdims=True))
    mean = tf.convert_to_tensor(pts_arr.mean(axis=0, keepdims=True))

    @tf.function
    def add_noise(x, y):
        dummy_x = tf.random.normal(tf.shape(x[..., :3])) * std + mean
        dummy_x = apply_embedder(dummy_x, tf.zeros_like(dummy_x),
                                 pos_embedder, views_embedder)
        dummy_y = tf.zeros_like(y)
        x = tf.concat([x, dummy_x], 0)
        y = tf.concat([y, dummy_y], 0)
        return x, y

    @tf.function
    def remove_high_frequency(x, y):
        mask = tf.concat([x[..., :33]*0 + 1, x[..., 33:]*0], -1)
        return x * mask, y

    dataset = dataset.map(add_noise).map(remove_high_frequency)

    with tqdm.tqdm(dataset) as pbar:
        for x, y in pbar:
            with tf.GradientTape() as tape:
                n_samples = x.shape[0]
                pred = model(x)
                loss = tf.keras.losses.MAE(pred[:n_samples//2],
                                           y[:n_samples//2])
                loss += tf.keras.losses.MAE(pred[n_samples//2:, -1],
                                            0.1*tf.ones([n_samples//2]))

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            pbar.set_postfix({'loss': tf.reduce_mean(loss).numpy()})

    height, width, focal = height//8, width//8, focal/8
    import matplotlib.pyplot as plt

    for i in range(len(poses)):
        rays_o, rays_d = get_rays(height, width, focal, poses[i, :, :-1])
        points, z_vals = rays_to_points(rays_o, rays_d, near, far,
                                        args.n_samples)
        points = apply_embedder(points, tf.zeros_like(points),
                                pos_embedder, views_embedder)

        flat_points = tf.reshape(points, [-1, points.shape[-1]])
        raw = batchify(model, chunk=args.chunk)(flat_points)
        raw = tf.reshape(raw, [*points.shape[:-1], -1])
        rgb_map, depth_map = render_raw_outputs(raw, z_vals, rays_d)

        plt.imshow(rgb_map)
        plt.show()

    # 1. RANDOM RAY SAMPLING
    # [N, H, W, ro+rd+rgb, 3]

    for i in range(start, n_iters):
        time0 = time.time()

        # Sample random ray batch
        # Random over all images
        batch = rays_rgb[i_batch:i_batch+n_rand]  # [B, 2+1, 3*?]
        batch = tf.transpose(batch, [1, 0, 2])

        # batch_rays[i, n, xyz] = ray origin or direction, id, 3D position
        # target_s[n, rgb] = example_id, observed color.
        batch_rays, target_s = batch[:2], batch[2]

        i_batch += n_rand
        if i_batch >= rays_rgb.shape[0]:
            np.random.shuffle(rays_rgb)
            i_batch = 0

        #####  Core optimization loop  #####
        with tf.GradientTape() as tape:
            # Make predictions for color, disparity, accumulated opacity.
            rgb, disp, acc, extras = render(
                H, W, focal, chunk=args.chunk, rays=batch_rays,
                verbose=i < 10, retraw=True, **render_kwargs_train)

            # Compute MSE loss between predicted and true RGB.
            img_loss = img2mse(rgb, target_s)
            trans = extras['raw'][..., -1]
            loss = img_loss
            psnr = mse2psnr(img_loss)

            # Add MSE loss for coarse-grained model
            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target_s)
                loss += img_loss0
                psnr0 = mse2psnr(img_loss0)

        gradients = tape.gradient(loss, grad_vars)
        optimizer.apply_gradients(zip(gradients, grad_vars))

        dt = time.time()-time0
        #####           end            #####

        # Rest is logging
        if i % args.i_weights == 0:
            for k in models:
                save_weights(models[k], k, i)

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(
                basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            render_path(poses[i_test], hwf, args.chunk, render_kwargs_test,
                        gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')

        if i % args.i_print == 0 or i < 10:
            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))

