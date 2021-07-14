import imageio
import numpy as np
import tqdm
import os

import matplotlib.pyplot as plt


def load_llff_data(basedir, bd_factor=.75):
    """
    INPUTS
        basedir: str
        bd_factor: ???

    poses: [n_images, 3, 5] shaped array
    bds: [n_images, 2] shaped array
         min, max depths of each image
    """
    poses, bds, images = load_raw_data(basedir)
    pts_arr = np.load(os.path.join(basedir, 'pts_arr.npy')).astype('float32')
    pts_rgb_arr = np.load(os.path.join(basedir, 'pts_rgb_arr.npy'))
    vis_arr = np.load(os.path.join(basedir, 'vis_arr.npy'))
    z_vals = np.load(os.path.join(basedir, 'z_vals.npy')).astype('float32')

    if bd_factor:
        # rescale every coordinates so that min(bds) equals 1/bd_factor
        scale = 1. / (bds.min()*bd_factor)
        poses[:, :3, 3] *= scale
        pts_arr *= scale
        z_vals *= scale
        bds *= scale

    return images, poses, bds, pts_arr, pts_rgb_arr, vis_arr, z_vals


def load_raw_data(basedir, load_imgs=True):
    """
    INPUTS
        basedir: path
        load_imgs: if True, returns poses, boundaries, images
                   if False, returns poses, boundaries
    """
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy')) \
                  .astype('float32')

    poses = poses_arr[:, :-2].reshape([-1, 3, 5])
    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate(
        [poses[..., 1:2], -poses[..., 0:1], poses[..., 2:]], -1)
    bds = poses_arr[:, -2:]

    imgfiles = [os.path.join(basedir, 'images', f)
                for f in sorted(os.listdir(os.path.join(basedir, 'images')))
                if os.path.splitext(f)[-1] in ['.JPG', '.jpg', '.png']]
    print(imgfiles)

    if poses.shape[0] != len(imgfiles):
        raise ValueError(f'Mismatch between imgs {len(imgfiles)} '
                         f'and poses {poses.shape[-1]} !!!!')
    
    if not load_imgs:
        return poses, bds
    
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)
 
    imgs = None
    imgs = [(imread(f)/255.).astype(np.float32) for f in tqdm.tqdm(imgfiles)]
    imgs = np.stack(imgs, 0)

    return poses, bds, imgs


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(at, up, pos):
    n = normalize(at) # z: eye - at
    u = normalize(np.cross(up, n)) # x
    v = normalize(np.cross(n, u)) # y
    return np.stack([u, v, n, pos], 1)


def points_to_camera(pts, c2w):
    w2c = c2w[:3, :3].T
    tt = np.matmul(w2c, (pts-c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt


def poses_avg(poses):
    """
    INPUTS
        poses: [n_images, 3, 5] shaped array
               camera positions of images

    center: the center of all cameras
    z: average z...
    up: average up...

    OUTPUTS
        c2w: c2w of average camera
    """
    center = poses[..., 3].mean(0)
    at = poses[..., 2].sum(0)
    up = poses[..., 1].sum(0)

    hwf = poses[0, :, -1:]
    c2w = np.concatenate([viewmatrix(at, up, center), hwf], 1)
    return c2w


def recenter_poses(poses):
    """
    INPUTS
        poses: [n_images, 3, 5]
    """
    poses_ = poses + 0

    # c2w [3, 5] -> [4, 4]
    c2w = poses_avg(poses) # c2w of average camera
    c2w = poses_34_to_44(c2w[:, :-1])
    w2c = np.linalg.inv(c2w)

    poses = poses_34_to_44(poses[..., :-1])

    poses = w2c @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


def poses_34_to_44(poses):
    assert poses.shape[-2] == 3 and poses.shape[-1] == 4
    n_dim = len(poses.shape)
    bottom = np.array([0, 0, 0, 1.], dtype=poses.dtype)
    bottom = np.reshape(bottom, [1]*(n_dim-1) + [4])
    bottom = np.tile(bottom, [*poses.shape[:-2], 1, 1])
    return np.concatenate([poses, bottom], -2)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from utils import get_rays

    basedir = './data/nerf_llff_data/fern/'
    images, poses, bds, pts_arr, pts_rgb_arr, vis_arr, z_vals = load_llff_data(
        basedir)

    height, width, focal = poses[0, :, -1] / 8
    for i in range(len(poses)):
        rays_o, rays_d = get_rays(height, width, focal, poses[i, :, :-1])
        pts = pts_arr[vis_arr[..., i] > 0]
        similarities = tf.keras.losses.cosine_similarity(
                rays_d[..., None, :], (pts - rays_o[0, 0])[None, None, :])
        img = pts_rgb_arr[vis_arr[..., i] > 0][similarities.numpy().argmin(axis=-1)]
        plt.imshow(img)
        plt.show()

    for i in range(len(poses)):
        pts = pts_arr # [vis_arr[..., i] > 0]
        rgb = pts_rgb_arr # [vis_arr[..., i] > 0]

        new_points = points_to_camera(pts, poses[i])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(new_points[..., 0], new_points[..., 1], c=rgb)
        ax.set_aspect('equal')
        plt.show()

