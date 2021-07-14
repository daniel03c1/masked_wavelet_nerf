import numpy as np
import tensorflow as tf


def img_to_mse(x, y):
    return tf.reduce_mean(tf.square(x - y))


def mse_to_psnr(x):
    return -10. * tf.log(x) / tf.log(10.)


def float_to_uint8(x: tf.Tensor):
    assert x.dtype == tf.float32
    return tf.cast(255 * tf.clip_by_value(x, 0, 1), tf.uint8)


def batchify(fn, chunk=None):
    if chunk is None:
        chunk = len(inputs)

    def batchified(inputs):
        return tf.concat(
            [fn(inputs[i:i+chunk]) for i in range(0, len(inputs), chunk)], 0)
    return batchified


def get_rays(height, width, focal, c2w):
    """
    get ray origins, directions from a pinhole camera

    INPUTS
        height: scalar
        width: scalar
        focal: scalar
        c2w: [4, 4]

    OUTPUTS
        rays_o: [height, width, 3]
        rays_d: [height, width, 3]
    """
    i, j = tf.meshgrid(tf.range(width, dtype=tf.float32),
                       tf.range(height, dtype=tf.float32), indexing='xy')
    dirs = tf.stack([(i-width/2)/focal, -(j-height/2)/focal, -tf.ones_like(i)],
                    -1)
    rays_d = tf.reduce_sum(dirs[..., tf.newaxis, :] * c2w[:3, :3], -1)
    rays_o = tf.broadcast_to(c2w[:3, -1], tf.shape(rays_d))
    return rays_o, rays_d


def rays_to_points(rays_o, rays_d, near, far, n_samples,
                   sampling_policy='linear', perturb=True):
    """
    INPUTS
        rays_o: [..., 3], origins
        rays_d: [..., 3], directions with magnitudes
        near, far: [...]
        sampling_policy: ('linear', 'inverse_depth')
        perturb: bool

    OUTPUTS
        points: [..., n_samples, 3]
    """
    near = near[..., None]
    far = far[..., None]
    z_vals = tf.linspace(0., 1., n_samples)
    if sampling_policy == 'linear':
        z_vals = near * (1-z_vals) + far * z_vals
    elif sampling_policy == 'inverse_depth':
        z_vals = 1 / ((1-z_vals)/near + z_vals/far)
    else:
        raise ValueError(f'invalid sampling policy ({sampling_policy})')

    if perturb:
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = tf.concat([mids, z_vals[..., -1:]], -1)
        lower = tf.concat([z_vals[..., :1], mids], -1)

        z_vals = lower + (upper-lower)*tf.random.uniform(z_vals.shape)

    # [..., n_samples, 3]
    points = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., None]
    return points, z_vals


def render_raw_outputs(raw, z_vals, rays_d,
                       feats_activation=tf.nn.sigmoid,
                       sigma_activation=tf.nn.relu,
                       background=None):
    """
    INPUTS
        raw: [..., n_samples, N+1]
             last channel will be interpreted as sigma
             for conventional RGB format, N equals 3
        z_vals: [..., n_samples]
        rays_d: [..., 3]
                direction of each ray
        feats_activation: activation function for N features
        sigma_activation: activation function for opacity
        background: if not None, remaining ...

    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
      depth_map: [num_rays]. Estimated distance to object.
    """
    feats = feats_activation(raw[..., :-1])
    sigma = sigma_activation(raw[..., -1])

    dists = tf.concat([z_vals[..., 1:] - z_vals[..., :-1], 
                       tf.broadcast_to([1e10], z_vals[..., :1].shape)], -1)
    dists = dists * tf.linalg.norm(rays_d[..., None, :], axis=-1)

    # "weights" means T(t)sigma(t) in the original equation
    alpha = 1. - tf.exp(-sigma * dists)
    weights = alpha * tf.math.cumprod(1 -alpha + 1e-10, axis=-1, exclusive=True)

    feats_map = tf.reduce_sum(weights[..., None] * feats, axis=-2)
    depth_map = tf.reduce_sum(weights * z_vals, axis=-1)

    if background is not None:
        acc_map = tf.reduce_sum(weights, -1)
        feats_map = feats_map + (1.-acc_map[..., None]) * background

    return feats_map, depth_map

