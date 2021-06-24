import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import seaborn as sns
import sklearn
import sklearn.cluster
import sklearn.manifold
import statsmodels.api as sm
import tensorflow as tf
import time
import tqdm
from sklearn.metrics.pairwise import cosine_similarity


'''
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(
                memory_limit=10240)])
    except RuntimeError as e:
        print(e)
'''


def posenc(x):
    rets = [x]
    for i in range(L_embed):
        for fn in [tf.sin, tf.cos]:
            rets.append(fn(2. ** i * x))
    return tf.concat(rets, -1)


L_embed = 6
embed_fn = posenc


def init_model(D=8, W=256):
    inputs = tf.keras.Input(shape=(3 + 3 * 2 * L_embed))
    outputs = inputs
    for i in range(D):
        outputs = tf.keras.layers.Dense(W, activation='relu')(outputs)

        if i % 4 == 0 and i > 0:
            outputs = tf.concat([outputs, inputs], -1)
    outputs = tf.keras.layers.Dense(4)(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def analyze_model(model, mins, maxs, n_samples=1024, D=8):
    inputs = tf.random.uniform([n_samples, 3])
    inputs = (maxs - mins) * inputs + mins
    inputs = embed_fn(inputs)
    outputs = [inputs, model.get_layer('dense')(inputs)]

    for i in range(1, D):
        if i % 4 == 1 and i > 1:
            outputs.append(model.get_layer(f'dense_{i}')(
                tf.concat([outputs[-1], inputs], -1)))
        else:
            outputs.append(model.get_layer(f'dense_{i}')(outputs[-1]))
    return outputs


def get_rays(H, W, focal, c2w):
    i, j = tf.meshgrid(tf.range(W, dtype=tf.float32), 
                       tf.range(H, dtype=tf.float32), indexing='xy')
    dirs = tf.stack([(i - W * .5) / focal, 
                     -(j - H * .5) / focal, 
                     -tf.ones_like(i)], -1)
    rays_d = tf.reduce_sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = tf.broadcast_to(c2w[:3, -1], tf.shape(rays_d))
    return rays_o, rays_d


def render_rays(network_fn, rays_o, rays_d, near, far, n_samples, rand=True):
    def batchify(fn, chunk=1024 * 16):
        return lambda inputs: tf.concat([fn(inputs[i:i + chunk]) 
                                 for i in range(0, inputs.shape[0], chunk)], 0)

    # Compute 3D query points
    z_vals = tf.linspace(near, far, n_samples)
    if rand:
        z_vals += tf.random.uniform(
            list(rays_o.shape[:-1]) + [n_samples]) * (far - near) / n_samples
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    # Run network
    pts_flat = tf.reshape(pts, [-1, 3])
    pts_flat = embed_fn(pts_flat)
    raw = batchify(network_fn)(pts_flat)
    raw = tf.reshape(raw, list(pts.shape[:-1]) + [4])

    # Compute opacities and colors
    sigma_a = tf.nn.relu(raw[..., 3])
    rgb = tf.math.sigmoid(raw[..., :3])

    # Do volume rendering
    dists = tf.concat([z_vals[..., 1:] - z_vals[..., :-1], 
                       tf.broadcast_to([1e10], z_vals[..., :1].shape)], -1)
    alpha = 1. - tf.exp(-sigma_a * dists)
    weights = alpha * tf.math.cumprod(1. - alpha + 1e-10, -1, exclusive=True)

    rgb_map = tf.reduce_sum(weights[..., None] * rgb, -2)
    depth_map = tf.reduce_sum(weights * z_vals, -1)
    acc_map = tf.reduce_sum(weights, -1)

    return rgb_map, depth_map, acc_map


def analyze_model(model, pts_mins, pts_maxs,
                  embed_fn,
                  n_samples=4096, D=8):
    inputs_ptr = tf.random.uniform([n_samples, 3])
    inputs_ptr = (pts_maxs - pts_mins) * inputs_ptr + pts_mins
    inputs_pts = embed_fn(inputs_ptr)

    names = [l.name for l in model.layers if 'dense' in l.name]
    outputs = [inputs_pts]
    for i in range(D):
        out = outputs[-1]
        if i == 5:
            out = tf.concat([inputs_pts, out], -1)
        outputs.append(model.get_layer(names[i])(out))

    # assume use_viewdirs is True
    outputs.append(model.get_layer(names[-1])(out))  # opacity

    return outputs


if __name__ == '__main__':
    data = np.load('tiny_nerf_data.npz')
    images = data['images']
    poses = data['poses']
    focal = data['focal']

    height, width = images.shape[1:3]

    testimg, testpose = images[101], poses[101]
    images = images[:100, ..., :3]
    poses = poses[:100]

    model = init_model()
    optimizer = tf.keras.optimizers.Adam(5e-4)

    n_samples = 128 # 64
    max_samples = 128 # 64
    n_iters = 1000 # 1000

    psnrs = []
    iternums = []
    i_plot = 25

    # test
    pts = []
    near, far = 2., 6.

    # for i in range(100):
    #     rays_o, rays_d = get_rays(*images.shape[:2], focal, poses[i])

    #     z_vals = tf.linspace(near, far, 2)
    #     pts.append(
    #         rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None])
    # pts = tf.concat(pts, axis=-2)
    # pts_mins = tf.reduce_min(pts, axis=(0, 1, 2))
    # pts_maxs = tf.reduce_max(pts, axis=(0, 1, 2))
    # model.summary()

    '''
    t = t0 = time.time()
    for i in range(n_iters + 1):
        img_i = np.random.randint(images.shape[0])

        ratio = i / n_iters
        target = images[img_i]
        pose = poses[img_i]

        H, W = target.shape[-3:-1]
        n_samples = max_samples
        n_samples = max_samples

        rays_o, rays_d = get_rays(H, W, focal, pose)
        x_shuffle = tf.random.shuffle(tf.range(H))[:int(H/2+(H/2)*ratio)]
        y_shuffle = tf.random.shuffle(tf.range(W))[:int(W/2+(W/2)*ratio)]
    
        # target = tf.gather(tf.gather(target, x_shuffle, axis=0),
        #                    y_shuffle, axis=1)
        # rays_o = tf.gather(tf.gather(rays_o, x_shuffle, axis=0),
        #                    y_shuffle, axis=1)
        # rays_d = tf.gather(tf.gather(rays_d, x_shuffle, axis=0),
        #                    y_shuffle, axis=1)

        with tf.GradientTape() as tape:
            rgb, depth, acc = render_rays(model, rays_o, rays_d, 
                                          near=2., far=6., 
                                          n_samples=n_samples, rand=True)
            loss = tf.reduce_mean(tf.square(rgb - target))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if i % i_plot == 0:
            # outputs = analyze_model(model, pts_mins, pts_maxs)
            # zeros = [tf.reduce_mean(tf.cast(o == 0, dtype=tf.float32)).numpy()
            #          for o in outputs[1:-1]]

            # Render the holdout view for logging
            rays_o, rays_d = get_rays(height, width, focal, testpose)
            rgb, depth, acc = render_rays(model, rays_o, rays_d, 
                                          near=2., far=6., 
                                          n_samples=n_samples)
            loss = tf.reduce_mean(tf.square(rgb - testimg))
            psnr = -10. * tf.math.log(loss) / tf.math.log(10.)

            print(f"{i} {(time.time() - t)/i_plot} secs per iter - psnr: {psnr}")
            t = time.time()

            psnrs.append(psnr)
            iternums.append(i)

    model.save('tiny.h5')
    '''
    model.load_weights('tiny.h5')

    def randomize_weights(model, layer, ratio):
        model.load_weights('tiny.h5')
        weights = model.get_weights()
        assert layer < len(weights) // 2

        n_units = weights[layer*2].shape[-1]
        masks = np.random.uniform(size=n_units) > ratio
        weights[layer*2] *= masks[np.newaxis, :]
        weights[layer*2+1] *= masks

        model.set_weights(weights)

    rgbs = []
    psnrs = []

    for i in tqdm.tqdm(range(len(model.get_weights())//2+1)):
        if i == 0: # No masking
            randomize_weights(model, i, 0)
        else:
            randomize_weights(model, i-1, 0.5)

        rays_o, rays_d = get_rays(height, width, focal, testpose)
        rgb, depth, acc = render_rays(model, rays_o, rays_d, 
                                      near=2., far=6., 
                                      n_samples=n_samples)
        loss = tf.reduce_mean(tf.square(rgb - testimg))
        psnr = -10. * tf.math.log(loss) / tf.math.log(10.)

        rgbs.append(rgb)
        psnrs.append(psnr)

    # plt.figure(figsize=(10,4))
    fig, axs = plt.subplots(ncols=5, nrows=2)
    axs[0, 0].set_title('GT')
    axs[0, 0].imshow(testimg)

    for i in range(9):
        row, col = (i+1)//5, (i+1)%5
        axs[row, col].set_title(f'{i}-{psnrs[i]:.5f}')
        axs[row, col].imshow(rgbs[i])
    plt.show()

    '''
    def center(x): 
        return (np.abs(x) / (np.abs(x).sum()+1e-8) * np.arange(len(x))).mean()

    outputs = analyze_model(model, pts_mins, pts_maxs, 
                            embed_fn,
                            n_samples=2048, D=8)

    total_tvalues = []
    X = sm.add_constant(outputs[0])
    for j in range(8):
        tvalues = []
        for i in range(outputs[1].shape[-1]):
            results = sm.OLS(outputs[j + 1][:, i].numpy(), X).fit()
            tvalues.append(np.nan_to_num(np.abs(results.tvalues[1:]))) # remove constant
        tvalues = sorted(tvalues, key=center)
        tvalues = np.stack(tvalues)
        total_tvalues.append(tvalues)
    total_tvalues = tf.stack(total_tvalues, 0)
    total_tvalues = np.abs(total_tvalues)
    max_tvalue = tf.math.reduce_max(total_tvalues)
    min_tvalue = tf.math.reduce_min(total_tvalues)

    fig, axs = plt.subplots(ncols=8)
    for j in range(8):
        print([center(t) for t in total_tvalues[j]])
        imshow = axs[j].imshow(tf.cast(total_tvalues[j] > 2.576, 'int32'),
                               vmin=0, vmax=1) # max_tvalue)
        axs[j].set_title(j)
    # plt.colorbar(imshow)
    plt.savefig('temp.png')
    # plt.show()

    plt.figure(figsize=(10,4))
    plt.subplot(121)
    plt.imshow(rgb)
    plt.title(f'Iteration: {i} ({time.time() - t0:.4f} secs)')
    plt.subplot(122)
    plt.plot(iternums, psnrs)
    plt.title(f'PSNR ({max(psnrs):.6f}, {psnrs[-1]:.6f})')
    plt.show()
    '''

