import tensorflow as tf
from utils import *


class UtilsTest(tf.test.TestCase):
    def setUp(self):
        self.height = 300
        self.width = 400
        self.focal = 100
        self.c2w = tf.eye(4)
        
    def test_get_grid(self):
        # downscale == 1
        height, width, focal = 4, 6, 1
        grid = get_grid(height, width, focal)
        self.assertEqual(grid.shape, [height, width, 3])
        print(grid)
        self.assertEqual(tf.reduce_sum(grid, axis=(0, 1)),
                         tf.zeros((3,)))

    def test_get_rays(self):
        # test [batch, 3] and [batch, n_samples, 3] shaped dirs
        for shape in [[32], [4, 2]]:
            dirs = tf.random.normal([*shape, 3])
            c2w = tf.random.normal([*shape, 3, 4])
            rays_o, rays_d = get_rays(dirs, c2w)
            self.assertEqual(rays_o.shape, [*shape, 3])
            self.assertEqual(rays_d.shape, [*shape, 3])

    def test_rays_to_points(self):
        rays_o = tf.random.normal([self.height, self.width, 3])
        rays_d = tf.random.normal([self.height, self.width, 3])
        near = tf.zeros([self.height, self.width])
        far = tf.ones([self.height, self.width])
        n_samples = 32

        points, z_vals = rays_to_points(rays_o, rays_d, near, far, n_samples)
        self.assertEqual(points.shape, [self.height, self.width, n_samples, 3])
        self.assertEqual(z_vals.shape, [self.height, self.width, n_samples])

    def test_render_raw_outputs(self):
        height, width = 3, 4
        n_samples = 16
        n_feats = 3

        raw = tf.random.normal([height, width, n_samples, n_feats+1])
        z_vals = tf.broadcast_to(tf.linspace(0., 1., n_samples),
                                 [height, width, n_samples])
        rays_d = tf.random.normal([height, width, 3])

        # without background
        feats_map, depth_map = render_raw_outputs(raw, z_vals, rays_d)
        self.assertEqual(feats_map.shape, [height, width, n_feats])
        self.assertEqual(depth_map.shape, [height, width])

        # with background
        background = tf.ones([height, width, n_feats])
        feats_map, depth_map = render_raw_outputs(raw, z_vals, rays_d,
                                                  background=background)
        self.assertEqual(feats_map.shape, [height, width, n_feats])
        self.assertEqual(depth_map.shape, [height, width])


if __name__ == '__main__':
    tf.test.main()

