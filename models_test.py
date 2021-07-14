import tensorflow as tf
from models import *


class ModelsTest(tf.test.TestCase):
    def setUp(self):
        self.points = tf.random.normal([128, 3])

    def test_Embedder(self):
        n_freqs = 5
        embedder = Embedder(n_freqs=n_freqs, include_inputs=False)
        self.assertEqual(embedder(self.points).shape[-1], 6*n_freqs)

        n_freqs = 3
        embedder = Embedder(n_freqs=n_freqs, include_inputs=True)
        self.assertEqual(embedder(self.points).shape[-1], 3*(1+2*n_freqs))

    def test_generate_nerf_model(self):
        input_chan = 33 # 3 + 5 * 3 * 2
        views_chan = 15 # 3 + 2 * 3 * 2

        model = generate_nerf_model(input_chan, views_chan)
        self.assertEqual(model.output_shape[-1], 4)


if __name__ == '__main__':
    tf.test.main()

