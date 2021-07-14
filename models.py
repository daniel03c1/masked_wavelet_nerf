import tensorflow as tf


class Embedder:
    def __init__(self, n_freqs, include_inputs=True, log_sampling=True):
        self.funcs = []
        if include_inputs:
            self.funcs.append(tf.identity)

        if log_sampling:
            frequencies = 2.**tf.linspace(0., n_freqs-1, n_freqs)
        else:
            frequencies = tf.linspace(2.**0., 2.**(n_freqs-1), n_freqs)

        for freq in frequencies:
            self.funcs.append(lambda x: tf.math.sin(x * freq))
            self.funcs.append(lambda x: tf.math.cos(x * freq))

    def __call__(self, inputs):
        return tf.concat([fn(inputs) for fn in self.funcs], -1)


def generate_nerf_model(input_chan, views_chan, output_chan=4,
                        depth=8, units=256, skips=None, use_views=False,
                        activation='relu'):
    if skips is None:
        skips = [4]

    inputs = tf.keras.Input(shape=(input_chan + views_chan))
    points, views = tf.split(inputs, [input_chan, views_chan], -1)

    outputs = points
    for i in range(depth):
        outputs = tf.keras.layers.Dense(units, activation=activation)(outputs)
        if i in skips:
            outputs = tf.concat([points, outputs], -1)

    if use_views:
        opacity = tf.keras.layers.Dense(1)(outputs)
        color = tf.keras.layers.Dense(256)(outputs)
        color = tf.concat([views, color], -1)
        color = tf.keras.layers.Dense(units//2, activation=activation)(color)
        color = tf.keras.layers.Dense(output_chan-1)(color)
        outputs = tf.concat([color, opacity], -1)
    else:
        outputs = tf.keras.layers.Dense(output_chan)(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

