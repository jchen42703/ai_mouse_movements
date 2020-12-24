import tensorflow as tf


class MinibatchDiscrimination(tf.keras.layers.Layer):
    """Concatenates to each sample information about how different the input
    features for that sample are from features of other samples in the same
    minibatch, as described in Salimans et. al. (2016). Useful for preventing
    GANs from collapsing to a single output. When using this layer, generated
    samples and reference samples should be in separate batches.


    Example:
        ```python
        # apply a convolution 1d of length 3 to a sequence with 10 timesteps,
        # with 64 output filters
        model = Sequential()
        model.add(Convolution1D(64, 3, border_mode='same',
                                input_shape=(10, 32)))
        # now model.output_shape == (None, 10, 64)
        # flatten the output so it can be fed into a minibatch discrimination
        # layer
        model.add(Flatten())
        # now model.output_shape == (None, 640)
        # add the minibatch discrimination layer
        model.add(MinibatchDiscrimination(5, 3))
        # now model.output_shape = (None, 645)
        ```

    Args:
        units: Number of similarity scores for each sample.
        row_size (int): The dimensionality of the space where closeness of
            samples is calculated.

    Input shape:
        (batch_size, input_dim)

    Output Shape:
        (batch_size, input_dim + units)

    Mainly adopted from:
        https://github.com/forcecore/Keras-GAN-Animeface-Character/
    """

    def __init__(self, units, row_size=3, **kwargs):
        super(MinibatchDiscrimination, self).__init__(**kwargs)
        self.units = units
        self.row_size = row_size

    def build(self, input_shape):
        """Assumes a 2D input shape; (None, input_dim).
        """
        # Paper is A B C, but here, doing B A C for shapes to work
        # Torch: (input_shape[1], self.units*self.row_size)
        T_shape = (self.units, input_shape[1], self.row_size)

        self.T = self.add_weight(
            shape=T_shape,
            initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        M = tf.reshape(tf.matmul(inputs, self.T),
                       (-1, self.units, self.row_size))
        diffs = tf.expand_dims(M, 3) - \
            tf.expand_dims(tf.transpose(M, [1, 2, 0]), 0)
        abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
        minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
        return tf.concat([inputs, minibatch_features], 1)

    def get_config(self):
        config = super(MinibatchDiscrimination, self).get_config()
        config.update({
            'units': self.units,
            'row_size': self.row_size
        })
        return config


if __name__ == '__main__':
    import tensorflow as tf
    x = tf.random.normal((6, 15))
    md = MinibatchDiscrimination(5, 3)
    out = md(x)
    print(out.shape)
    assert list(out.shape) == [6, 20]
