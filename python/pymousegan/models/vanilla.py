import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Bidirectional, Dense, \
    Input, LSTM, Reshape

from .abstract import Generator, Discriminator


class VanillaGenerator(Generator):
    """Basic Dense Only Generator.

    Credits to:
        https://github.com/corynguyen19/midi-lstm-gan/blob/master/mlp_gan.py
        for the generator structure.
    """

    def __init__(self, rand_noise_size=(100,), seq_shape=(100, 3),
                 output_act='tanh'):
        build_kwargs = {
            'output_act': output_act
        }
        super().__init__(rand_noise_size=rand_noise_size, seq_shape=seq_shape,
                         build_kwargs=build_kwargs)

    def build_model(self, output_act='tanh'):
        """Main method for creating the generator model.

        Returns:
            tf.keras.Model or tf.Sequential()
        """
        x = Input(shape=self.noise_size)
        dense_1_input = Dense(256, input_dim=self.noise_size)(x)
        bn_1 = BatchNormalization(momentum=0.8)(dense_1_input)
        dense_2 = Dense(512, activation='relu')(bn_1)
        bn_2 = BatchNormalization(momentum=0.8)(dense_2)
        dense_3 = Dense(1024, activation='relu')(bn_2)
        bn_3 = BatchNormalization(momentum=0.8)(dense_3)
        dense_last = Dense(np.prod(self.seq_shape),
                           activation=output_act)(bn_3)
        output_layer = Reshape((self.seq_shape))(dense_last)
        return tf.keras.Model(inputs=x, outputs=output_layer,
                              name='vanilla_generator')


class DenseBNGenerator(Generator):
    """Flexible Dense + BN Only Generator.
    """

    def __init__(self, rand_noise_size=(100,), seq_shape=(100, 3),
                 hidden_units_list=[512, 300], output_act='tanh'):
        build_kwargs = {
            'hidden_units_list': hidden_units_list,
            'output_act': output_act
        }
        super().__init__(rand_noise_size=rand_noise_size, seq_shape=seq_shape,
                         build_kwargs=build_kwargs)

    def build_model(self, hidden_units_list=[256, 512, 1024],
                    output_act='tanh'):
        """Main method for creating the generator model.

        Returns:
            tf.keras.Model or tf.Sequential()
        """
        num_dense = len(hidden_units_list)
        print(f'Creating a generator with {num_dense} dense layers.')

        x = Input(shape=self.noise_size)

        dense_layers = []  # Added just to store them if needed for future dev
        # Creating the LSTM layers
        for i, units in enumerate(hidden_units_list):
            # First layer (must take input from `x`)
            if i == 0:
                layer = Dense(units, activation='relu', name=f'dense_{i+1}')(x)
                bn = BatchNormalization(momentum=0.8)(layer)
            # Other layers
            else:
                layer = Dense(units, activation='relu',
                              name=f'dense_{i+1}')(bn)
                bn = BatchNormalization(momentum=0.8)(layer)
        dense_layers.append(layer)
        dense_layers.append(bn)

        dense_last = Dense(np.prod(self.seq_shape),
                           activation=output_act)(bn)
        output_layer = Reshape((self.seq_shape))(dense_last)
        return tf.keras.Model(inputs=x, outputs=output_layer,
                              name='dense_bn_generator')


class VanillaDiscriminator(Discriminator):
    """Basic discriminator class for the GAN.

    Credits to:
        https://github.com/corynguyen19/midi-lstm-gan/blob/master/mlp_gan.py
        for the discriminator structure.
    """

    def __init__(self, seq_shape=(100, 3), input_layer=None):
        build_kwargs = {'input_layer': input_layer}
        super().__init__(seq_shape=seq_shape, build_kwargs=build_kwargs)

    def build_model(self, input_layer=None):
        """Main method for creating the discriminator model.

        Returns:
            tf.keras.Model or tf.Sequential()
        """
        if input_layer is not None:
            x = input_layer
        else:
            x = Input(shape=self.seq_shape)

        lstm_1 = LSTM(512, return_sequences=True)(x)
        bilstm = Bidirectional(LSTM(512, return_sequences=False))(lstm_1)
        dense_1 = Dense(512, activation='relu')(bilstm)
        dense_2 = Dense(256, activation='relu')(dense_1)
        output_layer = Dense(1, activation='sigmoid')(dense_2)
        return tf.keras.Model(inputs=x, outputs=output_layer,
                              name='vanilla_discriminator')
