import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Average, BatchNormalization, \
    Bidirectional, Dense, Input, LSTM, \
    Reshape

from .abstract import Generator, Discriminator, MinibatchDiscriminator
from .minibatch_discrimination import MinibatchDiscrimination


def handle_return_sequences(output_layer_type, i, num_lstm):
    """Returns the value of `return_sequences` for LSTM layers based
    on the index and provided fields.

    Args:
        output_layer_type (str): specifies how the overall model
            outputs the sequence; one of 'dense', 'lstm', None.
            - When it is 'dense', we assume that the final layer expects a
            flattened sequence, such as for the discriminator.
    """
    # First layer (must take input from `reshaped`)
    if i == 0:
        # False when 1 element & Dense output
        return_sequences = output_layer_type == 'lstm' or num_lstm > 1
    # Final layer, return_sequences=True when the output layer is lstm
    elif i == (num_lstm - 1):
        # lstm final layer type -> needs to have the full sequence
        return_sequences = output_layer_type == 'lstm'
    # intermediate layers
    else:
        return_sequences = True

    return return_sequences


def recursive_create_LSTM(input_layer, hidden_units_list, output_layer_type):
    """Recursive creates intermediate LSTM layers based on
    hidden_units_list.
    """
    # Added just to store them if needed forfuture dev
    lstm_layers = []
    # Creating the LSTM layers
    num_lstm = len(hidden_units_list)
    for i, units in enumerate(hidden_units_list):
        return_sequences = handle_return_sequences(output_layer_type,
                                                   i, num_lstm)
        if i == 0:
            layer = LSTM(units, return_sequences=return_sequences,
                         name=f'lstm_{i+1}')(input_layer)
        else:
            layer = LSTM(units, return_sequences=return_sequences,
                         name=f'lstm_{i+1}')(layer)
        lstm_layers.append(layer)
    return (lstm_layers, layer)


def recursive_create_BiLSTM(input_layer, hidden_units_list, output_layer_type):
    """Recursive creates intermediate bidrectional LSTM layers based on
    hidden_units_list.
    """
    # Added just to store them if needed forfuture dev
    bilstm_layers = []
    # Creating the LSTM layers
    num_lstm = len(hidden_units_list)
    for i, units in enumerate(hidden_units_list):
        return_sequences = handle_return_sequences(output_layer_type,
                                                   i, num_lstm)
        if i == 0:
            layer = Bidirectional(LSTM(units,
                                       return_sequences=return_sequences,
                                       name=f'bilstm_{i+1}'))(input_layer)
        else:
            layer = Bidirectional(LSTM(units,
                                       return_sequences=return_sequences,
                                       name=f'bilstm_{i+1}'))(layer)
        bilstm_layers.append(layer)
    return (bilstm_layers, layer)


class LSTMGenerator(Generator):
    """Simple generator with a dense layer -> unidirectional LSTM layers.
    """

    def __init__(self, rand_noise_size=(100,), seq_shape=(100, 3),
                 hidden_units_list=[512, 300], output_act='tanh',
                 output_layer_type='lstm'):
        build_kwargs = {
            'hidden_units_list': hidden_units_list,
            'output_act': output_act
        }
        assert output_layer_type.lower() in ['lstm', 'dense'], \
            "output_layer_type must be one of 'lstm', 'dense'"
        self.output_layer_type = output_layer_type.lower()

        super().__init__(rand_noise_size=rand_noise_size, seq_shape=seq_shape,
                         build_kwargs=build_kwargs)

    def build_model(self, hidden_units_list=[512, 300], output_act='tanh'):
        """Main method for creating the generator model.

        Creates a flexible stacked LSTM generator with len(hidden_units_list)
        layers and each one corresponding to the hidden units in
        hidden_units_list.

        Returns:
            tf.keras.Model or tf.Sequential()
        """
        num_lstm = len(hidden_units_list)
        print(f'Creating a generator with {num_lstm} LSTM layers.')

        x = Input(shape=self.noise_size)
        dense_1_input = Dense(np.prod(self.seq_shape),
                              input_dim=self.noise_size, activation='relu')(x)
        reshaped = Reshape((self.seq_shape))(dense_1_input)
        lstm_layers, layer = recursive_create_LSTM(reshaped,
                                                   hidden_units_list,
                                                   self.output_layer_type)
        output_layer = self.get_output_layer(layer, output_act)

        return tf.keras.Model(inputs=x, outputs=output_layer,
                              name='lstm_generator')

    def get_output_layer(self, input_layer, output_act='tanh'):
        # inferring the output layer
        if self.output_layer_type == 'dense':
            dense_last = Dense(np.prod(self.seq_shape),
                               activation=output_act)(input_layer)
            output_layer = Reshape((self.seq_shape))(dense_last)
        else:
            # LSTM
            output_layer = LSTM(self.seq_shape[-1], return_sequences=True,
                                activation=output_act,
                                name='lstm_out')(input_layer)
        return output_layer


class LSTMDecoderGenerator(LSTMGenerator):
    """Simple generator with only unidirectional LSTM layers.
    """

    def __init__(self, rand_noise_size=(100, 1), seq_shape=(100, 3),
                 hidden_units_list=[512, 300], output_act='tanh',
                 output_layer_type='lstm'):
        super().__init__(rand_noise_size=rand_noise_size, seq_shape=seq_shape,
                         hidden_units_list=hidden_units_list,
                         output_act=output_act,
                         output_layer_type=output_layer_type)

    def build_model(self, hidden_units_list=[512, 300], output_act='tanh'):
        """Main method for creating the generator model.

        Creates a flexible stacked LSTM generator with len(hidden_units_list)
        layers and each one corresponding to the hidden units in
        hidden_units_list.

        Returns:
            tf.keras.Model or tf.Sequential()
        """
        num_lstm = len(hidden_units_list)
        print(f'Creating a generator with {num_lstm} LSTM layers.')

        x = Input(shape=self.noise_size)

        lstm_layers, layer = recursive_create_LSTM(x, hidden_units_list,
                                                   self.output_layer_type)

        output_layer = self.get_output_layer(layer, output_act)

        return tf.keras.Model(inputs=x, outputs=output_layer,
                              name='lstm_generator')


class BidirectionalLSTMDecoderGenerator(LSTMGenerator):
    """Simple generator with bidirectional LSTM layers.
    """

    def __init__(self, rand_noise_size=(100, 1), seq_shape=(100, 3),
                 hidden_units_list=[512, 300], output_act='tanh',
                 output_layer_type='lstm'):
        super().__init__(rand_noise_size=rand_noise_size, seq_shape=seq_shape,
                         hidden_units_list=hidden_units_list,
                         output_act=output_act,
                         output_layer_type=output_layer_type)

    def build_model(self, hidden_units_list=[512, 300], output_act='tanh'):
        """Main method for creating the generator model.

        Creates a flexible stacked LSTM generator with len(hidden_units_list)
        layers and each one corresponding to the hidden units in
        hidden_units_list.

        Returns:
            tf.keras.Model or tf.Sequential()
        """
        num_lstm = len(hidden_units_list)
        print(f'Creating a generator with {num_lstm} LSTM layers.')

        x = Input(shape=self.noise_size)

        lstm_layers, layer = recursive_create_BiLSTM(x, hidden_units_list,
                                                     self.output_layer_type)

        output_layer = self.get_output_layer(layer, output_act)

        return tf.keras.Model(inputs=x, outputs=output_layer,
                              name='bilstm_generator')


class BidirectionalLSTMDiscriminator(MinibatchDiscriminator):
    """Discriminator with bidirectional LSTM layers.
    """

    def __init__(self, seq_shape=(100, 3), hidden_units_list=[512, 300],
                 minibatch_discrim_units=None,
                 minibatch_discrim_row_size=None):
        build_kwargs = {
            'hidden_units_list': hidden_units_list,
        }
        super().__init__(seq_shape=seq_shape,
                         minibatch_discrim_units=minibatch_discrim_units,
                         minibatch_discrim_row_size=minibatch_discrim_row_size,
                         build_kwargs=build_kwargs)

    def build_model(self, hidden_units_list=[512, 300],
                    minibatch_discrim=None):
        """Main method for creating the discriminator model.

        Creates a flexible stacked LSTM discriminator with
        len(hidden_units_list) layers and each one corresponding to the
        hidden units in hidden_units_list.

        Args:
            hidden_units_list: Corresponds to the hidden units fed to each
                Bidrectional LSTM layer combo. Note that the resulting
                hidden units will actually be twice the provided value b/c
                of the Bidrectional layer.
            minibatch_discrim: If you want minibatch discrimination, this
                should be a dictionary representing the kwargs for
                MinibatchDiscrimination.
                - Should have the keys: 'units' or 'row_size'

                If you do not want it, make it anything other than a
                dictionary.
        Returns:
            tf.keras.Model or tf.Sequential()
        """
        num_lstm = len(hidden_units_list)
        print(f'Creating a discrminator with {num_lstm} LSTM layers.')

        x = Input(shape=self.seq_shape)

        lstm_layers, layer = recursive_create_BiLSTM(x, hidden_units_list,
                                                     'dense')

        # Minibatch Discrimination - to handle single point mode collapse
        if isinstance(minibatch_discrim, dict):
            layer = MinibatchDiscrimination(**minibatch_discrim)(layer)

        output_layer = Dense(1, activation='sigmoid')(layer)

        return tf.keras.Model(inputs=x, outputs=output_layer,
                              name='bidirectional_lstm_discriminator')


class LSTMDiscriminator(Discriminator):
    """Basic LSTM discriminator class for the GAN.
    """

    def __init__(self, seq_shape=(100, 3), input_layer=None):
        build_kwargs = {'input_layer': input_layer}
        super().__init__(seq_shape=seq_shape, build_kwargs=build_kwargs)

    def build_model(self, input_layer=None):
        """Main method for creating the discriminator model.

        Returns:
            tf.keras.Model or tf.Sequential()
        """
        # 8,010,241 params default
        if input_layer is not None:
            x = input_layer
        else:
            x = Input(shape=self.seq_shape)

        lstm_1 = LSTM(512, return_sequences=True)(x)
        bilstm_1 = Bidirectional(LSTM(512, return_sequences=True))(lstm_1)
        bilstm_2 = Bidirectional(LSTM(256, return_sequences=False))(bilstm_1)
        dense_1 = Dense(256, activation='relu')(bilstm_2)
        output_layer = Dense(1, activation='sigmoid')(dense_1)
        return tf.keras.Model(inputs=x, outputs=output_layer,
                              name='lstm_discriminator')


class DenseBidirectionalLSTMDiscriminator(Discriminator):
    """Discriminator with bidirectional LSTM and Dense layers.

    The theory is that the dense layers will offset some of the vanishing
    gradient issues that the LSTM layers bring.
    """

    def __init__(self, seq_shape=(100, 3), bn=False, lstm_hu_list=[512, 300],
                 dense_hu_list=[256]):
        build_kwargs = {
            'lstm_hu_list': lstm_hu_list,
            'dense_hu_list': dense_hu_list,
            'bn': bn
        }
        super().__init__(seq_shape=seq_shape,
                         build_kwargs=build_kwargs)

    def build_model(self, lstm_hu_list=[512, 300], dense_hu_list=[256],
                    bn=False):
        """Main method for creating the discriminator model.

        Creates a flexible stacked discriminator with len(lstm_hu_list) +
        len(dense_hu_list) layers and each one corresponding to the hidden
        units in lstm_hu_list and dense_hu_list.

        Returns:
            tf.keras.Model or tf.Sequential()
        """
        num_lstm = len(lstm_hu_list)
        print(f'Creating a discrminator with {num_lstm} LSTM layers.')
        num_dense = len(dense_hu_list)
        print(f'Creating a discrminator with {num_dense} dense layers.')

        x = Input(shape=self.seq_shape)

        layers, layer = recursive_create_BiLSTM(x, lstm_hu_list,
                                                'dense')

        for i, units in enumerate(dense_hu_list):
            # First layer (must take input from `x`)
            if i == 0:
                layer = Dense(units, activation='relu',
                              name=f'discrim_dense_{i+1}')(layers[-1])
                if bn:
                    layer = BatchNormalization(momentum=0.8)(layer)
            # Other layers
            else:
                layer = Dense(units, activation='relu',
                              name=f'discrim_dense_{i+1}')(layer)
                if bn:
                    layer = BatchNormalization(momentum=0.8)(layer)
            layers.append(layer)

        output_layer = Dense(1, activation='sigmoid')(layer)

        return tf.keras.Model(inputs=x, outputs=output_layer,
                              name='dense_bidirectional_lstm_discriminator')


class EnsembleLSTMDiscriminator(Discriminator):
    """Simple LSTM Ensemble discriminator class for the GAN.

    Similar to the discriminator in C-RNN-GAN.
    """

    def __init__(self, seq_shape=(100, 3), input_layer=None, num_bidirect=3,
                 bidirect_units=256):
        build_kwargs = {
            'input_layer': input_layer,
            'num_bidirect': num_bidirect,
            'bidirect_units': bidirect_units,
        }
        super().__init__(seq_shape=seq_shape, build_kwargs=build_kwargs)

    def build_model(self, input_layer=None, num_bidirect=3,
                    bidirect_units=256):
        """Main method for creating the discriminator model.

        Args:
            num_bidirect (int): Number of bidrectional LSTM layers
            bidirect_units (int): Number of units in each bidrectional LSTM
                layer

        Returns:
            tf.keras.Model or tf.Sequential()
        """
        if input_layer is not None:
            x = input_layer
        else:
            x = Input(shape=self.seq_shape)

        lstm_1 = LSTM(512, return_sequences=True)(x)
        # Feeds into separate bidrectionals
        bilstm_layers = []
        for i in range(num_bidirect):
            bilstm_layer = Bidirectional(LSTM(bidirect_units,
                                              return_sequences=False),
                                         name=f'bilstm_{i+1}')(lstm_1)
            bilstm_layers.append(bilstm_layer)

        # dense --> sigmoid for each bidrectional
        output_layers = []
        for i, layer in enumerate(bilstm_layers):
            output_layers.append(Dense(1, activation='sigmoid',
                                       name=f'dense_{i+1}')(layer))
        # Then averaging (element-wise)
        ensembled_output_layer = Average()(output_layers)

        # avg = Average()()
        return tf.keras.Model(inputs=x, outputs=ensembled_output_layer,
                              name='ensemble_lstm_discriminator')
