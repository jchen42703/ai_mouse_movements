import abc
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dense, Input, LSTM


class DTModel(metaclass=abc.ABCMeta):
    def __init__(self, seq_shape=(100, 2), dt_shape=(100,), build_kwargs={}):
        self.seq_shape = seq_shape
        self.dt_shape = dt_shape
        self.model = self.build_model(**build_kwargs)

    @abc.abstractmethod
    def build_model(self):
        return
    
    def separate_paths_dt(self, coords_dt):
        """
        Args:
            coords_dt (np.ndarray):
                should have shape (num_paths, 100, 3) or some variant of that
        """
        num_paths = coords_dt.shape[0]
        num_coords = coords_dt.shape[1] # usually 10
        paths = np.zeros((num_paths, num_coords, 2))
        dt = np.zeros((num_paths, num_coords))

        for i, path in enumerate(coords_dt):
            paths[i] = path[:, :-1]
            dt[i] = path[:, -1]
        
        return (paths, dt)


class DenseBNDTModel(DTModel):
    """Flexible Dense + BN Only Model.

    Infers that the dt_shape is (seq_shape[0], 1). Note: only (1,) is kept as
    dt_shape, because the dense layers keep the seq_shape[0] throughout the
    model.
    """

    def __init__(self, seq_shape=(100, 2),
                 hidden_units_list=[512, 300]):
        build_kwargs = {
            'hidden_units_list': hidden_units_list,
        }
        super().__init__(seq_shape=seq_shape, dt_shape=(1,),
                         build_kwargs=build_kwargs)

    def build_model(self, hidden_units_list=[256, 512, 1024]):
        """Main method for creating the model.

        Returns:
            tf.keras.Model or tf.Sequential()
        """
        num_dense = len(hidden_units_list)
        print(f'Creating a model with {num_dense} dense layers.')

        x = Input(shape=self.seq_shape)

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

        dense_last = Dense(self.dt_shape[0], activation='sigmoid')(bn)
        return tf.keras.Model(inputs=x, outputs=dense_last,
                              name='dense_bn_dt_model')


class LSTMDecoderDTModel(DTModel):
    """Simple model with only unidirectional LSTM layers.
    """

    def __init__(self, seq_shape=(100, 2),
                 hidden_units_list=[512, 300]):
        build_kwargs = {
            'hidden_units_list': hidden_units_list,
        }
        super().__init__(seq_shape=seq_shape, dt_shape=(seq_shape[0],),
                         build_kwargs=build_kwargs)

    def build_model(self, hidden_units_list=[512, 300]):
        """Main method for creating the model model.

        Creates a flexible stacked LSTM model with len(hidden_units_list)
        layers and each one corresponding to the hidden units in
        hidden_units_list.

        Returns:
            tf.keras.Model or tf.Sequential()
        """
        num_lstm = len(hidden_units_list)
        print(f'Creating a model with {num_lstm} LSTM layers.')

        x = Input(shape=self.seq_shape)

        lstm_layers = []  # Added just to store them if needed for future dev
        # Creating the LSTM layers
        for i, units in enumerate(hidden_units_list):
            # First layer (must take input from `reshaped`)
            if i == 0:
                return_sequences = True if num_lstm > 1 else False
                layer = LSTM(units, return_sequences=return_sequences,
                             name=f'lstm_{i+1}')(x)
            # Final layer, return_sequences=False
            elif i == (num_lstm - 1):
                layer = LSTM(units, return_sequences=False,
                             name=f'lstm_{i+1}')(layer)
            # intermediate layers
            else:
                layer = LSTM(units, return_sequences=True,
                             name=f'lstm_{i+1}')(layer)
            lstm_layers.append(layer)

        dense_last = Dense(self.dt_shape[0], activation='sigmoid')(layer)
        return tf.keras.Model(inputs=x, outputs=dense_last,
                              name='lstm_dt_model')
