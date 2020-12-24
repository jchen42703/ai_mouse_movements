import unittest
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from pymousegan.models.pred_dt import DenseBNDTModel, LSTMDecoderDTModel


class DTModelTester(unittest.TestCase):

    def test_DenseBNDTModel(self):
        """Tests that DenseBNDTModel can be trained with default settings.
        """
        coords_dt = np.load('../data/coords_dt.npy')
        # Runs one iteration of the model
        seed = 42
        np.random.seed(seed)
        tf.random.set_seed(seed)

        opt = Adam(lr=5e-4, beta_1=0.5)
        dt_model = DenseBNDTModel(seq_shape=(100, 2),
                                  hidden_units_list=[512, 300])
        dt_model.model.compile(opt, loss='mse')
        paths, dt = dt_model.separate_paths_dt(coords_dt)
        dt_model.model.summary()
        dt_model.model.fit(x=paths[:2], y=dt[:2], batch_size=1, epochs=1)

    def test_LSTMDecoderDTModel(self):
        """Tests that LSTMDecoderDTModel can be trained with default settings.
        """
        coords_dt = np.load('../data/coords_dt.npy')
        # Runs one iteration of the model
        seed = 42
        np.random.seed(seed)
        tf.random.set_seed(seed)

        opt = Adam(lr=5e-4, beta_1=0.5)
        dt_model = LSTMDecoderDTModel(seq_shape=(100, 2),
                                      hidden_units_list=[512, 300])
        dt_model.model.compile(opt, loss='mse')
        paths, dt = dt_model.separate_paths_dt(coords_dt)
        dt_model.model.summary()
        dt_model.model.fit(x=paths[:2], y=dt[:2], batch_size=1, epochs=1)


unittest.main(argv=[''], verbosity=2, exit=False)
