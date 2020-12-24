import unittest
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import os
import shutil
import types
from pymousegan.models.gans import RGAN
from pymousegan.io.data_aug import scale_translate_v3


class RGANTester(unittest.TestCase):
    def setUp(self):
        self.model_out = './model_out'

    def tearDown(self):
        if os.path.isdir(self.model_out):
            shutil.rmtree(self.model_out)

    def test_equality(self):
        x = tf.constant(2)
        y = tf.constant(2)
        assert tf.math.equal(x, y).numpy()
        self.assertTrue(tf.math.equal(x, y).numpy())

    def test_RGAN_data_aug(self):
        """Tests RGAN to see if it can train with data augmentation.
        """
        coords_dt = np.load('../data/coords_dt.npy')
        # Runs one iteration of the model
        seed = 42
        np.random.seed(seed)
        tf.random.set_seed(seed)

        d_opt = Adam(lr=5e-4, beta_1=0.5)
        g_opt = Adam(lr=1e-4, beta_1=0.5)
        gan = RGAN(d_opt=d_opt, g_opt=g_opt, noise_size=(100, 100),
                   lstm_hu_d=750, lstm_hu_g=500, seq_shape=(100, 3),
                   data_aug=scale_translate_v3, num_D_rounds=1, num_G_rounds=1,
                   gen_act='tanh')
        self.assertTrue(isinstance(gan.data_aug, types.FunctionType),
                        'gan.data_aug must be a function if provided.')
        # Saves weights in a temperorary directory
        os.mkdir(self.model_out)
        gan.train(coords_dt, num_epochs=1, batch_size=1, sample_interval=1,
                  output_dir=self.model_out, save_format='h5')

    def test_RGAN_model_loading_h5(self):
        """Tests that models are loaded properly for RGAN.
        """
        coords_dt = np.load('../data/coords_dt.npy')
        # Runs one iteration of the model
        seed = 42
        np.random.seed(seed)
        tf.random.set_seed(seed)

        d_opt = Adam(lr=5e-4, beta_1=0.5)
        g_opt = Adam(lr=1e-4, beta_1=0.5)
        gan = RGAN(d_opt=d_opt, g_opt=g_opt, noise_size=(100, 100),
                   lstm_hu_d=750, lstm_hu_g=500, seq_shape=(100, 3),
                   num_D_rounds=1, num_G_rounds=1, gen_act='tanh')

        # Saves weights in a temperorary directory
        os.mkdir(self.model_out)
        gan.train(coords_dt, num_epochs=1, batch_size=1, sample_interval=1,
                  output_dir=self.model_out, save_format='h5')
        # Testing if the loaded model is the same as the previous model
        # Load model in a new instance of VanillaGAN
        model_paths = {
            'discrim_path': os.path.join(self.model_out,
                                         'discrim_1_weights.h5'),
            'gen_path': os.path.join(self.model_out,
                                     'gen_1_weights.h5'),
            'combined_path': os.path.join(self.model_out,
                                          'combined_1_weights.h5')
        }
        gan2 = RGAN(d_opt=d_opt, g_opt=g_opt, noise_size=(100, 100),
                    lstm_hu_d=750, lstm_hu_g=500, seq_shape=(100, 3),
                    num_D_rounds=1, num_G_rounds=1, gen_act='tanh',
                    model_paths=model_paths)

        # Check if the loaded models == old model

        # Check if optimizer state is the same
        # Discriminator
        self.assertTrue(check_optimizer_state_equal(gan.discriminator.model,
                                                    gan2.discriminator.model))

        # Combined (includes generator)
        self.assertTrue(check_optimizer_state_equal(gan.combined,
                                                    gan2.combined))


def check_optimizer_state_equal(model1, model2):
    """Assumes that they have the same optimizer state.

    Assumes that the optimizer states have the same variables in the same
    order. (i.e. from loading the optimizer state directly)
    """
    # Checks learning rate
    if not tf.math.equal(model1.optimizer.lr, model2.optimizer.lr).numpy():
        print('Optimizers do not have the same lr.')
        print(f'{model1.optimizer.lr} != {model2.optimizer.lr}')

        return False

    # checks the length of the optimizer states
    if not len(model1.optimizer.weights) == len(model2.optimizer.weights):
        print('Optimizers do not have the len of weights.')
        print(f'{len(model1.optimizer.weights)} != {len(model2.optimizer.weights)}')
        return False

    # checks the values of the individual optimizer state weights
    for w1, w2 in zip(model1.optimizer.weights, model2.optimizer.weights):
        if not tf.math.equal(w1, w2).numpy().all():
            print('Weights not the same.')
            print(f'{w1} != {w2}')
            return False

    return True


unittest.main(argv=[''], verbosity=2, exit=False)
