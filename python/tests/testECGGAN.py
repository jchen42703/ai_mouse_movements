import unittest
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import os
import shutil
import types
from pymousegan.models.gans import ECGGAN
from pymousegan.io.data_aug import scale_translate_v3
from testRGAN import check_optimizer_state_equal


class ECGGANTester(unittest.TestCase):
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

    def test_ECGGAN_summary_no_minibatch_discrimination(self):
        """Initializes the GAN and gets all the model summaries.
        """
        d_opt = Adam(lr=5e-4, beta_1=0.5)
        g_opt = Adam(lr=1e-4, beta_1=0.5)
        gan = ECGGAN(d_opt=d_opt, g_opt=g_opt, seq_shape=(100, 3),
                     noise_size=(100, 100),
                     discrim_filters_list=[32, 64, 128, 256],
                     gen_filters_list=[128, 64, 32, 16], mbd_units=None,
                     mbd_rows=None, gen_act='tanh', data_aug=None)
        gan.discriminator.model.summary()
        gan.generator.model.summary()
        gan.combined.summary()

    def test_ECGGAN_summary_minibatch_discrimination(self):
        """Initializes the GAN and gets all the model summaries.
        """
        d_opt = Adam(lr=5e-4, beta_1=0.5)
        g_opt = Adam(lr=1e-4, beta_1=0.5)
        gan = ECGGAN(d_opt=d_opt, g_opt=g_opt, seq_shape=(100, 3),
                     noise_size=(100, 100),
                     discrim_filters_list=[32, 64, 128, 256],
                     gen_filters_list=[128, 64, 32, 16], gen_act='tanh',
                     data_aug=None)
        gan.discriminator.model.summary()
        gan.generator.model.summary()
        gan.combined.summary()

    def test_ECGGAN_data_aug(self):
        """Tests ECGGAN to see if it can train with data augmentation.
        """
        coords_dt = np.load('../data/coords_dt.npy')
        # Runs one iteration of the model
        seed = 42
        np.random.seed(seed)
        tf.random.set_seed(seed)

        d_opt = Adam(lr=5e-4, beta_1=0.5)
        g_opt = Adam(lr=1e-4, beta_1=0.5)
        gan = ECGGAN(d_opt=d_opt, g_opt=g_opt, seq_shape=(100, 3),
                     noise_size=(100, 100),
                     discrim_filters_list=[32, 64, 128, 256],
                     gen_filters_list=[128, 64, 32, 16], gen_act='tanh',
                     data_aug=scale_translate_v3)
        self.assertTrue(isinstance(gan.data_aug, types.FunctionType),
                        'gan.data_aug must be a function if provided.')
        # Saves weights in a temperorary directory
        os.mkdir(self.model_out)
        gan.train(coords_dt, num_epochs=1, batch_size=1, sample_interval=1,
                  output_dir=self.model_out, save_format='h5')

    def test_ECGGAN_model_loading_h5(self):
        """Tests that models are loaded properly for ECGGAN.
        """
        coords_dt = np.load('../data/coords_dt.npy')
        # Runs one iteration of the model
        seed = 42
        np.random.seed(seed)
        tf.random.set_seed(seed)

        d_opt = Adam(lr=5e-4, beta_1=0.5)
        g_opt = Adam(lr=1e-4, beta_1=0.5)
        gan = ECGGAN(d_opt=d_opt, g_opt=g_opt, seq_shape=(100, 3),
                     noise_size=(100, 100),
                     discrim_filters_list=[32, 64, 128, 256],
                     gen_filters_list=[128, 64, 32, 16], gen_act='tanh',
                     data_aug=scale_translate_v3)

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
        gan2 = ECGGAN(d_opt=d_opt, g_opt=g_opt, seq_shape=(100, 3),
                      noise_size=(100, 100),
                      discrim_filters_list=[32, 64, 128, 256],
                      gen_filters_list=[128, 64, 32, 16], gen_act='tanh',
                      data_aug=scale_translate_v3, model_paths=model_paths)

        # Check if the loaded models == old model

        # Check if optimizer state is the same
        # Discriminator
        self.assertTrue(check_optimizer_state_equal(gan.discriminator.model,
                                                    gan2.discriminator.model))

        # Combined (includes generator)
        self.assertTrue(check_optimizer_state_equal(gan.combined,
                                                    gan2.combined))


unittest.main(argv=[''], verbosity=2, exit=False)
