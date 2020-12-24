import unittest
import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import sigmoid

from pymousegan.models.gans import BasicGAN
from pymousegan.io.data_aug import scale_translate_v3
from pymousegan.models.lstm import LSTMGenerator, LSTMDecoderGenerator, \
    BidirectionalLSTMDiscriminator, handle_return_sequences
from testRGAN import check_optimizer_state_equal


class LSTMTester(unittest.TestCase):
    def setUp(self):
        self.model_out = './model_out'

    def tearDown(self):
        if os.path.isdir(self.model_out):
            shutil.rmtree(self.model_out)

    def test_handle_return_sequences(self):
        # first
        # 1 layer -> return true if lstm and false if dense
        self.assertTrue(handle_return_sequences('lstm', 0, 1))
        self.assertFalse(handle_return_sequences('dense', 0, 1))
        # multiple layers -> true regardless of output_layer_type
        self.assertTrue(handle_return_sequences('lstm', 0, 5))
        self.assertTrue(handle_return_sequences('dense', 0, 5))
        # intermediate -> true regardless of output_layer_type
        self.assertTrue(handle_return_sequences('lstm', 5, 14))
        self.assertTrue(handle_return_sequences('dense', 5, 14))
        # last -> true if lstm and false if dense
        self.assertTrue(handle_return_sequences('lstm', 13, 14))
        self.assertFalse(handle_return_sequences('dense', 13, 14))

    def test_LSTMGenerator(self):
        """Tests that LSTMGenerator loads properly.
        """
        generator = LSTMGenerator((100,), seq_shape=(100, 3),
                                  hidden_units_list=[512, 300])
        self.assertTrue(generator.generate_noise(16).shape == (16, 100),
                        'The generated noise should be shaped properly.')
        generator.model.summary()

    def test_LSTMDecoderGenerator(self):
        """Tests that LSTMDecoderGenerator loads properly.
        """
        generator = LSTMDecoderGenerator((100, 1), seq_shape=(100, 3),
                                         hidden_units_list=[512, 300])
        self.assertTrue(generator.generate_noise(16).shape == (16, 100, 1),
                        'The generated noise should be shaped properly.')
        generator.model.summary()

    def test_LSTMGenerator_sigmoid_output_act(self):
        """Tests that LSTMGenerator loads properly.
        """
        generator = LSTMGenerator((100,), seq_shape=(100, 3),
                                  hidden_units_list=[512, 300],
                                  output_act='sigmoid')
        self.assertTrue(generator.generate_noise(16).shape == (16, 100),
                        'The generated noise should be shaped properly.')
        self.assertTrue(generator.model.layers[-1].activation == sigmoid)
        generator.model.summary()

    def test_LSTMDecoderGenerator_sigmoid_output_act(self):
        """Tests that LSTMDecoderGenerator loads properly.
        """
        generator = LSTMDecoderGenerator((100, 1), seq_shape=(100, 3),
                                         hidden_units_list=[512, 300],
                                         output_act='sigmoid')
        self.assertTrue(generator.generate_noise(16).shape == (16, 100, 1),
                        'The generated noise should be shaped properly.')
        self.assertTrue(generator.model.layers[-1].activation == sigmoid)
        generator.model.summary()

    def test_BidirectionalLSTMDiscriminator_no_minibatch_discrim(self):
        # Default should have no minibatch discrimination
        discrim = BidirectionalLSTMDiscriminator(seq_shape=(100, 3),
                                                 hidden_units_list=[512, 300])
        self.assertTrue(discrim.model.layers[-2].output_shape == (None, 600))

        discrim.model.summary()

    def test_BidirectionalLSTMDiscriminator_with_minibatch_discrim(self):
        # Default should have no minibatch discrimination
        discrim = BidirectionalLSTMDiscriminator((100, 3), [512, 300], 5, 3)
        self.assertTrue(discrim.model.layers[-2].output_shape == (None, 605))

        discrim.model.summary()

    def test_train_BidirectionalLSTMDiscriminator_with_minibatch_discrim(self):
        """Tests that the model can be trained properly and loaded properly.
        """
        coords_dt = np.load('../data/coords_dt.npy')
        seed = 420
        np.random.seed(seed)
        tf.random.set_seed(seed)

        seq_shape = (100, 3)
        noise_size = (100, 100)
        d_opt = Adam(lr=1e-4, beta_1=0.5)
        g_opt = Adam(lr=1e-5, beta_1=0.5)

        discrim = BidirectionalLSTMDiscriminator(seq_shape, [256, 128], 5, 3)
        generator = LSTMDecoderGenerator(noise_size, seq_shape, [256, 256])

        gan = BasicGAN(discrim, generator, d_opt, g_opt,
                       data_aug=scale_translate_v3)
        os.mkdir(self.model_out)

        gan.train(coords_dt, num_epochs=1, batch_size=1, sample_interval=1,
                  output_dir=self.model_out, save_format='h5')

        gan.discriminator.model.summary(), gan.generator.model.summary()

        # Testing if the loaded model is the same as the previous model
        # Load model in a new instance of BasicGAN
        model_paths = {
            'discrim_path': os.path.join(self.model_out,
                                         'discrim_1_weights.h5'),
            'gen_path': os.path.join(self.model_out,
                                     'gen_1_weights.h5'),
            'combined_path': os.path.join(self.model_out,
                                          'combined_1_weights.h5')
        }
        gan2 = BasicGAN(discrim, generator, d_opt, g_opt,
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
