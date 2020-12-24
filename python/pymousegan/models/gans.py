import types
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, Dense, Flatten, Input, \
    LSTM, Reshape, Conv1D, LeakyReLU

from .vanilla import VanillaDiscriminator, VanillaGenerator
from .abstract import GAN
from .utils import load_model_no_model_config_from_hdf5
from pymousegan.metrics import soft_binary_accuracy
from .minibatch_discrimination import MinibatchDiscrimination


class BasicGAN(GAN):
    """Basic complete GAN class with label smoothing and data augmentation
    during training.
    """

    def __init__(self, discriminator, generator, d_opt, g_opt, data_aug=None,
                 model_paths={}, compile_kwargs={}):
        """
        Args:
            model_paths (dict): must contain keys that match the arguments for
                load_models.
                For instance, the keys in this class are:
                    {
                        'discrim_path':...
                        'gen_path':...
                        'combined_path':...
                    }
                which are string paths to the saved model.
        """
        super().__init__(discriminator=discriminator,
                         generator=generator,
                         d_opt=d_opt, g_opt=g_opt,
                         model_paths=model_paths,
                         compile_kwargs=compile_kwargs)
        assert isinstance(data_aug, types.FunctionType) | (not data_aug), \
            'data_aug must be a function or None.'
        self.data_aug = data_aug

    def connect_discrim_gen(self):
        """Connects the graphs of the discriminator and generator.
        """
        z = tf.keras.layers.Input(shape=self.generator.model.input.shape[1:])
        fake = self.generator.model(z)

        self.discriminator.model.trainable = False

        validity = self.discriminator.model(fake)

        self.combined = tf.keras.Model(z, validity, name='stacked_model')

    def load_models(self, discrim_path=None, gen_path=None,
                    combined_path=None):
        """Loads the model paths (assumes this is called before `compile_models`)
        """
        # Not strict checking
        # (allows self.load_models as long as the two are not None or False)
        if discrim_path and gen_path and combined_path:
            from tensorflow.keras.models import load_model
            print(f'Loading {discrim_path}...')
            custom_obj = {
                'soft_binary_accuracy': soft_binary_accuracy,
                'MinibatchDiscrimination': MinibatchDiscrimination
            }
            self.discriminator.model = load_model(discrim_path,
                                                  custom_objects=custom_obj)
            print(f'Loading {gen_path}...')
            self.generator.model = load_model(gen_path)
            print(f'Loading {combined_path}')
            self.connect_discrim_gen()

            # assumes optimizer state is also saved in the model paths
            self.combined = load_model_no_model_config_from_hdf5(self.combined,
                                                                 combined_path)

            self.d_opt = self.discriminator.model.optimizer
            self.g_opt = self.combined.optimizer

    def compile_models(self, loss='binary_crossentropy',
                       discrim_metrics=[soft_binary_accuracy]):
        """Compiles the models and creates the `combined` model field.

        If you're loading weights, make sure to call this after
        `load_models` to create the `combined` model properly.
        """
        self.discriminator.model.compile(loss=loss,
                                         optimizer=self.d_opt,
                                         metrics=discrim_metrics)
        self.connect_discrim_gen()
        self.combined.compile(loss=loss, optimizer=self.g_opt)

    def train_step(self, real_paths, gt, batch_size=128):
        """Single training step for GAN

        Args:
            real_paths (np.ndarray OR tf.tensor): The paths to model
                with shape (num_paths, path_count, 3)
            gt (np.ndarray): Tuple of the groundtruths.
                First element should be an array of all 1s (real).
                Second element should be an array of all 0s (fake).
            batch_size (int):

        Returns:
            d_loss: [Discriminator loss, accuracy]
            g_loss: generator loss
        """
        real, fake = gt
        # Training the discriminator
        # Select a random batch of note sequences
        idx = np.random.randint(0, real_paths.shape[0], batch_size)
        real_seqs = real_paths[idx]

        # data augmentation
        if self.data_aug:
            real_seqs = self.data_aug(real_seqs)

        noise = self.generator.generate_noise(batch_size)
        # Generate a batch of new note sequences
        gen_seqs = self.generator.model.predict(noise)

        # Train the discriminator
        smoothed_real = smooth_positive_labels(real)
        d_loss = self.train_discriminator_step(real_seqs, gen_seqs,
                                               smoothed_real, fake)
        # Train the generator (to have the discriminator label samples as real)
        g_loss = self.train_generator_step(noise, real)

        return (d_loss, g_loss)

    def train_discriminator_step(self, real_seqs, gen_seqs, real_label,
                                 fake_label):
        """Trains the discriminator and returns the loss.
        """
        self.discriminator.model.trainable = True
        d_loss_real = self.discriminator.model.train_on_batch(real_seqs,
                                                              real_label)
        d_loss_fake = self.discriminator.model.train_on_batch(gen_seqs,
                                                              fake_label)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        return d_loss

    def train_generator_step(self, noise, real_label):
        """Trains the generator and returns the loss.
        """
        self.discriminator.model.trainable = False
        g_loss = self.combined.train_on_batch(noise, real_label)
        return g_loss


def smooth_positive_labels(y):
    return y * 0.9


class AdditiveBasicGAN(BasicGAN):
    """The same as BasicGAN but the discriminator loss is additive instead of
    averaged. (RGAN)
    """

    def __init__(self, discriminator, generator, d_opt, g_opt, data_aug=None,
                 model_paths={}, compile_kwargs={}):
        super().__init__(discriminator=discriminator, generator=generator,
                         d_opt=d_opt, g_opt=g_opt, data_aug=data_aug,
                         model_paths=model_paths,
                         compile_kwargs=compile_kwargs)

    def train_discriminator_step(self, real_seqs, gen_seqs, real_label,
                                 fake_label):
        """Trains the discriminator and returns the loss.
        """
        self.discriminator.model.trainable = True
        d_loss_real = self.discriminator.model.train_on_batch(real_seqs,
                                                              real_label)
        d_loss_fake = self.discriminator.model.train_on_batch(gen_seqs,
                                                              fake_label)
        d_loss = np.add(d_loss_real, d_loss_fake)
        d_loss[1] = d_loss[1]/2  # acc should be averaged
        return d_loss


class AdditiveRoundsGAN(AdditiveBasicGAN):
    """The same as AdditiveBasicGAN but there is the option to asynchronously
    train the discriminator and generator in num_D_rounds and num_G_rounds
    """

    def __init__(self, discriminator, generator, d_opt, g_opt, data_aug=None,
                 num_D_rounds=1, num_G_rounds=1, model_paths={},
                 compile_kwargs={}):
        super().__init__(discriminator=discriminator, generator=generator,
                         d_opt=d_opt, g_opt=g_opt, data_aug=data_aug,
                         model_paths=model_paths,
                         compile_kwargs=compile_kwargs)
        self.num_D_rounds = num_D_rounds
        self.num_G_rounds = num_G_rounds

    def train_step(self, real_paths, gt, batch_size=128):
        """Single training step for RGAN

        Args:
            real_paths (np.ndarray OR tf.tensor): The paths to model
                with shape (num_paths, path_count, 3)
            gt (np.ndarray): Tuple of the groundtruths.
                First element should be an array of all 1s (real).
                Second element should be an array of all 0s (fake).
            batch_size (int):

        Returns:
            d_loss: [Discriminator loss, accuracy]
            g_loss: generator loss
        """
        real, fake = gt
        d_loss_all = []
        d_acc_all = []
        g_loss_all = []
        # Training the discriminator
        # Select a random batch of note sequences
        idx = np.random.randint(0, real_paths.shape[0], batch_size)
        real_seqs = real_paths[idx]

        # data augmentation
        if self.data_aug:
            real_seqs = self.data_aug(real_seqs)

        self.discriminator.model.trainable = True
        # Train the discriminator
        smoothed_real = smooth_positive_labels(real)
        for i in range(self.num_D_rounds):
            noise = self.generator.generate_noise(batch_size)
            # Generate a batch of new note sequences
            gen_seqs = self.generator.model.predict(noise)
            d_metrics = self.train_discriminator_step(real_seqs, gen_seqs,
                                                      smoothed_real, fake)
            d_loss_all.append(d_metrics[0])
            d_acc_all.append(d_metrics[1])

        # Train the generator (to have the discriminator label samples as real)
        for i in range(self.num_G_rounds):
            noise = self.generator.generate_noise(batch_size)
            g_loss_all.append(self.train_generator_step(noise, real))

        return ([np.mean(d_loss_all), np.mean(d_acc_all)],
                np.mean(g_loss_all))


class VanillaGAN(BasicGAN):
    """Basic GAN class for the vanilla models.
    """

    def __init__(self, noise_size, seq_shape, d_opt, g_opt, data_aug=None,
                 model_paths={}, compile_kwargs={}):
        super().__init__(discriminator=VanillaDiscriminator(seq_shape),
                         generator=VanillaGenerator(noise_size, seq_shape),
                         d_opt=d_opt, g_opt=g_opt, data_aug=data_aug,
                         model_paths=model_paths,
                         compile_kwargs=compile_kwargs)


class RGAN(AdditiveRoundsGAN):
    """RGAN class with label smoothing for the discriminator and additive
    discriminator loss during training.

    Architecture is based on the RGAN

    Usage:
        from pymousegan.models.gans import RGAN
        from tensorflow.keras.optimizers import Adam
        opt = Adam(lr=1e-4)
        rgan = RGAN(opt, opt)
        rgan.train(...)
    """

    def __init__(self, d_opt, g_opt, lstm_hu_d=100, lstm_hu_g=100,
                 noise_size=100, seq_shape=(100, 3), data_aug=None,
                 num_D_rounds=1, num_G_rounds=5, gen_act='sigmoid',
                 model_paths={}, compile_kwargs={}):
        discriminator = RGAN.create_discriminator(lstm_hu_d, seq_shape)
        generator = RGAN.create_generator(lstm_hu_g, noise_size, seq_shape,
                                          gen_act)
        super().__init__(discriminator=discriminator,
                         generator=generator,
                         d_opt=d_opt, g_opt=g_opt, data_aug=data_aug,
                         num_D_rounds=num_D_rounds, num_G_rounds=num_G_rounds,
                         model_paths=model_paths,
                         compile_kwargs=compile_kwargs)

    @staticmethod
    def create_discriminator(hidden_units_lstm, seq_shape=(100, 3)):
        from .abstract import Discriminator

        class RGANDiscriminator(Discriminator):
            """Abstract discriminator class for the GAN.
            """

            def __init__(self):
                super().__init__(seq_shape=seq_shape, build_kwargs={})

            def build_model(self):
                """Main method for creating the discriminator model.

                Returns:
                    tf.keras.Model
                """
                x = Input(self.seq_shape)
                lstm = LSTM(hidden_units_lstm)(x)
                out = Dense(1, activation='sigmoid')(lstm)
                return tf.keras.Model(inputs=x, outputs=out,
                                      name='rgan_discriminator')

        return RGANDiscriminator()

    @staticmethod
    def create_generator(hidden_units_lstm, noise_size=(100, 100),
                         seq_shape=(100, 3), gen_act='sigmoid'):
        from .abstract import Generator

        class RGANGenerator(Generator):
            """RGAN generator.
            """

            def __init__(self):
                super().__init__(rand_noise_size=noise_size,
                                 seq_shape=seq_shape,
                                 build_kwargs={})

            def build_model(self):
                """Main method for creating the generator model.

                Returns:
                    tf.keras.Model or tf.Sequential()
                """
                # Original was [seq_length, latent_dim]
                x = Input(shape=self.noise_size)
                lstm = LSTM(hidden_units_lstm)(x)
                out = Dense(np.prod(seq_shape),
                            activation=gen_act)(lstm)
                reshaped_out = Reshape(self.seq_shape)(out)
                return tf.keras.Model(inputs=x, outputs=reshaped_out,
                                      name='rgan_generator')
        return RGANGenerator()


class ECGGAN(AdditiveBasicGAN):
    """Combination of CNNs and LSTMs.

    This architecture is inspired by:
    https://github.com/MikhailMurashov/ecgGAN/blob/master/ecgGAN.ipynb
    """

    def __init__(self, d_opt, g_opt, seq_shape=(100, 3), noise_size=(100, 100),
                 discrim_filters_list=[32, 64, 128, 256],
                 gen_filters_list=[128, 64, 32, 16], mbd_units=5, mbd_rows=16,
                 gen_act='tanh', data_aug=None, model_paths={},
                 compile_kwargs={}):
        """
        Args:
            mbd_units: Number of units for the MinibatchDiscrimination layer
            mbd_rows: Number of rows for the MinibatchDiscrimination layer
        """
        discriminator = ECGGAN.create_discriminator(seq_shape,
                                                    discrim_filters_list,
                                                    mbd_units, mbd_rows)
        generator = ECGGAN.create_generator(noise_size, seq_shape,
                                            gen_filters_list, gen_act)
        super().__init__(discriminator=discriminator,
                         generator=generator,
                         d_opt=d_opt, g_opt=g_opt, data_aug=data_aug,
                         model_paths=model_paths,
                         compile_kwargs=compile_kwargs)

    @staticmethod
    def create_discriminator(seq_shape=(100, 3),
                             num_filters_list=[32, 64, 128, 256],
                             mbd_units=5, mbd_rows=16):
        from .abstract import MinibatchDiscriminator

        class ECGGANDiscriminator(MinibatchDiscriminator):
            """Fully convolutional discriminator.
            """

            def __init__(self):
                super().__init__(seq_shape=seq_shape,
                                 minibatch_discrim_units=mbd_units,
                                 minibatch_discrim_row_size=mbd_rows,
                                 build_kwargs={})

            def build_model(self, minibatch_discrim=None):
                """Main method for creating the discriminator model.

                Returns:
                    tf.keras.Model
                """
                input_layer = Input(self.seq_shape)
                for i, units in enumerate(num_filters_list):
                    # first conv takes input layer
                    if i == 0:
                        x = ECGGAN.conv_block(input_layer, units, index=i)
                    else:  # rest just take the previous leakyReLU
                        x = ECGGAN.conv_block(x, units, index=i)
                features = Flatten()(x)
                # Minibatch Discrimination - to handle single point mode
                # collapse
                if isinstance(minibatch_discrim, dict):
                    features = MinibatchDiscrimination(
                        **minibatch_discrim)(features)
                out = Dense(1, activation='sigmoid')(features)

                return tf.keras.Model(inputs=input_layer, outputs=out,
                                      name='ecggan_discriminator')

        return ECGGANDiscriminator()

    @ staticmethod
    def create_generator(noise_size=(100, 100), seq_shape=(100, 3),
                         num_filters_list=[128, 64, 32, 16],
                         gen_act='tanh'):
        from .abstract import Generator

        class ECGGANGenerator(Generator):
            """ECGGAN generator. Outputs directly with convolutions.
            """

            def __init__(self):
                super().__init__(rand_noise_size=noise_size,
                                 seq_shape=seq_shape,
                                 build_kwargs={})

            def build_model(self):
                """Main method for creating the generator model.

                Returns:
                    tf.keras.Model or tf.Sequential()
                """
                # Original was [seq_length, latent_dim]
                input_layer = Input(shape=self.noise_size)
                x = Bidirectional(LSTM(64, return_sequences=True,
                                       name='bilstm_1'))(input_layer)

                for i, units in enumerate(num_filters_list):
                    x = ECGGAN.conv_block(x, units, index=i)

                out_conv = Conv1D(self.seq_shape[-1], kernel_size=16,
                                  strides=1, padding='same',
                                  activation=gen_act)(x)
                return tf.keras.Model(inputs=input_layer, outputs=out_conv,
                                      name='ecggan_generator')
        return ECGGANGenerator()

    @staticmethod
    def conv_block(input_tensor, units, index=0, output_act=None):
        layer = Conv1D(units, kernel_size=16, strides=1, padding='same',
                       activation=output_act,
                       name=f'conv_{index}')(input_tensor)
        act = LeakyReLU()(layer)
        return act
