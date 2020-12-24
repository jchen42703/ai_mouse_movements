import unittest
from pymousegan.models.vanilla import VanillaGenerator, DenseBNGenerator


class VanillaTester(unittest.TestCase):

    def test_VanillaGenerator(self):
        """Tests that VanillaGenerator loads properly.
        """
        generator = VanillaGenerator((100,), seq_shape=(100, 3))
        self.assertTrue(generator.generate_noise(16).shape == (16, 100),
                        'The generated noise should be shaped properly.')
        generator.model.summary()

    def test_DenseBNGenerator(self):
        """Tests that DenseBNGenerator loads properly.
        """
        generator = DenseBNGenerator((100,), seq_shape=(100, 3),
                                     hidden_units_list=[512, 300])
        self.assertTrue(generator.generate_noise(16).shape == (16, 100),
                        'The generated noise should be shaped properly.')
        generator.model.summary()


unittest.main(argv=[''], verbosity=2, exit=False)
