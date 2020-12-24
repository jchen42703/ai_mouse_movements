import unittest
import os
import numpy as np

from tensorflow.keras.models import load_model

from pymouse.io.preprocess import minmax_normalize, minmax_unnormalize


class ModelPredictionsTest(unittest.TestCase):
    """Sanity checks on model predictions.
    """
    def setUp(self):
        """
        Initializing the parameters:
            n_channels:
            shapes: tuples with channels_first shapes
            images & labels: numpy arrays
            extractors: 2D and 3D versions for all the patch extractor classes.
        """
        # bringing the current working dir to ai_mouse_movements/
        os.chdir(os.path.abspath('../'))
        os.chdir(os.pardir)

        weights_name = 'lstm_weights_4mil_2250epochs_constant_lr.h5'
        model_path = os.path.join('js/src/model/', weights_name)
        self.model = load_model(model_path)


    def test_integer_predictions(self):
        """Checks that the predictions on integer destinations are acceptable.
        """
        # sample dest
        dest_test = [[600, 500], [300, 200], [200, 100], [1924, 5], [5, 1924],
                     [0, 0], [50, 50]]

        for dest_sample in dest_test:
            dest = np.asarray(dest_sample)[None] # shape (1, 2)
            dest_norm = minmax_normalize(dest, norm_range=[0, 1],
                                         minmax=[-2265.0, 2328.0])
            pred = self.model.predict(dest_norm)
            pred_unnorm = minmax_unnormalize(pred, minmax=[-2265.0, 2328.0],
                                             norm_range=[0, 1])
            
            self.assertTrue(np.where(pred_unnorm < 0)[0].size == 0,
                            'There should be no negative coordinates.')


unittest.main(argv=[''], verbosity=2, exit=False)