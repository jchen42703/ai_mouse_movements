import unittest
import os
import numpy as np

from scale import reflect


class ScaleTests(unittest.TestCase):
    """
    Testing all the functions in `utils.intensity_io.py`:
    * minmax_normalize
    * normalize_clip
    * whiten
    * clip_upper_lower_percentile
    """
    def setUp(self):
        self.shape = (3, 100, 2)
        self.rand_coords = np.arange(np.product(self.shape)).reshape(self.shape)


    def test_reflect(self):
        """Tests the `reflect` function.
        """
        reflected = reflect(self.rand_coords)
        out_shape = (self.shape[0] * 4, self.shape[1], self.shape[2])
        self.assertTrue(reflected.shape == out_shape)

        # checks that the signs are correctly changed for each coord
        for check_idx in range(self.shape[0]):
            check_arrs = []
            # each arr in check_arrs is a reflection of the first check arr
            for i in range(4):
                check_arrs.append(reflected[check_idx + self.shape[0]*i])
            # x axis, y-axis sign checks
            # lists of the 1st x/y coords in each check arr
            x_coords = [arr[0][0] for arr in check_arrs]
            y_coords = [arr[0][1] for arr in check_arrs]

            x_scale_order = np.asarray([1, -1, 1, -1]) * x_coords[0]
            y_scale_order = np.asarray([1, 1, -1, -1]) * y_coords[0]

            self.assertTrue(np.array_equal(x_coords, x_scale_order))
            self.assertTrue(np.array_equal(y_coords, y_scale_order))


unittest.main(argv=[''], verbosity=2, exit=False)
