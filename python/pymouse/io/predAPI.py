import numpy as np
import json
from tensorflow.keras.models import load_model
from pymouse.io.orig_preprocessor import translate_to_origin


class PredictAPI(object):
    def __init__(self, model_path, dest_json_path):
        self.model = load_model(model_path)
        with open(dest_json_path, 'r') as fp:
            self.dest_list = json.load(fp)
        print(f'Choosing from {len(self.dest_list)} destinations.')

    def get_rand_destination(self):
        """Gets a random destination coordinate from `dest.json`.
        """
        rand_idx = np.floor(np.random.uniform() * len(self.dest_list))
        return self.dest_list[int(rand_idx)]

    @staticmethod
    def scale_to_dest_single(coords, dest):
        """Scales coordinates such that the last coordinate matches `dest`.

        Args:
            coords (np.ndarray): with shape (1, 100, 3) or (100, 3)
            dest (list[float]): [X, Y] coordinates
        
        Returns:
            np.ndarray with shape (1, 100, 3) such that the last coordinate of
            the array has the X, Y = dest
        """
        coords = coords.squeeze()
        x_scale_factor = dest[0] / coords[-1][0]
        y_scale_factor = dest[1] / coords[-1][1]

        print(f'Scale factors: ({x_scale_factor}, {y_scale_factor})')

        scaled = coords * np.asarray([x_scale_factor, y_scale_factor, 1])
        return scaled

    @staticmethod
    def postprocess(pred, start, dest):
        """Postprocesses the prediction.

        Args:
            pred (np.ndarray): prediction of model (1, 100, 3)
            start, dest (lists): the start/destination [X, Y] coordinates
        
        Returns
            np.ndarray with shape (1, 100, 3)
        """
        # 1. translate to origin
        translated, _ = translate_to_origin(pred)
        # 2. Scale to (destination + offset)
        # Offset = -start
        new_dest = np.asarray(start) * (-1) + np.asarray(dest)
        scaled = PredictAPI.scale_to_dest_single(translated,
                                                 dest=new_dest.tolist())
        # 3. Remove offset from predictions (pred - offset)
        offset = np.asarray([-start[0], -start[1], 0]).reshape((1, 3))
        out = scaled - offset
        return out

    def predict(self, start, dest):
        """Predicts and postprocesses.

        Args:
            start, dest (lists): the start/destination [X, Y] coordinates

        Returns:
            processed prediction with shape (1, 100, 3)
        """
        pos_dest = self.get_rand_destination()
        pred = self.model.predict(np.asarray(pos_dest).reshape(1, 2))
        pred = PredictAPI.postprocess(pred, start, dest)
        return pred
