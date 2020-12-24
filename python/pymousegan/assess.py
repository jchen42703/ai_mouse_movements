import numpy as np
# from tensorflow.keras.models import load_model
from pymousegan.io.preprocessor import scale_coords_uniform_dest, Preprocessor


class BadPathChecker(Preprocessor):
    def __init__(self):
        # arbitrary assignment
        super().__init__(True, True, None)

    def get_oob_paths_idx(self, paths, threshold=0.5):
        """Get indices for the paths that have values > 1 + threshold.
        """
        bad_indices = np.unique(np.where(paths - 1 > threshold)[0])
        print(f'Number of Paths with Coords > 1: {len(bad_indices)}')
        self.bad_indices.extend(bad_indices)

    def check_bad_paths(self, model, num_rounds=10, noise_size=(100, 100)):
        """Tracks how many "bad paths" there are after `num_rounds` predictions.

        A bad path would be a path where if you translate the origin to (0, 0)
        and scale to the first quadrant, the path stretches out of bounds.
        """
        gen_size = [num_rounds] + list(noise_size)
        noise = np.random.normal(0, 1, size=gen_size)
        pred = model(noise).numpy()
        paths, dt = process_coords_dt(pred)

        self.get_oob_paths_idx(paths)
        self.get_nan_idx(paths)
        self.bad_indices = np.unique(self.bad_indices)
        print(f'{len(self.bad_indices)} / {num_rounds} are bad paths.')
        return (paths, dt)


def process_coords_dt(coords_dt):
    paths, dt = separate_coords_dt(neg2pos(coords_dt)[0])
    paths = scale_coords_uniform_dest(translate_to_origin(paths)[0])
    return (paths, dt)


def neg2pos(coords, copy=True):
    """Converts the negative coords to positive.
    Args:
        coords (np.ndarray): (num_paths, path_count, 2 (or 3))
    Returns:
        coords (np.ndarray): (num_paths, path_count, 2 (or 3))
        neg_idx (tuple[np.ndarray]): indices where the original coords are
            negative
    """
    if copy:
        coords = np.copy(coords)
    neg_idx = np.where(coords < 0)
    coords[neg_idx] = coords[neg_idx] * -1
    return (coords, neg_idx)


def translate_to_origin(paths):
    """Translates the starting points of the paths to (0, 0)
    Args:
        paths (np.ndarray): of arrays of coordinates with shape
            (num_paths, path_count, 2)
    Returns:
        paths (np.ndarray): translated path coords
            with the same shape as `paths`
        offset (np.ndarray): the offset to unnormalize the array
            start + offset = [0, 0]
            pred - offset = pred
    """
    # normalize to origin
    # normalize the starting point to origin
    # so that start + offset = [0, 0] & pred (assumes [0, 0])
    # so pred - offset = pred assuming start as the starting point
    offset = -paths[:, 0][:, None]
    paths = np.add(paths, offset)
    return (paths, offset)


def separate_coords_dt(coords_dt):
    """Separates the 3D coords_dt into the 2D path and 1D dt_arr.
    """
    paths = np.zeros((coords_dt.shape[0], coords_dt.shape[1], 2))
    dt_arr = np.zeros((coords_dt.shape[0], coords_dt.shape[1], 1))
    for i, arr in enumerate(coords_dt):
        paths[i] = arr[:, :-1]
        dt_arr[i] = arr[:, -1:]
    return (paths, dt_arr)
