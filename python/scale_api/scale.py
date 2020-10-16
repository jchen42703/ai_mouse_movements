import numpy as np


class Preprocessor(object):
    def __init__(self, model_mode='ann', combine_dt=True,
                 reflect_coords=True):
        if model_mode == 'ann':
            self.scale_maxes = [1924, 1924]
        elif model_mode == 'lstm':
            self.scale_maxes = [1, 1]
        
        self.COMBINE_DT_FLAG = combine_dt
        self.REFLECT_COORDS_FLAG = reflect_coords


    def preprocess_original_paths(self, paths):
        """Preprocesses only the paths (not the paths and time differences)
        """
        paths, _ = translate_to_origin(paths)
        coords = scale_coords_all(paths, scale_range=self.scale_maxes,
                                  stay_pos=True)
        if self.REFLECT_COORDS_FLAG:
            coords = reflect(coords)
        # since all input destinations are the same as the last coords of each
        # path, then we can just create a destination array


def neg2pos(coords, copy=False):
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


def pos2neg(coords, neg_idx):
    """Undos neg2pos.
    
    Args:
        coords (np.ndarray): (num_paths, path_count, 2 (or 3))
        neg_idx (tuple[np.ndarray]): indices where the original coords are
            negative
    
    Returns:
        coords (np.ndarray): (num_paths, path_count, 2 (or 3))
    """
    coords[neg_idx] = coords[neg_idx] * -1
    return coords


def scale_coords_all(coords, scale_maxes=[1924, 1924], stay_pos=False):
    """Scales the coords to match the destination's coords.

    Args:
        coords (np.ndarray): (num_paths, path_count, 2 (or 3))
            assumes that each coordinate path starts at (0, 0)
        scale_maxes (list/tuple): [max of x, max of y]
        stay_pos (boolean): whether or not the coordinates should stay as
            positive only or back to their original negative positions
    
    Returns:
        coords (np.ndarray): scaled array, such that the maximum of each 
            dimenion (x, y) matches the destination for each path.
            (num_paths, path_count, 2 (or 3))
    """
    coords, neg_idx = neg2pos(coords, copy=True)

    x_scale_factor = scale_maxes[0] / coords[:, :, 0].max()
    y_scale_factor = scale_maxes[1] / coords[:, :, 1].max()

    print(f'X Scale Factor: {x_scale_factor}')
    print(f'Y Scale Factor: {y_scale_factor}')

    coords[:, :, 0] = coords[:, :, 0] * x_scale_factor
    coords[:, :, 1] = coords[:, :, 1] * y_scale_factor

    # convert back to negative
    if not stay_pos:
        coords = pos2neg(coords, neg_idx)

    return coords


def scale_coords(path_coords, scale_maxes=[1924, 1924], stay_pos=False):
    """Scales single coords to match the destination's coords.

    Args:
        path_coords (list): path; a nested list of coords (x, y, dt)
            assumes that each coordinate path starts at (0, 0)
        scale_maxes (list/tuple): [max of x, max of y]
        stay_pos (boolean): whether or not the coordinates should stay as
            positive only or back to their original negative positions
    
    Returns:
        coords (np.ndarray): scaled array, such that the maximum of each 
            dimenion (x, y) matches the destination for each path.
            (1, path_count, 2 (or 3))
    """
    path_coords = np.asarray(path_coords)[None]
    return scale_coords_all(path_coords, scale_maxes=scale_maxes,
                            stay_pos=stay_pos)


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


def reflect(paths):
    """
    Args:
        paths (np.ndarray): (num_paths, path_count, 2)
    
    Returns:
        paths (np.ndarray): (num_paths * 4, path_count, 2)
            original paths along with the reflected paths
    """
    num_paths = paths.shape[0]
    paths = np.tile(paths, np.asarray([4, 1, 1])) # copying paths 4 times

    reflect_multiples = [np.asarray([1, 1]), np.asarray([-1, 1]),
                         np.asarray([1, -1]), np.asarray([-1, -1])]

    for i, multiple in enumerate(reflect_multiples):
        # reference to the manipulated section
        paths[num_paths*i:num_paths*(i+1)] *= multiple
    return paths


def predict_scale(start, dest, pred_model):
    """Official python version of the scale/predict API
    """
    offset = start * -1
    dest = dest + offset
    pred = pred_model.predict(dest)
    pred = scale_coords(pred, dest)
    pred = pred - offset
    return pred
