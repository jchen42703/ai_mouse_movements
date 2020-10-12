import numpy as np


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


def scale_coords_all(coords, scale_range=[1924, 1924], stay_pos=False):
    """Scales the coords to match the destination's coords.

    Args:
        coords (np.ndarray): (num_paths, path_count, 2 (or 3))
            assumes that each coordinate path starts at (0, 0)
        dest (np.ndarray): the min and max of this array determine the range
            to scale the coords to. Assumes that dest hasa the shape
            (num_paths, 2 (or 3))
    
    Returns:
        coords (np.ndarray): scaled array, such that the maximum of each 
            dimenion (x, y) matches the destination for each path.
            (num_paths, path_count, 2 (or 3))
    """
    coords, neg_idx = neg2pos(coords, copy=True)

    x_scale_factor = scale_range[0] / coords[:, :, 0].max()
    y_scale_factor = scale_range[1] / coords[:, :, 1].max()

    print(f'X Scale Factor: {x_scale_factor}')
    print(f'Y Scale Factor: {y_scale_factor}')

    coords[:, :, 0] = coords[:, :, 0] * x_scale_factor
    coords[:, :, 1] = coords[:, :, 1] * y_scale_factor

    # convert back to negative
    if not stay_pos:
        coords = pos2neg(coords, neg_idx)

    return coords


def predict_scale(start, dest, pred_model):
    """Official python version of the scale/predict API
    """
    offset = start * -1
    dest = dest + offset
    pred = pred_model.predict(dest)
    pred = scale_coords(pred, dest)
    pred = pred - offset
    return pred
)
