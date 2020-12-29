import numpy as np

from .preprocessor import mode


def json_to_numpy(data_list):
    """Converts the original data.json to a numpy array
    """
    arr = np.zeros((len(data_list), 100, 3))
    for i in range(len(data_list)):
        x = data_list[i]['x']
        y = data_list[i]['y']
        t = data_list[i]['t']
        path_dt = list(zip(x, y, t))
        arr[i] = np.asarray(path_dt)
    return arr


def t_to_dt_single(t):
    """Converts a single time-elapsed distribution to a distribution of lags.

    Args:
        t (1D np.ndarray/list/tuple): with n elements of elapsed times
    Returns:
        a numpy array of n lags
    """
    dt = np.zeros((len(t)))
    for i, elapsed in enumerate(t):
        if i == 0:
            dt[i] = 0
        else:
            dt[i] = t[i] - t[i-1]
    return dt


def t_to_dt(t_arr):
    """Converts the array of elapsed times to time differences (lags).
    """
    dt_arr = np.zeros(t_arr.shape)
    for i, t in enumerate(t_arr):
        dt_arr[i] = t_to_dt_single(t)
    return dt_arr


def minmax_normalize(arr, norm_range=[-1, 1], minmax=None):
    """
    Args:
        arr: numpy array
        minmax (Iterable[float]): (min, max) where
            min (float): minimum of the dataset
            max (float): maximum of the dataset
        norm_range: list of 2 integers specifying normalizing range

    Returns:
        Normalized array with outliers clipped in the specified range
    """
    if minmax is not None:
        min, max = minmax
    else:
        min, max = np.amin(arr), np.amax(arr)
    scale = (norm_range[1]-norm_range[0])
    norm_img = (scale * (arr - min) / (max - min)) + norm_range[0]
    return norm_img


def minmax_unnormalize(norm_arr, minmax, norm_range=[-1, 1]):
    """Unnormalizing an array after predicting. Undos minmax_normalize
    Args:
        norm_arr (np.ndarray): prediction array with shape
            (1, path_count, 3)
        minmax (Iterable[float]): (min, max) where
            min (float): minimum of the original test set (for prediction)
            max (float): maximum of the original test set (for prediction)
        norm_range: list of 2 integers specifying normalizing range

    Returns:
        the rescaled array
    """
    min, max = minmax
    arr = ((norm_arr - norm_range[0]) /
           (norm_range[1]-norm_range[0]) * (max - min)) + min
    return arr


def remove_bad_indices(paths, bad_indices):
    """Removes all arrays with indices in bad_coords from paths.
    """
    all_idx = list(range(paths.shape[0]))
    for bad_idx in bad_indices:
        bad_idx = int(bad_idx)
        if bad_idx in all_idx:
            all_idx.remove(bad_idx)
    assert (paths.shape[0] - len(bad_indices)) == len(all_idx)
    return paths[all_idx]


def get_oob_paths_idx(paths, value_thresh=1):
    """Get indices for the paths that have values > 1.
    """
    bad_indices = np.unique(np.where(paths > value_thresh)[0])
    print(f'Number of Paths with Coords > 1: {len(bad_indices)}')
    return bad_indices


def get_excessive_loops_idx(paths, num_loops=30):
    """Get indices for the paths that have excessive loops.
    Problem: Also removes a lot of paths that go straight up, but it's
    kinda okay because a lot of those paths are janky af.
    """
    bad_indices = []
    for i, arr in enumerate(paths):
        if mode(arr[:, 0])[1] >= num_loops:
            bad_indices.append(i)
    print(f'Number of Excessive Loops: {len(bad_indices)}')
    return bad_indices


def get_nan_idx(self, paths):
    """When the destination has 0 in either X, Y
    """
    bad_indices = np.unique(np.where(np.isnan(paths))[0])
    print(f'Number of Paths with NaN: {len(bad_indices)}')
    return bad_indices


def get_bad_dt(dt, dt_thresh=1000, elapsed_thresh=2000):
    dt_bad = np.unique(np.where(dt > dt_thresh)[0]).tolist()
    elapsed = np.sum(dt, axis=1)
    elapsed_bad = np.unique(np.where(elapsed > elapsed_thresh)[0]).tolist()
    print(f'{len(dt_bad)} paths that have a lag > dt_thresh.')
    print(f'{len(elapsed_bad)} paths that exceed elapsed_thresh.')
    return dt_bad + elapsed_bad


def remove_outliers(coords_dt, dt_thresh=1000, elapsed_thresh=2000,
                    value_thresh=1.05, num_loops=30):
    """Removes outliers from coords_dt array.

    Currently, only removes slow paths (when a path has a dt > dt_thresh OR
    when the total elapsed time is > elapsed_thresh).
    """
    # looking only at the dt
    bad_dt = get_bad_dt(coords_dt[:, :, -1], dt_thresh=dt_thresh,
                        elapsed_thresh=elapsed_thresh)

    # looking at the coords
    oob = get_oob_paths_idx(coords_dt[:, :, :-1], value_thresh=value_thresh)
    excessive = get_excessive_loops_idx(coords_dt[:, :, :-1],
                                        num_loops=num_loops)
    # removing the bad paths
    bad_path_idx = np.unique(bad_dt + oob + excessive)
    print(f'{len(bad_path_idx)} total number of outliers.')
    return remove_bad_indices(coords_dt, bad_path_idx)


def convert_and_preprocess(data_list, dt_thresh=1000, elapsed_thresh=1500,
                           value_thresh=1.05, num_loops=30):
    coords_dt = json_to_numpy(data_list)
    coords_dt[:, :, -1] = t_to_dt(coords_dt[:, :, -1])
    coords_dt = remove_outliers(coords_dt, dt_thresh, elapsed_thresh,
                                value_thresh=value_thresh, num_loops=num_loops)
    print('Before minmax normalization;' +
          f'dt min: {coords_dt[:, :, -1].min()}' +
          f'dt max: {coords_dt[:, :, -1].max()}')
    coords_dt[:, :, :-1] = minmax_normalize(coords_dt[:, :, :-1])
    coords_dt[:, :, -1] = minmax_normalize(coords_dt[:, :, -1], [0, 1])
    return coords_dt
