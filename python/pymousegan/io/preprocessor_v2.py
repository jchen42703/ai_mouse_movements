import numpy as np


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
            based on https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
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
            based on https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1   

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


def remove_outliers(coords_dt, dt_thresh=1000, elapsed_thresh=2000):
    """Removes outliers from coords_dt array.

    Currently, only removes slow paths (when a path has a dt > dt_thresh OR
    when the total elapsed time is > elapsed_thresh).
    """
    # looking only at the dt rn
    dt_bad = np.unique(np.where(coords_dt[:, :, -1] > dt_thresh)[0]).tolist()
    elapsed = np.sum(coords_dt[:, :, -1], axis=1)
    elapsed_bad = np.unique(np.where(elapsed > elapsed_thresh)[0]).tolist()
    print(f'{len(dt_bad)} paths that have a lag > dt_thresh.')
    print(f'{len(elapsed_bad)} paths that exceed elapsed_thresh.')
    bad_path_idx = np.unique(elapsed_bad + dt_bad)
    print(f'{len(bad_path_idx)} total number of outliers.')
    return remove_bad_indices(coords_dt, bad_path_idx)


def preprocess(data_list, dt_thresh=1000, elapsed_thresh=1500):
    coords_dt = json_to_numpy(data_list)
    coords_dt[:, :, -1] = t_to_dt(coords_dt[:, :, -1])
    coords_dt = remove_outliers(coords_dt, dt_thresh, elapsed_thresh)
    print('Before minmax normalization;' +
          f'dt min: {coords_dt[:, :, -1].min()}' +
          f'dt max: {coords_dt[:, :, -1].max()}')
    coords_dt[:, :, :-1] = minmax_normalize(coords_dt[:, :, :-1])
    coords_dt[:, :, -1] = minmax_normalize(coords_dt[:, :, -1], [0, 1])
    return coords_dt
