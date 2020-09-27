import numpy as np
import json
from tqdm import tqdm


def get_stats(events):
    """
    Min, Max, Mean, STD
    """
    events = np.array(events)
    print(f'Min: {events.min()}, Max: {events.max()},\nMean: {events.mean()}, \
            \nSTD: {events.std()}')


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
    norm_img = ((norm_range[1]-norm_range[0]) * (arr - min) / (max - min)) + norm_range[0]
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
    arr = ((norm_arr - norm_range[0]) / (norm_range[1]-norm_range[0]) * (max - min)) + min 
    return arr


def parse_data_into_coords_and_time_diffs(data_list):
    """Parses the data json dicttionary list into coordinates.
    
    Args:
        data_list (list[dict]: The original json loaded as a list of
            dictionaries
    
    Returns:
        coords_all (np.ndarray(list[[X, Y]])): numpy array of lists of
            paths (list of coordinates). The shape is (num_paths,), and
            there is no other dimension due to inconsistent path list length.
        time_diffs_all (np.ndarray(List[int])): numpy array of lists of
            timeDiffs. The shape is (num_paths,), and there is no other
            dimension due to inconsistent path list length.
    """
    # nested list of all lists of coordinates
    coords_all = []
    time_diffs_all = []
    for session_dict in tqdm(data_list):
        events = json.loads(session_dict['events'])
        # formatting so the list[dict] to list[[X, Y]]
        coords = []
        time_diffs = []
        for entry in events:
            # appending coordinate
            if 'clientX' in list(entry.keys()) and 'clientY' in list(entry.keys()):
                coords.append([entry.pop('clientX'), entry.pop('clientY')])
            if 'timeDiff' in list(entry.keys()):
                time_diffs.append(entry.pop('timeDiff'))
        # appending that session's path (list of coordinates)
        coords_all.append(coords), time_diffs_all.append(time_diffs)
    return np.asarray(coords_all), np.asarray(time_diffs_all)


def get_idx_n_path_length(coords_arr, n=100):
    """Gets all indices of the coords/time diffs arrays where there are >=n
    paths in the path lists (each sublist in the coord/time diff arrays).

    Args:
        coords_arr (np.ndarray(list[[X, Y]])): numpy array of lists of
            paths (list of coordinates). The shape is (num_paths,), and
            there is no other dimension due to inconsistent path list length.
        - generally, the output of `parse_data_into_coords_and_time_diffs`

    Returns:
        indices (list[int]): list of the indices to slice with
    """
    indices = [i for (i, paths_list) in \
               tqdm(enumerate(coords_arr), total=len(coords_arr)) \
               if len(paths_list) >= n]
    return indices


def extract_1st_n(arr, n=100):
    """Extracts the first `n` elements of the sublists in `arr`.

    Args:
        arr (np.ndarray): an array with nested iterables.
        n (int): number of elements to extract from each sublist

    Returns:
        list_n (np.ndarray): (converted to np.ndarray)
            arr, but with the sublists with only the first `n` elements.
            with the shape (len(arr), n, ..)
    """
    # get first 100 coords for the paths lists (sublists)
    list_n = [sub_list[:n] for sub_list in tqdm(arr) \
              if len(sub_list) >= n]
    if len(list_n) != len(arr):
        print(f'Warning! Some element sublists in the input are <{n} long.')
    return np.asarray(list_n)


def time_diff_to_dt(time_diffs):
    """Converts the time diffs to the actual changes in time.
    
    Args:
        time_diffs (List[int] or a 1D Iterable): iterable of timeDiffs

    Returns:
        dt_list (List[int]): list of delta times
            The delta times are calculated as dt = current-prev rather than
            with respect to the starting point.
    """
    start_time = time_diffs[0]
    dt_list = []
    for i in range(len(time_diffs)):
        if i == 0: dt_list.append(0)
        else:
            dt_list.append(time_diffs[i]-time_diffs[i-1])
    return dt_list


def preprocess_coords(coords_arr):
    """Preprocesses the coordinates

    1. Normalize with offset.

    Args:
        coords_arr (np.ndarray): of arrays of coordinates with shape
            (num_paths, path_count, 2)
    
    Returns:
        normalized_paths (np.ndarray): preprocessed path coords
            with the same shape as `coords_arr`
        offset (np.ndarray): the offset to unnormalize the array
            start + offset = [0, 0]
            pred - offset = pred
    """
    # normalize to origin
    # normalize the starting point to origin
    # so that start + offset = [0, 0] & pred (assumes [0, 0])
    # so pred - offset = pred assuming start as the starting point
    offset = -coords_arr[:, 0][:, None]
    normalized_paths = np.add(coords_arr, offset)
    return (normalized_paths, offset)


def time_diffs_to_dt_all(time_diffs_arr):
    """Converts all of the time diffs to changes in time (wrt to the previous
    time).
    
    Args:
        time_diffs_arr (Iterable[Iterable[int]]): Nested iterable of an
            iterable of integers (the time diffs).
                - This is generally a np.ndarray with shape 
                (num_paths, path_count).
    
    Returns:
        a np.ndarray with the same shape as `time_diffs`

    """
    time_diffs_new = [time_diff_to_dt(time_diffs) \
                      for time_diffs in time_diffs_arr]
    return np.asarray(time_diffs_new)


def get_destinations(coords_arr):
    """Gets the destination coordinates from an array of paths.

    Args:
        coords_arr (np.ndarray): with shape (num_paths, path_count, 2)

    Returns:
        destinations (np.ndarray): with shape (num_paths, 2)
    """
    destinations = np.zeros((len(coords_arr), 2))
    for (i, path_arr) in enumerate(coords_arr):
        destinations[i] = path_arr[-1]
    return destinations


def preprocess(data_list, path_count=100, combine_coords_dt=True):
    """Preprocesses the loaded list of dictionaries (from json).

    Args:
        data_list (List[dict]):
        path_count (int): number of path coordinates/dt lists for each path
        combine_coords_dt (boolean): Whether or not to combine the coordinates
            array and the dt arrays. Defaults to True. Refer to the 'Returns'
            documentation below for more information.

    Returns:
        destinations (np.ndarray): with shape (num_paths, 2)
            The last element of each path
        offset (np.ndarray): the offset to unnormalize the array
            start + offset = [0, 0]
            pred - offset = pred

        if combine_coords_dt:
            normalized_paths_dt (np.ndarray): with shape
                (num_paths, path_count, 3)
                where each path is [x, y, dt] and dt is the time elapsed
                from previous 
        else:
            coords (np.ndarray): with shape (num_paths, path_count, 2)
                where each path is [x, y]
            dt_arr (np.ndarray): with shape (num_paths, path_count)
                where each `path` contains the dt's corresponding to the
                respective coordinates in `coords`.
    """
    # getting the path lists to have a constant length (path count)
    coords, time_diffs = parse_data_into_coords_and_time_diffs(data_list)
    # all places where the path_count condition is met
    idx = get_idx_n_path_length(coords, n=path_count)
    # extracting those lists and the 1st n elements of those lists/arrays
    coords = extract_1st_n(coords[idx], n=path_count)
    time_diffs = extract_1st_n(time_diffs[idx], n=path_count)
    # processing coords and converting time diffs to dt
    coords, offset = preprocess_coords(coords)
    destinations = get_destinations(coords)
    dt_arr = time_diffs_to_dt_all(time_diffs)
    # clipping the dt to 1s because some are REALLY long (>5 seconds)
    dt_arr = np.clip(dt_arr, 0, 1000)

    if combine_coords_dt:
        return (destinations, offset, np.dstack([coords, dt_arr]))
    else:
        return (destinations, offset, coords, dt_arr)
