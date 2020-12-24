import numpy as np
import os

class Preprocessor(object):
    """Class for preprocessing data.
    
    All preprocessing procedures:
    * Scale all of the paths to have a destination of [1, 1].
    * Option to remove all of the coordinates with large loops
        * This exists if you have issues scaling the coordinates and they go
        OOB during post-processing.
    * Option to combine the coordinates and the time differences (delays).
    """
    def __init__(self, filter_coords=True, combine_dt=True, save_dir=None):
        """
        Args:
            filter_coords (boolean): Whether or not to filter the coordinates
            save_dir (None or str): Path to the directory to where the .npy
                files will be saved in.
        """
        self.FILTER_FLAG = filter_coords
        self.COMBINE_COORDS_DT_FLAG = combine_dt
        self.save_dir = save_dir
        self.bad_indices = [] # coordinates to remove

    @staticmethod
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

    def get_oob_paths_idx(self, paths):
        """Get indices for the paths that have values > 1.
        """
        bad_indices = np.unique(np.where(paths > 1)[0])
        print(f'Number of Paths with Coords > 1: {len(bad_indices)}')
        self.bad_indices.extend(bad_indices)

    def get_excessive_loops_idx(self, paths):
        """Get indices for the paths that have excessive loops.
        
        Problem: Also removes a lot of paths that go straight up, but it's
        kinda okay because a lot of those paths are janky af.
        """
        bad_indices = []
        for i, arr in enumerate(paths):
            if mode(arr[:, 0])[1] >= 30:
                bad_indices.append(i)
        print(f'Number of Excessive Loops: {len(bad_indices)}')
        self.bad_indices.extend(bad_indices)
    
    def get_nan_idx(self, paths):
        """When the destination has 0 in either X, Y
        """
        bad_indices = np.unique(np.where(np.isnan(paths))[0])
        print(f'Number of Paths with NaN: {len(bad_indices)}')
        self.bad_indices.extend(bad_indices)

    def preprocess_coords(self, paths, dest=[1, 1]):
        coords = scale_coords_uniform_dest(paths, dest=dest)
        if self.FILTER_FLAG:
            self.get_oob_paths_idx(coords)
            self.get_excessive_loops_idx(coords)
            self.get_nan_idx(coords)
            self.bad_indices = np.unique(self.bad_indices)
            coords = Preprocessor.remove_bad_indices(coords, self.bad_indices)
        return coords

    def preprocess_dt(self, dt_arr):
        """Just filters out the indices that were filtered in
        `preprocess_coords`
        """
        if self.FILTER_FLAG:
            print(f'Number of bad indices: {len(self.bad_indices)}')
            dt_arr = Preprocessor.remove_bad_indices(dt_arr, self.bad_indices)
        return dt_arr

    def combine_coords_dt(self, coords, dt_arr):
        return np.dstack([coords, dt_arr])        

    def preprocess(self, paths, dt_arr, dest=[1, 1]):
        """Main preprocessing method. Preprocesses and saves if specified.
        """
        coords = self.preprocess_coords(paths, dest=dest)
        dt_arr = self.preprocess_dt(dt_arr)

        if self.COMBINE_COORDS_DT_FLAG:
            coords_dt = self.combine_coords_dt(coords, dt_arr)

            if isinstance(self.save_dir, str):
                save_path = os.path.join(self.save_dir, 'coords_dt.npy')
                print(f'Saving: {save_path}')
                np.save(save_path, coords_dt)

            return coords_dt
        else:
            if isinstance(self.save_dir, str):
                save_path = os.path.join(self.save_dir, 'coords.npy')
                print(f'Saving: {save_path}')
                np.save(save_path, coords)

                save_path = os.path.join(self.save_dir, 'dt_arr.npy')
                print(f'Saving: {save_path}')
                np.save(save_path, dt_arr)

            return (coords, dt_arr)


def mode(x):
    values, counts = np.unique(x, return_counts=True)
    m = counts.argmax()
    return values[m], counts[m]


def scale_to_dest(path, dest=[1, 1], verbose=0):
    """Scales a single path to have a specified destination.

    Args:
        path (np.ndarray): a path with shape (100, 2 (or 3))
        verbose (boolean/int): whether or not to print the scale factors
    Returns:
        np.ndarray with the same shape as `path` but with the destination, dest
    """
    x_scale_factor = dest[0] / path[-1][0]
    y_scale_factor = dest[1] / path[-1][1]

    if verbose:
        print(f'X Scale Factor: {x_scale_factor}')
        print(f'Y Scale Factor: {y_scale_factor}')

    path[:, 0] = path[:, 0] * x_scale_factor
    path[:, 1] = path[:, 1] * y_scale_factor

    return path


def scale_coords_uniform_dest(coords, dest=[1, 1]):
    scaled = np.zeros(coords.shape)

    for i, arr in enumerate(coords):
        scaled[i] = scale_to_dest(arr, dest=dest, verbose=0)
    return scaled
