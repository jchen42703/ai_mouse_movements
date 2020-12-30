import matplotlib.pyplot as plt
import numpy


# Predict
def predict(model, inputs):
    values = numpy.array(inputs)
    v = model.predict(values.reshape(1, 2))
    return v


def get_paths_coords(paths):
    """
    Gets the X,Y coordinates for `paths`

    Args:
        paths (np.ndarray): with shape (num_paths, num_coords_per_path, 2)

    Returns:
        (X, Y): tuple of lists of the coordinates
    """
    X = []
    Y = []
    for path in paths:
        for point in path:
            X.append(point[0])
            Y.append(point[1])
    return (X, Y)


def plot_paths(paths):
    """Utility function to plot to an existing plt.figure.
    """
    for path in paths:
        X = []
        Y = []

        for point in path:
            X.append(point[0])
            Y.append(point[1])

        plt.plot(X, Y)
    return


def plot_single_label_pred_paths(model, label_path, pred_input,
                                 figsize=(10, 5)):
    """Plots single label, pred pair paths.

    Args:
        label_path (np.ndarray): array of path coordinates with
            shape (num_coords_per_path, 2)
        pred_input (np.ndarray): destination coord with shape (2,)
        figsize (tuple[int]): (width, height) of figure

    Returns:
        (fig, ax) of the plotted figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # plotting label
    X, Y = get_paths_coords(label_path[None])
    line, = ax.plot(X, Y, linestyle=':', color='blue')
    line.set_label('Label')

    # plotting pred
    pred_path = model.predict(pred_input[None])
    pred_X, pred_Y = get_paths_coords(pred_path)
    pred_line, = ax.plot(pred_X, pred_Y, linestyle='-', color='green')
    pred_line.set_label('Predicted')

    dest = ax.scatter(pred_input[0], pred_input[1], color='red')
    dest.set_label('Destination')

    ax.legend()
    plt.show()
    return (fig, ax)


def plot_label_pred_paths(model, label_paths, pred_inputs):
    """Plots multiple paths.

    Args:
        label_paths (np.ndarray): with shape
            (num_paths, num_coords_per_path, 2)
        pred_inputs (np.ndarray): with shape (num_paths, 2)
    """
    plt.figure(1)
    plt.title('user input')
    plot_paths(label_paths)

    plt.figure(2)
    # shape from (num_paths, 2) to (num_paths, 1, 2)
    pred_paths = model.predict(pred_inputs[:, None])
    plot_paths(pred_paths)
    plt.title('generated')
    return


def plot(model, history, train_inputs, train_paths):
    plt.figure(1)
    plt.title('user input')
    plot_paths(train_paths)

    plt.figure(2)
    size = train_inputs.__len__()
    for x in range(size):
        v = train_inputs[x]
        paths = predict(model, v)
        plot_paths(paths)

    plt.title('generated')

    plt.figure(3)
    plt.title('accuracy')
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['loss'])
    plt.ylabel('value')
    plt.xlabel('epoch')
    plt.legend(['acc', 'loss'], loc='upper left')
    plt.show()

    return


def extract_dts_from_coords_dt(coords_dt):
    """

    Args:
        coords_dt (np.ndarray): with shape (num_paths, path_count, 3)

    Returns:
        dt_list (list): nested list of path dts
    """
    # dt_list is a nested list of paths dts
    dt_list = []
    for path in coords_dt:
        path_dt = []
        # Note: path is an array of coords (path_count, 3)
        for coord in path:
            path_dt.append(coord[-1])
        dt_list.append(path_dt)
    return dt_list


def plot_dt_list(dt_sublist, num_rows=3, num_cols=3, figsize=(24, 24)):
    """Plots distributions of the time deltas.

    Args:
        dt_sublist (list[List[float]]): nested list of dts (sublist to plot)
    """
    num_plots = num_rows*num_cols
    assert len(dt_sublist) == num_plots, \
        'The length of the sublist must be the same as rows*cols'
    print(f'Plotting {num_plots} plots.')
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for row in range(num_rows):
        for col in range(num_cols):
            curr_path = row*col
            axes[row, col].hist(dt_sublist[curr_path])
    plt.show()
