from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
import tensorflow.keras.backend as K


def soft_binary_accuracy(y_true, y_pred, threshold=0.5):
    """Calculates how often predictions matches binary labels.
    Standalone usage:
    >>> y_true = [[1], [1], [0], [0]]
    >>> y_pred = [[1], [1], [0], [0]]
    >>> m = tf.keras.metrics.binary_accuracy(y_true, y_pred)
    >>> assert m.shape == (4,)
    >>> m.numpy()
    array([1., 1., 1., 1.], dtype=float32)
    Args:
      y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
      y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
      threshold: (Optional) Float representing the threshold for deciding
        whether prediction values are 1 or 0.
    Returns:
      Binary accuracy values. shape = `[batch_size, d0, .. dN-1]`
    """
    y_pred = ops.convert_to_tensor_v2(y_pred)
    threshold = math_ops.cast(threshold, y_pred.dtype)
    y_pred = math_ops.cast(y_pred > threshold, y_pred.dtype) * 0.9
    return K.mean(math_ops.equal(y_true, y_pred), axis=-1)


if __name__ == '__main__':
    import numpy as np
    soft_labels = np.full((25, 1), 0.9)
    pred = np.full((25, 1), 0.7)

    acc = soft_binary_accuracy(soft_labels, pred)
    print(f'Accuracy: {acc}')
