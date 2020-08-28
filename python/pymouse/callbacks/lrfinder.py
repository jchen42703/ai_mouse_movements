import tensorflow as tf
import numpy as np


class LRFinder(tf.keras.callbacks.Callback):
  def __init__(self, start_lr, end_lr):
    super().__init__()
    self.start_lr = start_lr
    self.end_lr = end_lr

  def on_train_begin(self, logs={}):
    self.lrs = []
    self.losses = []
    tf.keras.backend.set_value(self.model.optimizer.lr, self.start_lr)

    n_steps = self.params['steps'] if self.params['steps'] is not None else round(self.params['samples'] / self.params['batch_size'])
    n_steps *= self.params['epochs']
    self.by = (self.end_lr - self.start_lr) / n_steps


  def on_batch_end(self, batch, logs={}):
    lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
    self.lrs.append(lr)
    self.losses.append(logs.get('loss'))
    lr += self.by
    tf.keras.backend.set_value(self.model.optimizer.lr, lr)


def find_lr_range(lrfinder_cb):
    """Finds the min/max desirable lr from the LRFinder callback.

    Args:
        lrfinder_cb (tf.keras.Callback):

    Returns:
        (lr_min, lr_max)
    """
    def smooth(y, box_pts):
      """smoothes an array by taking the average of the `box_pts` point around each point"""
      box = np.ones(box_pts)/box_pts
      y_smooth = np.convolve(y, box, mode='same')
      return y_smooth

    smoothed_losses = smooth(lrfinder_cb.losses, 20)

    # Sub-sample the (smoothed) losses between the point where it reaches its max and the point where it reaches its min
    min_ = np.argmin(smoothed_losses)
    max_ = np.argmax(smoothed_losses)
    smoothed_losses_ = smoothed_losses[min_: max_]

    smoothed_diffs = smooth(np.diff(smoothed_losses), 20)
    min_ = np.argmax(smoothed_diffs <= 0)  # where the (smoothed) loss starts to decrease
    max_ = np.argmax(smoothed_diffs >= 0)  # where the (smoothed) loss restarts to increase
    max_ = max_ if max_ > 0 else smoothed_diffs.shape[0]  # because max_ == 0 if it never restarts to increase

    smoothed_losses_ = smoothed_losses[min_: max_]  # restrain the window to the min_, max_ interval
    # Take min and max loss in this restrained window
    min_smoothed_loss_ = min(smoothed_losses_[:-1])
    max_smoothed_loss_ = max(smoothed_losses_[:-1])
    delta = max_smoothed_loss_ - min_smoothed_loss_

    lr_arg_max = np.argmax(smoothed_losses_ <= min_smoothed_loss_ + .05 * delta)
    lr_arg_min = np.argmax(smoothed_losses_ <= min_smoothed_loss_ + .5 * delta)

    lr_arg_min += min_
    lr_arg_max += min_

    lrs = lrfinder_cb.lrs[lr_arg_min: lr_arg_max]
    lr_min, lr_max = min(lrs), max(lrs)
    return (lr_min, lr_max)
