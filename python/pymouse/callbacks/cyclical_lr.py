import tensorflow as tf
import numpy as np


class SGDRScheduler(tf.keras.callbacks.Callback):
    '''Cosine annealing learning rate scheduler with periodic restarts.

    # Usage
        ```python
            schedule = SGDRScheduler(min_lr=1e-5,
                                     max_lr=1e-2,
                                     lr_decay=0.9,
                                     cycle_length=5,
                                     mult_factor=1.5)
            model.fit(X_train, Y_train, epochs=100, callbacks=[schedule])
        ```

    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        lr_decay: Reduce the max_lr after the completion of each cycle.
                  Ex. To reduce the max_lr by 20% after each cycle, set this value to 0.8.
        cycle_length: Initial number of epochs in a cycle.
        mult_factor: Scale epochs_to_restart after each full cycle completion.

    # References
        Original paper: http://arxiv.org/abs/1608.03983
    '''
    def __init__(self,
                 min_lr,
                 max_lr,
                 lr_decay=1,
                 cycle_length=10,
                 mult_factor=2):

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay

        self.batch_since_restart = 0
        self.next_restart = cycle_length

        self.cycle_length = cycle_length
        self.mult_factor = mult_factor

        self.history = {}

    def clr(self):
        '''Calculate the learning rate.'''
        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
        return lr

    def on_train_begin(self, logs={}):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        self.steps_per_epoch = self.params['steps'] if self.params['steps'] is not None else round(self.params['samples'] / self.params['batch_size'])
        logs = logs or {}
        tf.keras.backend.set_value(self.model.optimizer.lr, self.max_lr)

    def on_batch_end(self, batch, logs={}):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.history.setdefault('lr', []).append(tf.keras.backend.get_value(self.model.optimizer.lr))
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.batch_since_restart += 1
        tf.keras.backend.set_value(self.model.optimizer.lr, self.clr())

    def on_epoch_end(self, epoch, logs={}):
        '''Check for end of current cycle, apply restarts when necessary.'''
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
            self.next_restart += self.cycle_length
            self.max_lr *= self.lr_decay
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs={}):
        '''Set weights to the values from the end of the most recent cycle for best performance.'''
        self.model.set_weights(self.best_weights)



def find_clr_params_checkpoint(num_epochs, lr_range, lr_decay=0.9, cycle_length=3, mult_factor=1.5):
    """Determines the new paramters for SGDRScheduler after loading a model from a checkpoint for
    training.
    
    Note: only max_lr is decayed, so it's misleading
        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
        ...
        self.max_lr *= self.lr_decay

    Args:
        num_epochs (int): Number of epochs elapsed
        lr_range (tuple/list): learning rate range for CLR
        lr_decay (float):
        cycle_length (int):
        mult_factor (float):

    Returns:
        lr_range: the new learning rate range
        final_cycle_length: The cycle length to input to the callback after
            loading the checkpoint
    """
    min_lr = lr_range[0]
    cycle_lengths = []
    num_cycles = 0
    print(f'Original cycle length: {cycle_length}')
#     iter_epoch = int(num_epochs) # makes copy
    while sum(cycle_lengths) < num_epochs:
#         iter_epoch = iter_epoch-cycle_length
        if num_cycles != 0:
            # end of cycle
            cycle_length = np.ceil(cycle_length * mult_factor)
        cycle_lengths.append(cycle_length)
        num_cycles += 1

    assert len(cycle_lengths) == num_cycles
    lr_range = list(map(lambda x: x**(lr_decay*num_cycles), lr_range))
    # Only max_lr is decayed
    lr_range = (min_lr, lr_range[-1])
    print("cycle lengths: ", cycle_lengths)
    print(f'Number of cycles: {num_cycles},\nFinal cycle length: {cycle_lengths[-1]}, \nlr_range: {lr_range}')
    return lr_range, cycle_lengths[-1]
