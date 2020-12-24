import numpy as np


def sine_wave(seq_length=30, num_samples=28*5*100, num_signals=1,
              freq_low=1, freq_high=5, amplitude_low=0.1, amplitude_high=0.9,
              random_seed=None, **kwargs):

    ix = np.arange(seq_length) + 1
    samples = []
    for i in range(num_samples):
        signals = []
        for i in range(num_signals):
            f = np.random.uniform(low=freq_high, high=freq_low)     # frequency
            A = np.random.uniform(low=amplitude_high,
                                  high=amplitude_low)        # amplitude
            # offset
            offset = np.random.uniform(low=-np.pi, high=np.pi)
            signals.append(A*np.sin(2*np.pi*f*ix/float(seq_length) + offset))
        samples.append(np.array(signals).T)
    # the shape of the samples is num_samples x seq_length x num_signals
    samples = np.array(samples)
    return samples
