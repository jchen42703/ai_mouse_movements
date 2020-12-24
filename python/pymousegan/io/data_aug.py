import numpy as np


def scale_translate_v2(real_seqs):
    """
    Procedures:
    - multiplicatively scaled

    Args:
        real_seqs (np.ndarray): batch of real sequences with shape
            (batch_size, seq_shape)
    Returns:
        augmented sequences
    """
    scale_factor = np.random.uniform(-1, 1)
    diff = 1 - np.abs(scale_factor)
    translateX = np.random.choice([0, np.random.uniform(-diff, diff)])
    translateY = np.random.choice([0, np.random.uniform(-diff, diff)])
    scaled = real_seqs * np.array([scale_factor, scale_factor, 1])
    return np.add(scaled, np.array([translateX, translateY, 0]))


def scale_translate_v3(real_seqs):
    """Version 3 of scale translate.

    Additions to version 2:
        Scaling now includes all quadrants instead of just the first and third.

    Procedure:
        * Still square scaling for X and Y except for the direction.
        * Translations are now more flexible to handle all quadrants.
    """
    scale_factor = np.random.uniform(0, 1)
    diff = 1 - scale_factor
    scaleX = scale_factor * np.random.choice([-1, 1])
    scaleY = scale_factor * np.random.choice([-1, 1])

    # Handles different quadrants (QI-QIV)
    if scaleX > 0:
        translateX = np.random.uniform(-1, diff)
    else:
        translateX = np.random.uniform(-diff, 1)

    if scaleY > 0:
        translateY = np.random.uniform(-1, diff)
    else:
        translateY = np.random.uniform(-diff, 1)

    scaled = real_seqs * np.array([scaleX, scaleY, 1])
    translated = scaled + np.array([translateX, translateY, 0])
    return translated


def reflect(real_seqs):
    """Reflect the path across x/y axes.
    """
    reflectX = np.random.choice([-1, 1])
    reflectY = np.random.choice([-1, 1])
    reflected = real_seqs * np.array([reflectX, reflectY, 1])
    return reflected


def unique_batch_reflect(real_seqs: np.array):
    """Reflects but makes each path in the sequence have unique reflection
    factor. Slower than `reflect` but returns more diverse paths in a batch.
    """
    reflected = np.zeros(real_seqs.shape)
    for i, path in enumerate(real_seqs):
        reflectX = np.random.choice([-1, 1])
        reflectY = np.random.choice([-1, 1])
        reflected[i] = path * np.array([reflectX, reflectY, 1])
    return reflected
