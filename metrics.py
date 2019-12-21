""" Evaluation metrics. """

import numpy as np


def snr(original, target):
    """ Signal-to-noise ratio. """
    return 10 * np.log10(np.sum(original ** 2) / np.sum((original - target) ** 2))


def nc(w1, w2):
    """ Normalized correlation coefficient. """
    return (w1 * w2).sum() / np.sqrt((w1 ** 2).sum()) / np.sqrt((w2 ** 2).sum())
