""" Signal preprocessing tools. """

import numpy as np


def signal_to_2d(signal, height):
    """
    Reshaping a 1D signal of shape (length, )
    to a 2D signal with shape (height, ).
    Padding the signal with zeros if needed.
    """
    length = signal.shape[0]

    if length % height > 0:
        pad_size = height - length % height
        signal = np.pad(signal, (0, pad_size))

    return signal.reshape((-1, height)).T


def signal_to_1d(signal, crop_length=None):
    """
    Reshaping a 2D signal of shape (height, length)
    to a 1D signal with shape (height x length,).
    Cropping the signal into a given length if `crop_length` is provided.
    """
    signal = signal.T.reshape((-1,))
    if crop_length is not None:
        signal = signal[:crop_length]

    return signal


def split_into_frames(signal, width):
    """
    Splitting the signal into frame of given `width`.
    """
    length = signal.shape[-1]

    if length % width > 0:
        pad_size = width - length % width
        signal = np.pad(signal, ((0, 0), (0, pad_size)))

    frames = np.split(signal, signal.shape[1] // width, axis=1)

    return frames


def combine_frames(frames):
    """
    Combines frames into a complete signal.
    """
    return np.hstack(frames)
