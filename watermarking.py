""" Watermarking algorithms. """

import numpy as np
from numpy.linalg import svd
from scipy.linalg import schur


class SignalWatermarking:
    """ Base class. """
    def __init__(self, watermark, transform=None):
        self.watermark = watermark
        self.transform = transform
        self._precompute()

    def _precompute(self):
        """ Precomputations needed before embedding the watermark. """
        pass

    def embed_watermark(self, signal):
        """ Embedding the watermark into the signal. """
        pass

    def extract_watermark(self, signal_w, public_data):
        """ Extracting the watermark from the signal. """
        pass


class SVDWatermarking(SignalWatermarking):
    def __init__(self, watermark, scale_svs=True):
        super().__init__(watermark)
        self.scale_svs = scale_svs

    def _precompute(self):
        """ Precomputations needed before embedding the watermark. """
        self._U_w, self._S_w, self._Vt_w = svd(self.watermark, full_matrices=False)

    def embed_watermark(self, signal, k=0.1):
        """ Embedding the watermark into the signal. """
        assert len(signal.shape) == 2

        # SVD decomposition of the signal
        U, S, Vt = svd(signal, full_matrices=False)

        # embedding the singular values of the watermark matrix
        # into the singular values of the signal matrix
        scale_factor = S[0] / self._S_w[0] if self.scale_svs else 1.0
        D = S + k * self._S_w * scale_factor

        # computing the signal with the watermark
        signal_w = U @ np.diag(D) @ Vt

        return signal_w, S

    def extract_watermark(self, signal_w, S, k=0.1):
        """ Extracting the watermark from the signal. """
        # SVD decomposition of the signal with the watermark
        U, D, Vt = svd(signal_w, full_matrices=False)

        # computing the singular values of the watermark
        scale_factor = S[0] / self._S_w[0] if self.scale_svs else 1.0
        S_w = (D - S) / k / scale_factor

        # building the watermark
        watermark = self._U_w @ np.diag(S_w) @ self._Vt_w

        return watermark


class SchurWatermarking(SignalWatermarking):
    def __init__(self, watermark):
        super().__init__(watermark)

    def _precompute(self):
        """ Precomputations needed before embedding the watermark. """
        self._T_w, self._U_w = schur(self.watermark)

    def embed_watermark(self, signal, k=0.1):
        """ Embedding the watermark into the signal. """
        assert len(signal.shape) == 2

        # embedding the singular values of the watermark matrix
        # into the singular values of the signal matrix
        signal_w = signal + k * self._T_w

        return signal_w, signal

    def extract_watermark(self, signal_w, T, k=0.1):
        """ Extracting the watermark from the signal. """
        # computing the singular values of the watermark
        T_w = (signal_w - T) / k

        # building the watermark
        watermark = self._U_w @ T_w @ self._U_w.T

        return watermark


def watermark_single(segment, model, k=0.01):
    segment_w, pb_data = model.embed_watermark(segment, k=k)

    return segment_w, pb_data


def extract_from_single(segment_w, model, pb_data, k=0.01):
    watermark = model.extract_watermark(segment_w, pb_data, k=k)

    return watermark


def watermark_many(segments, model, k=0.01):
    result = [watermark_single(s, model, k) for s in segments]
    segments_w, wp_data = zip(*result)

    return segments_w, wp_data


def extract_from_many(segments_w, model, wp_data, k=0.01):
    watermarks = [extract_from_single(s_w, model, wp, k)
                  for s_w, wp in zip(segments_w, wp_data)]

    return watermarks
