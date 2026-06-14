r"""Change detection with a Mahalanobis-type metric. Ported from ruptures (BSD-2-Clause)."""

from __future__ import annotations

import numpy as np
from numpy.linalg import inv

from ruptures_lite.base import BaseCost
from ruptures_lite.exceptions import NotEnoughPoints


class CostMl(BaseCost):
    r"""Mahalanobis-type cost function."""

    model = "mahalanobis"

    def __init__(self, metric=None):
        """Create a new instance.

        Args:
            metric (ndarray, optional): PSD matrix that defines a
                Mahalanobis-type pseudo distance. If None, defaults to the
                Mahalanobis matrix. Shape (n_features, n_features).
        """
        self.metric = metric  # metric matrix
        self.has_custom_metric = False if self.metric is None else True
        self.gram = None
        self.min_size = 2

    def fit(self, signal):
        """Set parameters of the instance."""
        s_ = signal.reshape(-1, 1) if signal.ndim == 1 else signal

        # fit a Mahalanobis metric if self.has_custom_metric is False
        if self.has_custom_metric is False:
            covar = np.cov(s_.T)
            self.metric = inv(covar.reshape(1, 1) if covar.size == 1 else covar)

        self.gram = s_.dot(self.metric).dot(s_.T)
        self.signal = s_
        return self

    def error(self, start, end):
        """Return the approximation cost on the segment [start:end].

        Raises:
            NotEnoughPoints: when the segment is too short (less than `min_size`).
        """
        if end - start < self.min_size:
            raise NotEnoughPoints
        sub_gram = self.gram[start:end, start:end]
        val = np.diagonal(sub_gram).sum()
        val -= sub_gram.sum() / (end - start)
        return val
