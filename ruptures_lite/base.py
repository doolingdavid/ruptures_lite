r"""Base classes.

All estimators and cost functions are subclasses of
:class:`BaseEstimator` and :class:`BaseCost` respectively.

Ported from ``ruptures.base`` (BSD-2-Clause). See LICENSE.
"""

from __future__ import annotations

import abc

from ruptures_lite.utils import pairwise


class BaseEstimator(metaclass=abc.ABCMeta):
    """Base class for all change point detection estimators."""

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        """Call the segmentation algorithm."""
        pass

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        """Call the segmentation algorithm."""
        pass

    @abc.abstractmethod
    def fit_predict(self, *args, **kwargs):
        """Call the segmentation algorithm."""
        pass


class BaseCost(object, metaclass=abc.ABCMeta):
    """Base class for all segment cost classes."""

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        """Set the parameters of the cost function (e.g. the Gram matrix)."""
        pass

    @abc.abstractmethod
    def error(self, start, end):
        """Return the cost on segment [start:end]."""
        pass

    def sum_of_costs(self, bkps):
        """Return the sum of segment costs for the given segmentation.

        Args:
            bkps (list): list of change points. By convention,
                ``bkps[-1] == n_samples``.

        Returns:
            float: sum of costs
        """
        soc = sum(self.error(start, end) for start, end in pairwise([0] + bkps))
        return soc

    @property
    @abc.abstractmethod
    def model(self):
        pass
