r"""Efficient kernel change point detection (dynamic programming / PELT).

Ported from ``ruptures.detection.kernelcpd`` (BSD-2-Clause). The compiled
``ekcpd`` engine is replaced by the pure-numpy port in
:mod:`ruptures_lite.detection._kernel_dp`.
"""

from __future__ import annotations

import numpy as np

from ruptures_lite.base import BaseEstimator
from ruptures_lite.costs import cost_factory
from ruptures_lite.utils import from_path_matrix_to_bkps_list, sanity_check
from ruptures_lite.exceptions import BadSegmentationParameters

from ._kernel_dp import ekcpd_compute, ekcpd_pelt_compute


class KernelCPD(BaseEstimator):
    """Optimal kernel change point detection (dynamic programming or PELT).

    For the special case where the cost function derives from a kernel.
    """

    def __init__(self, kernel="linear", min_size=2, jump=1, params=None):
        r"""Create a KernelCPD instance.

        Available kernels:

        - ``linear``: :math:`k(x,y) = x^T y`.
        - ``rbf``: :math:`k(x, y) = \exp(-\gamma \|x-y\|^2)` where
          :math:`\gamma>0` (``gamma``) is a user-defined parameter.
        - ``cosine``: :math:`k(x,y) = (x^T y)/(\|x\|\|y\|)`.

        Args:
            kernel (str, optional): name of the kernel, ["linear", "rbf", "cosine"]
            min_size (int, optional): minimum segment length.
            jump (int, optional): not considered, set to 1.
            params (dict, optional): a dictionary of parameters for the kernel.

        Raises:
            AssertionError: if the kernel is not implemented.
        """
        self.kernel_name = kernel
        err_msg = "Kernel not found: {}.".format(self.kernel_name)
        assert self.kernel_name in ["linear", "rbf", "cosine"], err_msg
        self.model_name = "l2" if self.kernel_name == "linear" else self.kernel_name
        self.params = params
        # load the associated cost function (used for parameters, e.g. gamma)
        if self.params is None:
            self.cost = cost_factory(model=self.model_name)
        else:
            self.cost = cost_factory(model=self.model_name, **self.params)
        self.min_size = max(min_size, self.cost.min_size)

        self.jump = 1  # set to 1
        self.n_samples = None
        self.segmentations_dict = dict()  # {n_bkps: bkps_list}
        # map kernel name to the engine's kernel identifier
        self._engine_kernel = {
            "linear": "linear",
            "rbf": "gaussian",
            "cosine": "cosine",
        }[self.kernel_name]

    def fit(self, signal):
        """Update parameters (no computation in this function)."""
        self.segmentations_dict = dict()
        self.cost.fit(signal.astype(np.double))
        self.n_samples = signal.shape[0]
        return self

    def predict(self, n_bkps=None, pen=None):
        """Return the optimal breakpoints. Must be called after the fit method.

        Args:
            n_bkps (int, optional): number of change points. Defaults to None.
            pen (float, optional): penalty value (>0). Defaults to None. Not
                considered if ``n_bkps`` is not None.

        Raises:
            AssertionError: if ``pen`` or ``n_bkps`` is not strictly positive.
            BadSegmentationParameters: in case of impossible segmentation configuration

        Returns:
            list: sorted list of breakpoints
        """
        # Our KernelCPD with Pelt implies at least one change point.
        if not sanity_check(
            n_samples=self.cost.signal.shape[0],
            n_bkps=1 if n_bkps is None else n_bkps,
            jump=self.jump,
            min_size=self.min_size,
        ):
            raise BadSegmentationParameters

        gamma = getattr(self.cost, "gamma", None)

        # dynamic programming if the user passed a number of change points
        if n_bkps is not None:
            n_bkps = int(n_bkps)
            err_msg = "The number of changes must be positive: {}".format(n_bkps)
            assert n_bkps > 0, err_msg
            if n_bkps in self.segmentations_dict:
                return self.segmentations_dict[n_bkps]
            path_matrix_flat = ekcpd_compute(
                self.cost.signal, n_bkps, self.min_size, self._engine_kernel, gamma
            )
            # from the path matrix, get all segmentations for k=1,...,n_bkps
            for k in range(1, n_bkps + 1):
                self.segmentations_dict[k] = from_path_matrix_to_bkps_list(
                    path_matrix_flat, k, self.n_samples, n_bkps, self.jump
                )
            return self.segmentations_dict[n_bkps]

        # PELT if the user passed a penalty
        if pen is not None:
            assert pen > 0, "The penalty must be positive: {}".format(pen)
            path_matrix = ekcpd_pelt_compute(
                self.cost.signal, pen, self.min_size, self._engine_kernel, gamma
            )
            my_bkps = list()
            ind = self.n_samples
            while ind > 0:
                my_bkps.append(ind)
                ind = int(path_matrix[ind])
            return my_bkps[::-1]

    def fit_predict(self, signal, n_bkps=None, pen=None):
        """Fit to the signal and return the optimal breakpoints."""
        self.fit(signal)
        return self.predict(n_bkps=n_bkps, pen=pen)
