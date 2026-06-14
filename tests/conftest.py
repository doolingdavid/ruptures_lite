"""Shared pytest fixtures and ruptures availability gate.

Parity tests compare ``ruptures_lite`` to the real ``ruptures`` package. On the
locked-down target environment ``ruptures`` is not installed, so those tests are
skipped automatically (the preprocessing and compat tests still run).
"""

import warnings

import numpy as np
import pytest

import ruptures_lite as rpt

try:
    import ruptures as gt

    HAS_RUPTURES = True
except ImportError:  # pragma: no cover
    gt = None
    HAS_RUPTURES = False

requires_ruptures = pytest.mark.skipif(
    not HAS_RUPTURES, reason="reference `ruptures` package not installed"
)


def _scalar(x):
    return float(np.asarray(x).ravel()[0])


@pytest.fixture(scope="session")
def signals():
    """A spread of signal shapes (univariate, multivariate, 2D Gaussian, wavy)."""
    warnings.simplefilter("ignore")
    return {
        "1d": rpt.pw_constant(180, 1, n_bkps=3, noise_std=1, seed=3)[0],
        "3d": rpt.pw_constant(180, 3, n_bkps=4, noise_std=2, seed=7)[0],
        "normal2d": rpt.pw_normal(180, n_bkps=3, seed=2)[0],
        "wavy": rpt.pw_wavy(180, n_bkps=3, noise_std=0.3, seed=4)[0],
    }
