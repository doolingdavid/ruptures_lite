"""Cost-function parity: ``CostX.error(s, e)`` must match ruptures to ~1e-9."""

import warnings

import numpy as np
import pytest

import ruptures_lite as rpt
from conftest import HAS_RUPTURES, requires_ruptures, _scalar

if HAS_RUPTURES:
    import ruptures as gt

TOL = 1e-9

# costs that consume an ordinary (n_samples, n_features) signal
PLAIN_MODELS = ["l1", "l2", "normal", "rbf", "cosine", "mahalanobis", "rank", "clinear"]
SEGMENTS = [(0, 40), (10, 60), (30, 120), (5, 15), (0, 180)]


@requires_ruptures
@pytest.mark.parametrize("model", PLAIN_MODELS)
def test_plain_cost_parity(model):
    warnings.simplefilter("ignore")
    rng = np.random.RandomState(0)
    signal = rng.randn(180, 3)
    cm = rpt.costs.cost_factory(model).fit(signal)
    cg = gt.costs.cost_factory(model).fit(signal)
    for (s, e) in SEGMENTS:
        if e - s < cm.min_size:
            continue
        assert abs(_scalar(cm.error(s, e)) - _scalar(cg.error(s, e))) < TOL


@requires_ruptures
def test_linear_cost_parity():
    warnings.simplefilter("ignore")
    signal, _ = rpt.pw_linear(180, n_features=2, n_bkps=3, noise_std=1, seed=1)
    cm = rpt.costs.CostLinear().fit(signal)
    cg = gt.costs.CostLinear().fit(signal)
    for (s, e) in SEGMENTS:
        assert abs(_scalar(cm.error(s, e)) - _scalar(cg.error(s, e))) < TOL


@requires_ruptures
@pytest.mark.parametrize("order", [2, 4])
def test_ar_cost_parity(order):
    warnings.simplefilter("ignore")
    signal = rpt.pw_wavy(180, n_bkps=3, noise_std=0.5, seed=2)[0]
    cm = rpt.costs.CostAR(order=order).fit(signal)
    cg = gt.costs.CostAR(order=order).fit(signal)
    for (s, e) in SEGMENTS:
        if e - s < cm.min_size:
            continue
        assert abs(_scalar(cm.error(s, e)) - _scalar(cg.error(s, e))) < TOL
