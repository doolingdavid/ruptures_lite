"""Detection parity: identical breakpoints vs ruptures across the full matrix of
estimators x cost models x parameters (n_bkps and pen)."""

import warnings

import pytest

import ruptures_lite as rpt
from conftest import HAS_RUPTURES, requires_ruptures

if HAS_RUPTURES:
    import ruptures as gt

MODELS = ["l1", "l2", "normal", "rbf", "cosine", "mahalanobis", "rank"]
SIG_KEYS = ["1d", "3d", "normal2d", "wavy"]


def _run(mod, estimator_name, signal, build_kwargs, predict_kwargs):
    cls = getattr(mod, estimator_name)
    try:
        algo = cls(**build_kwargs).fit(signal)
        return list(algo.predict(**predict_kwargs))
    except Exception as exc:  # parity also covers identical error behaviour
        return ("ERR", type(exc).__name__)


@requires_ruptures
@pytest.mark.parametrize("sig_key", SIG_KEYS)
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("jump", [1, 5])
@pytest.mark.parametrize("min_size", [2, 5])
def test_dynp(signals, sig_key, model, jump, min_size):
    warnings.simplefilter("ignore")
    sig = signals[sig_key]
    bk = dict(model=model, min_size=min_size, jump=jump)
    assert _run(rpt, "Dynp", sig, bk, dict(n_bkps=3)) == _run(
        gt, "Dynp", sig, bk, dict(n_bkps=3)
    )


@requires_ruptures
@pytest.mark.parametrize("sig_key", SIG_KEYS)
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("jump", [1, 5])
@pytest.mark.parametrize("pen", [10.0, 50.0])
def test_pelt(signals, sig_key, model, jump, pen):
    warnings.simplefilter("ignore")
    sig = signals[sig_key]
    bk = dict(model=model, min_size=2, jump=jump)
    assert _run(rpt, "Pelt", sig, bk, dict(pen=pen)) == _run(
        gt, "Pelt", sig, bk, dict(pen=pen)
    )


@requires_ruptures
@pytest.mark.parametrize("estimator", ["Binseg", "BottomUp"])
@pytest.mark.parametrize("sig_key", SIG_KEYS)
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("n_bkps", [2, 4])
def test_greedy(signals, estimator, sig_key, model, n_bkps):
    warnings.simplefilter("ignore")
    sig = signals[sig_key]
    bk = dict(model=model, min_size=2, jump=5)
    assert _run(rpt, estimator, sig, bk, dict(n_bkps=n_bkps)) == _run(
        gt, estimator, sig, bk, dict(n_bkps=n_bkps)
    )


@requires_ruptures
@pytest.mark.parametrize("sig_key", SIG_KEYS)
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("n_bkps", [2, 4])
def test_window(signals, sig_key, model, n_bkps):
    warnings.simplefilter("ignore")
    sig = signals[sig_key]
    bk = dict(width=40, model=model, min_size=2, jump=5)
    assert _run(rpt, "Window", sig, bk, dict(n_bkps=n_bkps)) == _run(
        gt, "Window", sig, bk, dict(n_bkps=n_bkps)
    )


@requires_ruptures
@pytest.mark.parametrize("sig_key", ["1d", "3d", "normal2d"])
@pytest.mark.parametrize("kernel", ["linear", "rbf", "cosine"])
@pytest.mark.parametrize("min_size", [1, 2, 5])
@pytest.mark.parametrize("n_bkps", [2, 4])
def test_kernelcpd_nbkps(signals, sig_key, kernel, min_size, n_bkps):
    warnings.simplefilter("ignore")
    sig = signals[sig_key]
    bk = dict(kernel=kernel, min_size=min_size)
    assert _run(rpt, "KernelCPD", sig, bk, dict(n_bkps=n_bkps)) == _run(
        gt, "KernelCPD", sig, bk, dict(n_bkps=n_bkps)
    )


@requires_ruptures
@pytest.mark.parametrize("sig_key", ["1d", "3d", "normal2d"])
@pytest.mark.parametrize("kernel", ["linear", "rbf", "cosine"])
@pytest.mark.parametrize("pen", [5.0, 20.0, 100.0])
def test_kernelcpd_pelt(signals, sig_key, kernel, pen):
    warnings.simplefilter("ignore")
    sig = signals[sig_key]
    bk = dict(kernel=kernel, min_size=2)
    assert _run(rpt, "KernelCPD", sig, bk, dict(pen=pen)) == _run(
        gt, "KernelCPD", sig, bk, dict(pen=pen)
    )


@requires_ruptures
@pytest.mark.parametrize("jump", [1, 5])
def test_linear_cost_detection(jump):
    warnings.simplefilter("ignore")
    sig, _ = rpt.pw_linear(180, n_features=2, n_bkps=3, noise_std=1, seed=9)
    bk = dict(model="linear", min_size=2, jump=jump)
    assert _run(rpt, "Dynp", sig, bk, dict(n_bkps=3)) == _run(
        gt, "Dynp", sig, bk, dict(n_bkps=3)
    )


@requires_ruptures
@pytest.mark.parametrize("jump", [1, 5])
def test_ar_cost_detection(jump):
    warnings.simplefilter("ignore")
    sig = rpt.pw_wavy(180, n_bkps=3, noise_std=0.5, seed=11)[0]
    bk = dict(model="ar", min_size=5, jump=jump)
    assert _run(rpt, "Pelt", sig, bk, dict(pen=50.0)) == _run(
        gt, "Pelt", sig, bk, dict(pen=50.0)
    )
