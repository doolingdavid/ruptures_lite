"""Tests for the categorical/numeric preprocessing layer and detect() wrapper.

These have no ruptures equivalent; they check that planted regime switches in a
mixed-dtype DataFrame are recovered.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import ruptures_lite as rpt
from ruptures_lite.preprocessing import FrameEncoder, prepare_signal


def _planted_frame(n=300, seed=0):
    """A frame with a categorical state flip and a numeric mean shift at row 150."""
    rng = np.random.RandomState(seed)
    state = np.array(["A"] * 150 + ["B"] * 150)
    temp = np.concatenate([rng.normal(0, 0.3, 150), rng.normal(4, 0.3, 150)])
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({"state": state, "temp": temp}, index=idx)


def test_frame_encoder_shapes_and_provenance():
    df = _planted_frame()
    enc = FrameEncoder(scale=True)
    X = enc.fit_transform(df)
    assert X.dtype == np.float64 and X.flags["C_CONTIGUOUS"]
    # one numeric column + 2 one-hot levels for "state"
    assert X.shape == (300, 3)
    assert "temp" in enc.columns_
    assert "state=A" in enc.columns_ and "state=B" in enc.columns_
    # numeric standardized: ~zero mean, ~unit std
    j = enc.columns_.index("temp")
    assert abs(X[:, j].mean()) < 1e-9
    assert abs(X[:, j].std() - 1.0) < 1e-9


def test_drop_first_and_ordinal():
    df = _planted_frame()
    enc = FrameEncoder(drop_first=True)
    X = enc.fit_transform(df)
    assert X.shape == (300, 2)  # one level dropped
    enc2 = FrameEncoder(encoding="ordinal")
    X2 = enc2.fit_transform(df)
    assert X2.shape == (300, 2)  # state collapsed to a single ordinal column


def test_unseen_category_maps_to_zero_block():
    df = _planted_frame()
    enc = FrameEncoder().fit(df)
    df2 = df.copy()
    df2.loc[df2.index[0], "state"] = "C"  # unseen level
    X = enc.transform(df2)
    a = enc.columns_.index("state=A")
    b = enc.columns_.index("state=B")
    assert X[0, a] == 0.0 and X[0, b] == 0.0


def test_nan_policy():
    df = _planted_frame()
    df.loc[df.index[10], "temp"] = np.nan
    with pytest.raises(ValueError):
        FrameEncoder(nan_policy="raise").fit_transform(df)
    X = FrameEncoder(nan_policy="fill").fit_transform(df)
    assert np.isfinite(X).all()


def test_prepare_signal_array_passthrough():
    arr = np.arange(20.0).reshape(-1, 1)
    out = prepare_signal(arr)
    assert out.shape == (20, 1) and out.dtype == np.float64
    out1d = prepare_signal(np.arange(20.0))
    assert out1d.shape == (20, 1)


def test_detect_recovers_planted_changepoint():
    warnings.simplefilter("ignore")
    df = _planted_frame()
    res = rpt.detect(df, method="dynp", model="l2", n_bkps=1)
    assert len(res.breakpoints) == 1
    assert abs(res.breakpoints[0] - 150) <= 3
    # timestamp returned because of the DatetimeIndex
    assert len(res.timestamps) == 1
    assert isinstance(res.timestamps[0], pd.Timestamp)
    # per-segment summary present
    assert res.segments is not None and len(res.segments) == 2


def test_detect_default_penalty_path():
    warnings.simplefilter("ignore")
    df = _planted_frame()
    res = rpt.detect(df, method="pelt", model="l2")  # no pen/n_bkps -> BIC default
    assert len(res.breakpoints) >= 1
    assert any(abs(b - 150) <= 5 for b in res.breakpoints)
