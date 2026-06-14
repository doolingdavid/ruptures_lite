r"""High-level convenience wrapper around the estimators.

``detect`` takes a numpy array *or* a pandas DataFrame (mixed numeric/categorical
columns are encoded automatically via :mod:`ruptures_lite.preprocessing`), runs
the requested search method, and returns the breakpoints together with a tidy
per-segment summary. When the DataFrame has a ``DatetimeIndex``, the change-point
timestamps are returned as well.

This mirrors the ergonomics of the existing
``sandia_talk/src/changepoint.py::detect`` but generalises it to multivariate and
categorical inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

from ruptures_lite.detection import Binseg, BottomUp, Dynp, KernelCPD, Pelt, Window
from ruptures_lite.preprocessing import FrameEncoder, _is_dataframe

_ESTIMATORS = {
    "pelt": Pelt,
    "dynp": Dynp,
    "binseg": Binseg,
    "bottomup": BottomUp,
    "window": Window,
    "kernelcpd": KernelCPD,
}

# methods that accept a penalty in predict()
_PEN_METHODS = {"pelt", "binseg", "bottomup", "window", "kernelcpd"}
# methods that accept n_bkps in predict()
_NBKPS_METHODS = {"dynp", "binseg", "bottomup", "window", "kernelcpd"}


@dataclass
class DetectionResult:
    """Result of :func:`detect`."""

    bkps: list                      # ruptures-style, last element == n_samples
    breakpoints: list               # bkps[:-1] (interior change points)
    timestamps: list = field(default_factory=list)  # index values at breakpoints
    segments: object = None         # pandas DataFrame (or None)
    signal: object = None           # the encoded float64 matrix actually segmented
    encoder: object = None          # the fitted FrameEncoder (or None)
    estimator: object = None        # the fitted estimator


def default_penalty(signal):
    """A BIC-flavoured default penalty for ``model='l2'`` on a (scaled) signal.

    ``pen = n_dims * log(n_samples)`` assumes roughly unit-variance dimensions
    (true after standardization). Use an explicit ``pen`` for other models.
    """
    n_samples, n_dims = signal.shape
    return n_dims * log(n_samples)


def detect(
    data,
    method="pelt",
    model="l2",
    pen=None,
    n_bkps=None,
    epsilon=None,
    min_size=2,
    jump=5,
    width=100,
    params=None,
    custom_cost=None,
    kernel="linear",
    encode=True,
    scale=True,
    categorical=None,
    numeric=None,
    drop_first=False,
    encoding="onehot",
    nan_policy="raise",
    max_cardinality=None,
):
    """Detect change points in an array or DataFrame.

    Args:
        data: ndarray of shape (n_samples,) / (n_samples, n_features), or a
            pandas DataFrame (encoded automatically when ``encode`` is True).
        method: one of "pelt", "dynp", "binseg", "bottomup", "window", "kernelcpd".
        model: cost model for the non-kernel methods ("l2", "l1", "rbf", "normal",
            "linear", "ar", "mahalanobis", "rank", "cosine", "clinear").
        pen: penalty (for pelt/binseg/bottomup/window/kernelcpd). If neither
            ``pen`` nor ``n_bkps`` is given for a penalty method, a BIC default is
            used (and reported on the result).
        n_bkps: number of change points (for dynp/binseg/bottomup/window/kernelcpd).
        epsilon: reconstruction budget (binseg/bottomup/window).
        min_size, jump, width, params, custom_cost: passed to the estimator.
        kernel: kernel name when ``method='kernelcpd'`` ("linear"/"rbf"/"cosine").
        encode: when ``data`` is a DataFrame, run the categorical/numeric encoder.
        scale, categorical, numeric, drop_first, encoding, nan_policy,
            max_cardinality: forwarded to :class:`FrameEncoder`.

    Returns:
        DetectionResult
    """
    if method not in _ESTIMATORS:
        raise ValueError(
            "Unknown method {!r}; choose from {}".format(
                method, sorted(_ESTIMATORS)
            )
        )

    # ---- build the numeric signal --------------------------------------
    encoder = None
    index = None
    if _is_dataframe(data):
        index = data.index
        if encode:
            encoder = FrameEncoder(
                categorical=categorical,
                numeric=numeric,
                scale=scale,
                encoding=encoding,
                drop_first=drop_first,
                nan_policy=nan_policy,
                max_cardinality=max_cardinality,
            )
            signal = encoder.fit_transform(data)
        else:
            signal = np.ascontiguousarray(data.values, dtype=np.float64)
    else:
        signal = np.ascontiguousarray(np.asarray(data, dtype=np.float64))
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)

    # ---- instantiate the estimator -------------------------------------
    if method == "kernelcpd":
        algo = KernelCPD(kernel=kernel, min_size=min_size, params=params)
    elif method == "window":
        algo = Window(
            width=width,
            model=model,
            custom_cost=custom_cost,
            min_size=min_size,
            jump=jump,
            params=params,
        )
    else:
        algo = _ESTIMATORS[method](
            model=model,
            custom_cost=custom_cost,
            min_size=min_size,
            jump=jump,
            params=params,
        )
    algo.fit(signal)

    # ---- predict -------------------------------------------------------
    if n_bkps is not None:
        if method == "pelt":
            raise ValueError("method='pelt' does not accept n_bkps; use 'dynp'.")
        bkps = algo.predict(n_bkps=n_bkps)
    elif pen is not None:
        if method not in _PEN_METHODS:
            raise ValueError("method={!r} does not accept pen".format(method))
        bkps = algo.predict(pen=pen)
    elif epsilon is not None:
        bkps = algo.predict(epsilon=epsilon)
    else:
        # nothing specified: fall back to a BIC default for penalty methods
        if method not in _PEN_METHODS:
            raise ValueError(
                "method={!r} requires n_bkps (or epsilon)".format(method)
            )
        pen = default_penalty(signal)
        bkps = algo.predict(pen=pen)

    breakpoints = list(bkps[:-1])

    # ---- timestamps ----------------------------------------------------
    timestamps = []
    if index is not None:
        timestamps = [index[i] for i in breakpoints if 0 <= i < len(index)]

    # ---- per-segment summary ------------------------------------------
    segments = _build_segments(data, bkps, index)

    return DetectionResult(
        bkps=list(bkps),
        breakpoints=breakpoints,
        timestamps=timestamps,
        segments=segments,
        signal=signal,
        encoder=encoder,
        estimator=algo,
    )


def _build_segments(data, bkps, index):
    """Return a tidy per-segment DataFrame (or None if pandas is unavailable)."""
    if pd is None:
        return None
    starts = [0] + list(bkps[:-1])
    ends = list(bkps)  # exclusive ends
    rows = []
    is_df = _is_dataframe(data)
    if not is_df:
        arr = np.asarray(data)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
    for st, en in zip(starts, ends):
        row = {"start": st, "end": en, "size": en - st}
        if index is not None:
            row["start_label"] = index[st]
            row["end_label"] = index[en - 1]
        if is_df:
            num = data.select_dtypes(include=[np.number])
            for c in num.columns:
                row["mean_{}".format(c)] = float(num[c].iloc[st:en].mean())
        else:
            seg = arr[st:en]
            for j in range(seg.shape[1]):
                row["mean_x{}".format(j)] = float(np.nanmean(seg[:, j]))
        rows.append(row)
    return pd.DataFrame(rows)
