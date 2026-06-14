r"""Categorical / numeric preprocessing for change point detection.

This layer is **not** part of upstream ruptures -- it is the value-add of
``ruptures_lite``. It turns a heterogeneous :class:`pandas.DataFrame` (mixed
numeric and categorical columns) into a contiguous ``float64`` matrix that the
estimators can consume, while recording the provenance of every output column so
that a detected change point can be traced back to the original feature(s).

Design choices (all version-robust on old Anaconda stacks):

* One-hot encoding uses plain numpy equality tests (not
  ``sklearn.OneHotEncoder``, whose ``sparse`` keyword was renamed to
  ``sparse_output`` in scikit-learn 1.2 and would break across versions).
* Standardization is a manual z-score with a guard for zero-variance columns
  (no ``StandardScaler`` dependency, though the result is identical).
* Categories are learned at :meth:`fit` time; unseen categories at
  :meth:`transform` time map to an all-zero block (graceful, no error).
"""

from __future__ import annotations

import numpy as np

try:  # pandas is available in the target environment
    import pandas as pd
    from pandas.api.types import (
        is_bool_dtype,
        is_integer_dtype,
        is_numeric_dtype,
    )
except ImportError:  # pragma: no cover - pandas guaranteed in target env
    pd = None


def _is_dataframe(obj):
    return pd is not None and isinstance(obj, pd.DataFrame)


class FrameEncoder:
    """Encode a mixed-dtype DataFrame into a numeric matrix for segmentation.

    Args:
        categorical (list, optional): column names to treat as categorical. If
            ``None``, auto-detect (object / category / bool dtypes, plus integer
            columns whose cardinality is ``<= max_cardinality`` when that is set).
        numeric (list, optional): column names to treat as numeric. If ``None``,
            every remaining column with a numeric dtype.
        scale (bool, optional): z-score numeric columns. Defaults to True.
        encoding ({"onehot", "ordinal"}, optional): categorical encoding.
            Defaults to "onehot" (recommended; imposes no artificial ordering).
        drop_first (bool, optional): for one-hot, drop the first level of each
            categorical (reduces collinearity). Defaults to False.
        max_cardinality (int, optional): when auto-detecting, also treat integer
            columns with at most this many unique values as categorical.
            Defaults to None (integers stay numeric).
        nan_policy ({"raise", "fill"}, optional): how to handle missing values.
            "raise" (default) errors on any NaN. "fill" replaces numeric NaN with
            the fitted column mean and encodes categorical NaN as an all-zero block.
    """

    def __init__(
        self,
        categorical=None,
        numeric=None,
        scale=True,
        encoding="onehot",
        drop_first=False,
        max_cardinality=None,
        nan_policy="raise",
    ):
        assert encoding in ("onehot", "ordinal"), "encoding must be onehot/ordinal"
        assert nan_policy in ("raise", "fill"), "nan_policy must be raise/fill"
        self.categorical = categorical
        self.numeric = numeric
        self.scale = scale
        self.encoding = encoding
        self.drop_first = drop_first
        self.max_cardinality = max_cardinality
        self.nan_policy = nan_policy

    # -- fitting -----------------------------------------------------------
    def _infer_columns(self, df):
        cols = list(df.columns)
        if self.categorical is not None:
            cat = list(self.categorical)
        else:
            cat = []
            for c in cols:
                col = df[c]
                # bool / object / string / pandas-categorical -> categorical;
                # use pandas predicates so this is robust across pandas versions
                # (old pandas: object dtype; pandas >= 2/3: StringDtype, etc.).
                if is_bool_dtype(col) or not is_numeric_dtype(col):
                    cat.append(c)
                elif (
                    self.max_cardinality is not None
                    and is_integer_dtype(col)
                    and col.nunique(dropna=True) <= self.max_cardinality
                ):
                    cat.append(c)
        if self.numeric is not None:
            num = list(self.numeric)
        else:
            num = [
                c
                for c in cols
                if c not in cat
                and is_numeric_dtype(df[c])
                and not is_bool_dtype(df[c])
            ]
        return cat, num

    def fit(self, df):
        """Learn columns, category levels and scaling statistics."""
        if not _is_dataframe(df):
            raise TypeError("FrameEncoder expects a pandas DataFrame.")
        self.cat_cols_, self.num_cols_ = self._infer_columns(df)

        # category levels (sorted for determinism)
        self.categories_ = {}
        for c in self.cat_cols_:
            levels = pd.unique(df[c].dropna())
            try:
                levels = sorted(levels.tolist())
            except TypeError:  # unorderable -> keep first-seen order
                levels = list(levels)
            self.categories_[c] = levels

        # numeric scaling stats
        self.means_ = {}
        self.stds_ = {}
        for c in self.num_cols_:
            col = df[c].astype(float)
            self.means_[c] = float(col.mean())
            std = float(col.std(ddof=0))
            self.stds_[c] = std if std > 1e-12 else 1.0

        # output column provenance
        self.columns_ = []
        for c in self.num_cols_:
            self.columns_.append(c)
        for c in self.cat_cols_:
            levels = self.categories_[c]
            start = 1 if (self.encoding == "onehot" and self.drop_first) else 0
            if self.encoding == "ordinal":
                self.columns_.append(c)
            else:
                for level in levels[start:]:
                    self.columns_.append("{}={}".format(c, level))
        return self

    # -- transforming ------------------------------------------------------
    def _check_nans(self, df, columns):
        if self.nan_policy == "raise":
            bad = [c for c in columns if df[c].isna().any()]
            if bad:
                raise ValueError(
                    "NaNs found in columns {}; set nan_policy='fill' to impute.".format(
                        bad
                    )
                )

    def transform(self, df):
        """Encode ``df`` into a contiguous float64 matrix.

        Returns:
            numpy.ndarray: shape (n_rows, len(self.columns_)).
        """
        if not _is_dataframe(df):
            raise TypeError("FrameEncoder expects a pandas DataFrame.")
        self._check_nans(df, self.num_cols_ + self.cat_cols_)

        parts = []
        # numeric block
        for c in self.num_cols_:
            x = df[c].astype(float).values.astype(np.float64)
            if self.nan_policy == "fill":
                x = np.where(np.isnan(x), self.means_[c], x)
            if self.scale:
                x = (x - self.means_[c]) / self.stds_[c]
            parts.append(x.reshape(-1, 1))

        # categorical block
        for c in self.cat_cols_:
            levels = self.categories_[c]
            col_values = df[c].values
            if self.encoding == "ordinal":
                code_map = {lvl: i for i, lvl in enumerate(levels)}
                codes = np.array(
                    [code_map.get(v, np.nan) for v in col_values], dtype=np.float64
                )
                if self.nan_policy == "fill":
                    codes = np.where(np.isnan(codes), 0.0, codes)
                parts.append(codes.reshape(-1, 1))
            else:
                start = 1 if self.drop_first else 0
                for level in levels[start:]:
                    parts.append((col_values == level).astype(np.float64).reshape(-1, 1))

        if not parts:
            raise ValueError("No usable columns found to build the signal.")
        signal = np.ascontiguousarray(np.hstack(parts), dtype=np.float64)
        return signal

    def fit_transform(self, df):
        return self.fit(df).transform(df)


def prepare_signal(data, return_encoder=False, **kwargs):
    """Convenience wrapper: encode ``data`` into a float64 matrix.

    Args:
        data: a pandas DataFrame (encoded with :class:`FrameEncoder`) or an
            array-like (returned as a contiguous float64 2-D array unchanged).
        return_encoder (bool): if True, also return the fitted FrameEncoder.
        **kwargs: forwarded to :class:`FrameEncoder`.

    Returns:
        ndarray, or (ndarray, FrameEncoder) if ``return_encoder``.
    """
    if _is_dataframe(data):
        enc = FrameEncoder(**kwargs)
        signal = enc.fit_transform(data)
        return (signal, enc) if return_encoder else signal

    arr = np.ascontiguousarray(np.asarray(data, dtype=np.float64))
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return (arr, None) if return_encoder else arr
