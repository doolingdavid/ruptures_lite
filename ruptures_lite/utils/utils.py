"""Miscellaneous helpers.

Ported verbatim from ``ruptures.utils.utils`` (BSD-2-Clause). See LICENSE.
"""

from __future__ import annotations

from itertools import tee
from math import ceil


def pairwise(iterable):
    """S -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def unzip(seq):
    """Reverse zip."""
    return zip(*seq)


def sanity_check(n_samples, n_bkps, jump, min_size):
    """Check whether a partition is possible given the segmentation parameters.

    Args:
        n_samples (int): number of points in the signal
        n_bkps (int): number of breakpoints
        jump (int): the start index of each regime can only be a multiple of
            ``jump``.
        min_size (int): minimum size of a segment.

    Returns:
        bool: True if a valid configuration of breakpoints exists.
    """
    n_adm_bkps = n_samples // jump  # number of admissible breakpoints

    if n_bkps > n_adm_bkps:
        return False
    if n_bkps * ceil(min_size / jump) * jump + min_size > n_samples:
        return False
    return True
