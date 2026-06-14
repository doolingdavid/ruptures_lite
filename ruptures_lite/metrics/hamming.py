"""Hamming metric for segmentation. Ported from ruptures (BSD-2-Clause)."""

from __future__ import annotations

from ruptures_lite.metrics.randindex import randindex


def hamming(bkps1, bkps2):
    """Modified Hamming distance for partitions (scaled to [0, 1]).

    Returns:
        float: Hamming distance.
    """
    return 1 - randindex(bkps1=bkps1, bkps2=bkps2)
