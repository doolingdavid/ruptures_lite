r"""Rand index (randindex). Ported from ruptures (BSD-2-Clause)."""

from __future__ import annotations

from ruptures_lite.metrics.sanity_check import sanity_check


def randindex(bkps1, bkps2):
    """Compute the Rand index (between 0 and 1) between two segmentations.

    Uses the efficient implementation of Prates, L. (2021), ArXiv:2112.03738.

    Returns:
        float: Rand index
    """
    sanity_check(bkps1, bkps2)
    n_samples = bkps1[-1]
    bkps1_with_0 = [0] + bkps1
    bkps2_with_0 = [0] + bkps2
    n_bkps1 = len(bkps1)
    n_bkps2 = len(bkps2)

    disagreement = 0
    beginj = 0  # avoids unnecessary computations
    for index_bkps1 in range(n_bkps1):
        start1 = bkps1_with_0[index_bkps1]
        end1 = bkps1_with_0[index_bkps1 + 1]
        for index_bkps2 in range(beginj, n_bkps2):
            start2 = bkps2_with_0[index_bkps2]
            end2 = bkps2_with_0[index_bkps2 + 1]
            nij = max(min(end1, end2) - max(start1, start2), 0)
            disagreement += nij * abs(end1 - end2)

            # we can skip the rest of the iteration, nij will be 0
            if end1 < end2:
                break
            else:
                beginj = index_bkps2 + 1

    disagreement /= n_samples * (n_samples - 1) / 2
    return 1.0 - disagreement
