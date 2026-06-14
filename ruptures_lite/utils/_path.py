"""Pure-Python reimplementation of the compiled ``convert_path_matrix`` helper.

This replaces ``ruptures.utils._utils.convert_path_matrix`` (a Cython/C
extension) with an exact, line-for-line Python port of ``convert_path_matrix_c``
so that no compiled extension is required. See LICENSE for attribution.
"""

from __future__ import annotations

from math import ceil


def from_path_matrix_to_bkps_list(path_matrix_flat, n_bkps, n_samples, n_bkps_max, jump):
    """Back-walk the flattened DP path matrix to a breakpoint list.

    Exact port of ``convert_path_matrix_c`` (ruptures).

    Args:
        path_matrix_flat: flat int array of shape ``(n_bkps_max + 1) * (n_samples + 1)``
            (row-major, indexed ``t * (n_bkps_max + 1) + k``).
        n_bkps (int): number of breakpoints for the requested segmentation.
        n_samples (int): number of samples.
        n_bkps_max (int): the maximum number of breakpoints used when the path
            matrix was filled.
        jump (int): subsampling step (KernelCPD forces this to 1).

    Returns:
        list: sorted breakpoints, with ``bkps_list[-1] == n_samples``.
    """
    q = int(ceil(float(n_samples) / float(jump)))
    bkps_list = [0] * (n_bkps + 1)
    bkps_list[n_bkps] = q
    k = 0
    while k < n_bkps:
        k += 1
        bkps_list[n_bkps - k] = int(
            path_matrix_flat[
                bkps_list[n_bkps - k + 1] * (n_bkps_max + 1) + (n_bkps - k + 1)
            ]
        )
    for k in range(n_bkps + 1):
        bkps_list[k] = bkps_list[k] * jump
    bkps_list[n_bkps] = n_samples
    return list(bkps_list)
