r"""Pure-numpy reimplementation of the compiled kernel change-point engine.

This is a faithful, line-for-line port of ruptures' C sources
(``ekcpd_computation.c``, ``ekcpd_pelt_computation.c``, ``kernels.c``) used by
:class:`KernelCPD`. Porting these removes the only hard compiled dependency in
ruptures, so the package runs from source with no build step.

The kernel conventions of the C code are reproduced exactly, including details
that differ from the ``CostRbf``/``CostCosine``/``CostL2`` cost classes:

* **linear**   ``k(x, y) = <x, y>``            (diagonal ``= ||x||^2``)
* **gaussian** ``k(x, y) = exp(-clip(gamma*||x-y||^2, 0.01, 100))``
  (diagonal ``= exp(-0.01)``, *not* ``1.0`` -- the clip lower bound applies)
* **cosine**   ``k(x, y) = <x, y> / (||x|| ||y||)``  (diagonal ``= 1.0``)

Segment cost on ``[s, t)`` (the kernel "mean change" cost) is

    c(s, t) = sum_i K[i, i]  -  (1 / (t - s)) * sum_{i, j in [s, t)} K[i, j].

See LICENSE for attribution.
"""

from __future__ import annotations

import numpy as np


def _gram_matrix(signal, kernel, gamma=None):
    """Full Gram matrix using the C code's kernel conventions (diagonal included)."""
    sig = np.ascontiguousarray(signal, dtype=np.double)
    if kernel == "linear":
        return sig.dot(sig.T)
    if kernel == "gaussian":
        # squared euclidean distances (with the diagonal == 0)
        sq_norms = np.einsum("ij,ij->i", sig, sig)
        sqdist = sq_norms[:, None] + sq_norms[None, :] - 2.0 * sig.dot(sig.T)
        np.maximum(sqdist, 0.0, out=sqdist)  # guard tiny negatives from round-off
        scaled = np.clip(gamma * sqdist, 0.01, 100.0)
        return np.exp(-scaled)
    if kernel == "cosine":
        norms = np.sqrt(np.einsum("ij,ij->i", sig, sig))
        denom = np.outer(norms, norms)
        return sig.dot(sig.T) / denom
    raise ValueError("Unknown kernel: {}".format(kernel))


def ekcpd_compute(signal, n_bkps, min_size, kernel, gamma=None):
    """Dynamic-programming kernel CPD. Port of ``ekcpd_compute``.

    Returns:
        ndarray: flat int path matrix of shape ``(n_samples + 1) * (n_bkps + 1)``,
        row-major (indexed ``t * (n_bkps + 1) + k``).
    """
    K = _gram_matrix(signal, kernel, gamma)
    n_samples = K.shape[0]

    D = np.zeros(n_samples + 1, dtype=np.double)
    S = np.zeros(n_samples + 1, dtype=np.double)
    M_V = np.zeros((n_samples + 1, n_bkps + 1), dtype=np.double)
    M_path = np.zeros((n_samples + 1, n_bkps + 1), dtype=np.intc)

    for t in range(1, n_samples + 1):
        diag_element = K[t - 1, t - 1]
        D[t] = D[t - 1] + diag_element

        # incremental update of the within-segment kernel sums S[s] = S_{s, t}
        c_r = 0.0
        for s in range(t - 1, -1, -1):
            c_r += K[s, t - 1]
            S[s] += 2.0 * c_r - diag_element

        # cost on y_{0..t} with 0 break points
        M_V[t, 0] = D[t] - S[0] / t
        for s in range(min_size, t - min_size + 1):
            c_cost = D[t] - D[s] - S[s] / (t - s)
            n_bkps_max = min(n_bkps, s // min_size)
            for k in range(1, n_bkps_max + 1):
                c_cost_sum = M_V[s, k - 1] + c_cost
                if s == k * min_size:
                    # smallest admissible s for k breakpoints: initialize.
                    M_V[t, k] = c_cost_sum
                    M_path[t, k] = s
                    continue
                if M_V[t, k] > c_cost_sum:
                    M_V[t, k] = c_cost_sum
                    M_path[t, k] = s

    return M_path.reshape(-1)


def ekcpd_pelt_compute(signal, beta, min_size, kernel, gamma=None):
    """PELT kernel CPD. Port of ``ekcpd_pelt_compute``.

    Returns:
        ndarray: int path array of shape ``(n_samples + 1,)`` where ``path[t]``
        is the optimal last breakpoint before ``t``.
    """
    K = _gram_matrix(signal, kernel, gamma)
    n_samples = K.shape[0]

    D = np.zeros(n_samples + 1, dtype=np.double)
    S = np.zeros(n_samples + 1, dtype=np.double)
    M_V = np.zeros(n_samples + 1, dtype=np.double)
    M_path = np.zeros(n_samples + 1, dtype=np.intc)
    M_pruning = np.zeros(n_samples + 1, dtype=np.double)

    s_min = 0

    # for t < 2*min_size there cannot be any change point.
    for t in range(1, 2 * min_size):
        diag_element = K[t - 1, t - 1]
        D[t] = D[t - 1] + diag_element
        c_r = 0.0
        for s in range(t - 1, -1, -1):
            c_r += K[s, t - 1]
            S[s] += 2.0 * c_r - diag_element
        c_cost = D[t] - D[0] - S[0] / t
        M_V[t] = c_cost + beta

    for t in range(2 * min_size, n_samples + 1):
        diag_element = K[t - 1, t - 1]
        D[t] = D[t - 1] + diag_element
        c_r = 0.0
        for s in range(t - 1, s_min - 1, -1):
            c_r += K[s, t - 1]
            S[s] += 2.0 * c_r - diag_element

        # init at s = s_min
        s = s_min
        c_cost = D[t] - D[s] - S[s] / (t - s)
        c_cost_sum = M_V[s] + c_cost
        M_pruning[s] = c_cost_sum
        c_cost_sum += beta
        M_V[t] = c_cost_sum
        M_path[t] = s

        # search for the minimum penalized sum of costs
        for s in range(max(s_min + 1, min_size), t - min_size + 1):
            c_cost = D[t] - D[s] - S[s] / (t - s)
            c_cost_sum = M_V[s] + c_cost
            M_pruning[s] = c_cost_sum
            c_cost_sum += beta
            if M_V[t] > c_cost_sum:
                M_V[t] = c_cost_sum
                M_path[t] = s

        # pruning
        while (M_pruning[s_min] >= M_V[t]) and (s_min < t - min_size + 1):
            if s_min == 0:
                s_min += min_size
            else:
                s_min += 1

    return M_path
