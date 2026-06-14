"""ruptures_lite -- a pure-source, Python-3.8-safe port of ``ruptures``.

Offline multivariate change point detection using only numpy, scipy, pandas and
scikit-learn (no compiled extensions, no install step -- drop the folder on
``sys.path``). The estimator/cost API mirrors ruptures v1.1.10 so existing
ruptures code runs with a one-line import change::

    import ruptures_lite as rpt
    algo = rpt.Pelt(model="l2").fit(signal)
    bkps = algo.predict(pen=10)

Extras beyond ruptures:

* :mod:`ruptures_lite.preprocessing` -- categorical/numeric encoding
  (:class:`FrameEncoder`, :func:`prepare_signal`).
* :func:`ruptures_lite.detect` -- one-call DataFrame -> breakpoints + segments.

See LICENSE for upstream (BSD-2-Clause) attribution.
"""

from __future__ import annotations

from . import costs, datasets, metrics
from .datasets import pw_constant, pw_linear, pw_normal, pw_wavy
from .detection import Binseg, BottomUp, Dynp, KernelCPD, Pelt, Window
from .exceptions import BadSegmentationParameters, NotEnoughPoints
from .highlevel import DetectionResult, default_penalty, detect
from .preprocessing import FrameEncoder, prepare_signal
from .show import display
from .version import __ruptures_compat__, __version__

__all__ = [
    # estimators
    "Binseg",
    "BottomUp",
    "Dynp",
    "KernelCPD",
    "Pelt",
    "Window",
    # subpackages
    "costs",
    "datasets",
    "metrics",
    # datasets
    "pw_constant",
    "pw_linear",
    "pw_normal",
    "pw_wavy",
    # exceptions
    "BadSegmentationParameters",
    "NotEnoughPoints",
    # display
    "display",
    # value-add
    "FrameEncoder",
    "prepare_signal",
    "detect",
    "DetectionResult",
    "default_penalty",
    # version
    "__version__",
    "__ruptures_compat__",
]
