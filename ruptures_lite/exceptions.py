"""Custom warnings and error classes used across ruptures_lite.

Ported verbatim from ``ruptures.exceptions`` (BSD-2-Clause). See LICENSE.
"""


class NotEnoughPoints(Exception):
    """Raised when there are not enough points to compute a cost function."""


class BadSegmentationParameters(Exception):
    """Raised when a segmentation is impossible given the parameters."""
