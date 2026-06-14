"""Utility functions to generate data sets."""

from __future__ import annotations

from .pw_constant import pw_constant
from .pw_linear import pw_linear
from .pw_normal import pw_normal
from .pw_wavy import pw_wavy

__all__ = ["pw_constant", "pw_linear", "pw_normal", "pw_wavy"]
