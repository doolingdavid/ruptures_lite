"""Factory function for Cost classes.

Ported from ``ruptures.costs.factory`` (BSD-2-Clause). See LICENSE.
"""

from __future__ import annotations

from ruptures_lite.base import BaseCost


def cost_factory(model, *args, **kwargs):
    for cls in BaseCost.__subclasses__():
        if cls.model == model:
            return cls(*args, **kwargs)
    raise ValueError("Not such model: {}".format(model))
