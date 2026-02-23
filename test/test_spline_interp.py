"""
Unit tests for gwsurrogate.spline_interp_Cwrapper.

Verifies `interpolate` and `interpolate_many` against scipy CubicSpline
(natural BCs) and cross-validates the batch function against the loop.
"""

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from gwsurrogate.spline_interp_Cwrapper import interpolate


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _natural_scipy(x, y, xnew):
    """Reference: scipy CubicSpline with natural boundary conditions."""
    return CubicSpline(x, y, bc_type="natural")(xnew)
