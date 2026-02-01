"""
EQ-INSAR Core: Earthquake source physics and deformation models.

Available source models:
- Davis (1986): Point source in elastic half-space
"""

from .moment_tensor import (
    mw_to_m0,
    m0_to_mw,
    double_couple_moment_tensor,
    slip_from_moment,
)
from .davis import davis_point_source

__all__ = [
    # Moment tensor
    "mw_to_m0",
    "m0_to_mw",
    "double_couple_moment_tensor",
    "slip_from_moment",
    # Source models
    "davis_point_source",
]
