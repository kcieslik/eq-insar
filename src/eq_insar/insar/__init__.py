"""
EQ-INSAR InSAR: Line-of-sight projection, phase conversion, and noise models.

This module provides functions for converting 3D displacement to InSAR
observables including:
- LOS projection for various satellite geometries
- Phase conversion and wrapping
- Simple random noise model
"""

from .projection import (
    compute_los_displacement,
    compute_los_vector,
    displacement_to_phase,
    wrap_phase,
    phase_to_displacement,
)
from .noise import (
    generate_random_noise,
    generate_orbital_ramp,
)

__all__ = [
    # Projection
    "compute_los_displacement",
    "compute_los_vector",
    "displacement_to_phase",
    "wrap_phase",
    "phase_to_displacement",
    # Noise models
    "generate_random_noise",
    "generate_orbital_ramp",
]
