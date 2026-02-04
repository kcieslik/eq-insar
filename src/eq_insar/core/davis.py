"""
Davis (1986) point source Green's functions for elastic half-space.

Computes surface displacement from a point moment tensor source using
the analytical solution for a homogeneous elastic half-space.

References
----------
- Davis, P.M. (1986). Surface deformation associated with dipping
  hydrofracture. Journal of Geophysical Research.
"""

import numpy as np
from typing import Tuple

from ..constants import DEFAULT_SHEAR_MODULUS_PA, DEFAULT_POISSON_RATIO


def davis_point_source(
    x: np.ndarray,
    y: np.ndarray,
    xcen: float,
    ycen: float,
    depth: float,
    Mxx: float,
    Myy: float,
    Mzz: float,
    Mxy: float,
    Myz: float,
    Mzx: float,
    nu: float = DEFAULT_POISSON_RATIO,
    mu: float = DEFAULT_SHEAR_MODULUS_PA,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute surface displacement from a point moment tensor source.

    Based on Davis (1986) formulation for a point source in an elastic
    half-space. This is the far-field approximation valid when the
    observation distance is much larger than the source dimension.

    Coordinate system:
    - x: East (positive eastward)
    - y: North (positive northward)
    - z: Up (positive upward at surface, z=0)
    - Depth is positive downward

    Parameters
    ----------
    x, y : np.ndarray
        Observation coordinates in meters (can be 1D or 2D arrays)
    xcen, ycen : float
        Source epicenter location in meters
    depth : float
        Source depth in meters (positive downward)
    Mxx, Myy, Mzz, Mxy, Myz, Mzx : float
        Moment tensor components in N·m (ENU convention)
    nu : float
        Poisson's ratio (default: 0.25)
    mu : float
        Shear modulus in Pa (default: 30 GPa)

    Returns
    -------
    Ue, Un, Uz : np.ndarray
        Surface displacement components in meters:
        - Ue: East component (positive eastward)
        - Un: North component (positive northward)
        - Uz: Vertical component (positive upward)

    Notes
    -----
    The point source approximation is valid when:
    - Source dimension << observation distance
    - Source dimension << depth

    Best suited for small-to-moderate earthquakes (Mw < 6.5) or
    far-field observations.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(-50000, 50000, 101)
    >>> y = np.linspace(-50000, 50000, 101)
    >>> X, Y = np.meshgrid(x, y)
    >>> # Simple vertical dip-slip source
    >>> Ue, Un, Uz = davis_point_source(X, Y, 0, 0, 10000,
    ...     Mxx=0, Myy=0, Mzz=0, Mxy=0, Myz=1e18, Mzx=0)
    """
    # Relative coordinates from source
    x1 = x - xcen  # East offset
    x2 = y - ycen  # North offset
    d = depth      # Depth (positive down)

    # Distance from source to observation points
    R2 = x1**2 + x2**2 + d**2
    R = np.sqrt(R2)

    # Protect against singularity at/near source location
    # Use minimum distance of 1% of depth
    min_R = max(depth * 0.01, 1.0)  # At least 1 meter
    R = np.maximum(R, min_R)
    R2 = R**2

    # Powers of R
    R3 = R**3
    R5 = R**5

    # Auxiliary terms involving R + depth
    Rpd = R + d      # R + depth
    Rpd2 = Rpd**2
    Rpd3 = Rpd**3

    # Coefficients for Green's functions
    alfa = (3.0 * R + d) / (R3 * Rpd3)
    beta = (2.0 * R + d) / (R3 * Rpd2)
    eta = 1.0 / (R * Rpd2)
    psi = 1.0 / (R * Rpd)

    # Pre-computed coordinate terms
    x11a = x1**2 * alfa
    x11b = x1**2 * beta
    x22a = x2**2 * alfa
    x22b = x2**2 * beta
    anu = 1.0 - 2.0 * nu  # 1 - 2ν appears frequently

    # =========================================================================
    # Green's functions for each moment tensor component
    # =========================================================================

    # ----- Mxx (East-East) contribution -----
    G11_x = x1 / 2.0 * (-1.0 / R3 + 3.0 * x1**2 / R5 + anu * (3.0 * eta - x11a))
    G11_y = x2 / 2.0 * (-1.0 / R3 + 3.0 * x1**2 / R5 + anu * (eta - x11a))
    G11_z = -0.5 * (d / R3 - 3.0 * x1**2 * d / R5 - anu * (psi - x11b))

    # ----- Myy (North-North) contribution -----
    G22_x = x1 / 2.0 * (-1.0 / R3 + 3.0 * x2**2 / R5 + anu * (eta - x22a))
    G22_y = x2 / 2.0 * (-1.0 / R3 + 3.0 * x2**2 / R5 + anu * (3.0 * eta - x22a))
    G22_z = -0.5 * (d / R3 - 3.0 * x2**2 * d / R5 - anu * (psi - x22b))

    # ----- Mzz (Up-Up) contribution -----
    G33_x = x1 / 2.0 * (3.0 * d**2 / R5 - 2.0 * nu / R3)
    G33_y = x2 / 2.0 * (3.0 * d**2 / R5 - 2.0 * nu / R3)
    G33_z = -d / 2.0 * (-3.0 * d**2 / R5 + 2.0 * nu / R3)

    # ----- Mxy (East-North) contribution -----
    G12_x = x2 * (3.0 * x1**2 / R5 + anu * (eta - x11a))
    G12_y = x1 * (3.0 * x2**2 / R5 + anu * (eta - x22a))
    G12_z = -x1 * x2 * (-3.0 * d / R5 + anu * beta)

    # ----- Mzx (Up-East) contribution -----
    G31_x = 3.0 * x1**2 * d / R5
    G31_y = 3.0 * x1 * x2 * d / R5
    G31_z = x1 * d * (3.0 * d / R5)

    # ----- Myz (North-Up) contribution -----
    G23_x = G31_y  # = 3*x1*x2*d/R5 (symmetric)
    G23_y = 3.0 * x2**2 * d / R5
    G23_z = x2 * d * (3.0 * d / R5)

    # =========================================================================
    # Scale factor and sum contributions
    # =========================================================================

    # Point source scaling: 1/(2πμ)
    # This converts moment (N·m) to displacement (m)
    scale = 1.0 / (2.0 * np.pi * mu)

    # Sum contributions from all moment tensor components
    Ue = scale * (
        Mxx * G11_x + Myy * G22_x + Mzz * G33_x +
        Mxy * G12_x + Myz * G23_x + Mzx * G31_x
    )
    Un = scale * (
        Mxx * G11_y + Myy * G22_y + Mzz * G33_y +
        Mxy * G12_y + Myz * G23_y + Mzx * G31_y
    )
    Uz = scale * (
        Mxx * G11_z + Myy * G22_z + Mzz * G33_z +
        Mxy * G12_z + Myz * G23_z + Mzx * G31_z
    )

    return Ue, Un, Uz
