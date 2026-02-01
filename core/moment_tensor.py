"""
Moment tensor construction and magnitude/moment conversions.

Provides functions for:
- Converting between moment magnitude (Mw) and seismic moment (M0)
- Constructing double-couple moment tensors from fault geometry
- Computing fault slip from seismic moment

References
----------
- Aki, K. & Richards, P.G. (2002). Quantitative Seismology, 2nd ed.
- Hanks, T.C. & Kanamori, H. (1979). A moment magnitude scale. JGR.
- Kanamori, H. (1977). The energy release in great earthquakes. JGR.
"""

import numpy as np
from typing import Tuple

from ..constants import DEFAULT_SHEAR_MODULUS_PA


def mw_to_m0(Mw: float) -> float:
    """
    Convert moment magnitude to scalar seismic moment.

    Uses Hanks & Kanamori (1979) relation:
        log10(M0) = 1.5 * Mw + 9.1  (M0 in N·m)

    Parameters
    ----------
    Mw : float
        Moment magnitude

    Returns
    -------
    M0 : float
        Scalar seismic moment in N·m

    Examples
    --------
    >>> mw_to_m0(6.0)
    1.2589254117941662e+18
    >>> mw_to_m0(7.0)
    3.9810717055349694e+19
    """
    return 10.0 ** (1.5 * Mw + 9.1)


def m0_to_mw(M0: float) -> float:
    """
    Convert scalar seismic moment to moment magnitude.

    Inverse of mw_to_m0.

    Parameters
    ----------
    M0 : float
        Scalar seismic moment in N·m

    Returns
    -------
    Mw : float
        Moment magnitude
    """
    return (np.log10(M0) - 9.1) / 1.5


def slip_from_moment(
    M0: float,
    length_m: float,
    width_m: float,
    mu: float = DEFAULT_SHEAR_MODULUS_PA,
) -> float:
    """
    Calculate average fault slip from seismic moment and fault dimensions.

    Uses the relation: M0 = μ * A * D
    where μ is shear modulus, A is fault area, D is average slip.

    Parameters
    ----------
    M0 : float
        Scalar seismic moment in N·m
    length_m : float
        Fault length in meters
    width_m : float
        Fault width (down-dip) in meters
    mu : float
        Shear modulus in Pa (default: 30 GPa)

    Returns
    -------
    slip_m : float
        Average slip in meters
    """
    area = length_m * width_m
    return M0 / (mu * area)


def double_couple_moment_tensor(
    strike_deg: float,
    dip_deg: float,
    rake_deg: float,
    M0: float,
) -> Tuple[float, float, float, float, float, float]:
    """
    Construct double-couple moment tensor from fault geometry.

    Uses Aki & Richards (2002) convention for fault orientation:
    - Strike: Azimuth of fault trace, clockwise from North (0-360°)
             When standing on the fault, the hanging wall is on the right.
    - Dip: Angle from horizontal to fault plane (0-90°)
           Measured perpendicular to strike, toward hanging wall.
    - Rake: Direction of slip on fault plane (-180 to 180°)
           Measured from strike direction in the fault plane.
           - 0°: Left-lateral strike-slip
           - 90°: Reverse/thrust (hanging wall up)
           - 180° or -180°: Right-lateral strike-slip
           - -90°: Normal fault (hanging wall down)

    Parameters
    ----------
    strike_deg : float
        Fault strike in degrees (0-360, clockwise from North)
    dip_deg : float
        Fault dip in degrees (0-90, from horizontal)
    rake_deg : float
        Slip rake in degrees (-180 to 180)
    M0 : float
        Scalar seismic moment in N·m

    Returns
    -------
    Mxx, Myy, Mzz, Mxy, Myz, Mzx : float
        Moment tensor components in ENU coordinates (N·m)
        - Mxx: East-East component
        - Myy: North-North component
        - Mzz: Up-Up component
        - Mxy: East-North component
        - Myz: North-Up component
        - Mzx: Up-East component

    Notes
    -----
    The returned moment tensor has zero trace (Mxx + Myy + Mzz = 0),
    which is characteristic of a pure double-couple source.
    """
    # Convert to radians
    strike = np.deg2rad(strike_deg)
    dip = np.deg2rad(dip_deg)
    rake = np.deg2rad(rake_deg)

    # Fault normal vector (pointing into footwall) in ENU
    # From Aki & Richards (2002) eq. 4.88-4.89
    n_e = np.sin(dip) * np.cos(strike)
    n_n = -np.sin(dip) * np.sin(strike)
    n_u = np.cos(dip)

    # Slip vector (direction of hanging wall motion relative to footwall) in ENU
    # From Aki & Richards (2002) eq. 4.88-4.89
    s_e = np.cos(rake) * np.sin(strike) - np.sin(rake) * np.cos(dip) * np.cos(strike)
    s_n = np.cos(rake) * np.cos(strike) + np.sin(rake) * np.cos(dip) * np.sin(strike)
    s_u = np.sin(rake) * np.sin(dip)

    # Normalize vectors (should already be unit vectors, but ensure numerical stability)
    n = np.array([n_e, n_n, n_u])
    s = np.array([s_e, s_n, s_u])
    n = n / (np.linalg.norm(n) + 1e-20)
    s = s / (np.linalg.norm(s) + 1e-20)

    # Double-couple moment tensor: M = M0 * (s⊗n + n⊗s)
    M = M0 * (np.outer(s, n) + np.outer(n, s))

    # Extract components (indices: 0=E, 1=N, 2=U)
    Mxx = M[0, 0]  # Mee
    Myy = M[1, 1]  # Mnn
    Mzz = M[2, 2]  # Muu
    Mxy = M[0, 1]  # Men
    Myz = M[1, 2]  # Mnu
    Mzx = M[2, 0]  # Mue

    return Mxx, Myy, Mzz, Mxy, Myz, Mzx


def fault_dimensions_from_magnitude(
    Mw: float,
    fault_type: str = "all",
) -> Tuple[float, float]:
    """
    Estimate fault dimensions from magnitude using scaling relations.

    Uses Wells & Coppersmith (1994) empirical relations.

    Parameters
    ----------
    Mw : float
        Moment magnitude
    fault_type : str
        Fault type: 'strike-slip', 'reverse', 'normal', or 'all'

    Returns
    -------
    length_km, width_km : float
        Estimated fault length and width in km
    """
    # Wells & Coppersmith (1994) coefficients for subsurface rupture length
    # log10(L) = a + b*Mw
    coeffs = {
        "strike-slip": (-2.57, 0.62),
        "reverse": (-2.42, 0.58),
        "normal": (-1.88, 0.50),
        "all": (-2.44, 0.59),
    }

    # Width coefficients
    width_coeffs = {
        "strike-slip": (-0.76, 0.27),
        "reverse": (-1.61, 0.41),
        "normal": (-1.14, 0.35),
        "all": (-1.01, 0.32),
    }

    fault_type = fault_type.lower().replace("_", "-").replace(" ", "-")
    if fault_type not in coeffs:
        fault_type = "all"

    a_l, b_l = coeffs[fault_type]
    a_w, b_w = width_coeffs[fault_type]

    length_km = 10.0 ** (a_l + b_l * Mw)
    width_km = 10.0 ** (a_w + b_w * Mw)

    return length_km, width_km
