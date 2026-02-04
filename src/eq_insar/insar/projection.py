"""
InSAR line-of-sight projection and phase conversion functions.

Provides functions for:
- Converting 3D displacement vectors to satellite line-of-sight (LOS)
- Converting LOS displacement to interferometric phase
- Phase wrapping and unwrapping
"""

import numpy as np
from typing import Tuple, Union, Optional

from ..constants import SENTINEL1_WAVELENGTH_M, get_satellite


def compute_los_vector(
    incidence_deg: float,
    heading_deg: float,
) -> Tuple[float, float, float]:
    """
    Compute unit vector pointing from ground to satellite (LOS direction).

    The LOS vector components represent how much of each displacement
    direction contributes to the measured range change.

    Parameters
    ----------
    incidence_deg : float
        Incidence angle from vertical (degrees)
        0° = looking straight down, 90° = horizontal
    heading_deg : float
        Satellite heading/azimuth from North (degrees)
        Measured clockwise from North
        - Ascending (northward): typically -13° to -10°
        - Descending (southward): typically -167° to -170°

    Returns
    -------
    los_e, los_n, los_u : float
        LOS unit vector components (East, North, Up)
        Positive values indicate that motion in that direction
        results in range decrease (motion toward satellite).

    Notes
    -----
    For a typical Sentinel-1 ascending geometry (inc=33°, head=-13°):
    - los_e ≈ 0.12 (eastward motion → small range decrease)
    - los_n ≈ 0.53 (northward motion → range decrease)
    - los_u ≈ 0.84 (uplift → range decrease)

    For descending (inc=33°, head=-167°):
    - los_e ≈ -0.12 (westward motion → range decrease)
    - los_n ≈ -0.53 (southward motion → range decrease)
    - los_u ≈ 0.84 (uplift → range decrease)
    """
    inc = np.deg2rad(incidence_deg)
    head = np.deg2rad(heading_deg)

    # LOS unit vector pointing from ground to satellite
    # Positive = motion toward satellite (range decrease)
    los_e = -np.sin(inc) * np.sin(head)
    los_n = np.sin(inc) * np.cos(head)
    los_u = np.cos(inc)

    return los_e, los_n, los_u


def compute_los_displacement(
    Ue: np.ndarray,
    Un: np.ndarray,
    Uz: np.ndarray,
    incidence_deg: Optional[float] = None,
    heading_deg: Optional[float] = None,
    satellite: Optional[str] = None,
    orbit: str = "ascending",
) -> np.ndarray:
    """
    Project 3D displacement to satellite line-of-sight (LOS).

    Positive LOS displacement = motion toward satellite (range decrease).
    In interferograms, this corresponds to a particular phase sign
    (typically negative phase for range decrease with standard convention).

    Parameters
    ----------
    Ue, Un, Uz : np.ndarray
        East, North, Up displacement components (meters)
    incidence_deg : float, optional
        Incidence angle from vertical (degrees)
        Either provide this + heading_deg, or use satellite parameter
    heading_deg : float, optional
        Satellite heading from North (degrees)
    satellite : str, optional
        Satellite name for automatic geometry. Options:
        'sentinel1', 'alos2', 'terrasar', 'cosmo', 'radarsat2', etc.
        If provided, overrides incidence_deg and heading_deg
    orbit : str
        Orbit direction: 'ascending' or 'descending'
        Only used if satellite is specified

    Returns
    -------
    d_los : np.ndarray
        Line-of-sight displacement (meters)
        Positive = motion toward satellite

    Examples
    --------
    >>> import numpy as np
    >>> # Using explicit geometry
    >>> d_los = compute_los_displacement(Ue, Un, Uz, incidence_deg=33, heading_deg=-13)
    >>>
    >>> # Using satellite configuration
    >>> d_los = compute_los_displacement(Ue, Un, Uz, satellite='sentinel1', orbit='ascending')
    """
    # Get geometry from satellite configuration or explicit parameters
    if satellite is not None:
        sat = get_satellite(satellite)
        incidence_deg = sat.incidence_deg
        heading_deg = sat.get_heading(orbit)
    elif incidence_deg is None or heading_deg is None:
        raise ValueError(
            "Must provide either (incidence_deg, heading_deg) or satellite name"
        )

    # Get LOS unit vector
    los_e, los_n, los_u = compute_los_vector(incidence_deg, heading_deg)

    # Project displacement onto LOS
    d_los = Ue * los_e + Un * los_n + Uz * los_u

    return d_los


def displacement_to_phase(
    displacement_m: np.ndarray,
    wavelength_m: float = SENTINEL1_WAVELENGTH_M,
    satellite: Optional[str] = None,
) -> np.ndarray:
    """
    Convert LOS displacement to interferometric phase.

    Uses the two-way path relation:
        phase = -4π * displacement / wavelength

    The negative sign follows the convention that:
    - Range decrease (motion toward satellite) → negative phase
    - Range increase (motion away from satellite) → positive phase

    Parameters
    ----------
    displacement_m : np.ndarray
        LOS displacement in meters (positive = toward satellite)
    wavelength_m : float
        Radar wavelength in meters (default: Sentinel-1 C-band)
    satellite : str, optional
        Satellite name to automatically use correct wavelength.
        Overrides wavelength_m if provided.

    Returns
    -------
    phase : np.ndarray
        Interferometric phase in radians (unwrapped)

    Notes
    -----
    For Sentinel-1 (λ = 5.55 cm):
    - 1 cm LOS displacement ≈ 2.26 radians ≈ 0.36 fringes
    - One fringe (2π) ≈ 2.77 cm LOS displacement

    For ALOS-2 (λ = 22.9 cm):
    - 1 cm LOS displacement ≈ 0.55 radians
    - One fringe (2π) ≈ 11.5 cm LOS displacement
    """
    if satellite is not None:
        sat = get_satellite(satellite)
        wavelength_m = sat.wavelength_m

    return -4.0 * np.pi * displacement_m / wavelength_m


def phase_to_displacement(
    phase: np.ndarray,
    wavelength_m: float = SENTINEL1_WAVELENGTH_M,
    satellite: Optional[str] = None,
) -> np.ndarray:
    """
    Convert interferometric phase to LOS displacement.

    Inverse of displacement_to_phase.

    Parameters
    ----------
    phase : np.ndarray
        Interferometric phase in radians
    wavelength_m : float
        Radar wavelength in meters
    satellite : str, optional
        Satellite name to automatically use correct wavelength

    Returns
    -------
    displacement_m : np.ndarray
        LOS displacement in meters
    """
    if satellite is not None:
        sat = get_satellite(satellite)
        wavelength_m = sat.wavelength_m

    return -phase * wavelength_m / (4.0 * np.pi)


def wrap_phase(phase: np.ndarray) -> np.ndarray:
    """
    Wrap phase to [-π, π] interval.

    Simulates the inherent phase ambiguity in interferometric measurements.

    Parameters
    ----------
    phase : np.ndarray
        Unwrapped phase in radians

    Returns
    -------
    wrapped : np.ndarray
        Wrapped phase in [-π, π] radians
    """
    return np.angle(np.exp(1j * phase))


def fringe_count(
    displacement_m: np.ndarray,
    wavelength_m: float = SENTINEL1_WAVELENGTH_M,
) -> np.ndarray:
    """
    Calculate number of fringes from displacement.

    One fringe represents λ/2 of LOS displacement (two-way path).

    Parameters
    ----------
    displacement_m : np.ndarray
        LOS displacement in meters
    wavelength_m : float
        Radar wavelength in meters

    Returns
    -------
    n_fringes : np.ndarray
        Number of fringes (can be fractional)
    """
    return 2.0 * np.abs(displacement_m) / wavelength_m
