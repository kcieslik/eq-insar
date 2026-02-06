"""
Single interferogram generator for EQ-INSAR.

Generates synthetic InSAR data for a single earthquake event using
the Davis (1986) point source model.
"""

import numpy as np
from typing import Dict, Optional, Union

from ..constants import (
    DEFAULT_SHEAR_MODULUS_PA,
    DEFAULT_POISSON_RATIO,
    get_satellite,
    SatelliteConfig,
)
from ..core import (
    davis_point_source,
    mw_to_m0,
    m0_to_mw,
    double_couple_moment_tensor,
)
from ..insar import (
    compute_los_displacement,
    displacement_to_phase,
    wrap_phase,
    generate_random_noise,
    generate_orbital_ramp,
)


def generate_synthetic_insar(
    # Source parameters
    Mw: Optional[float] = None,
    M0: Optional[float] = None,
    strike_deg: float = 0.0,
    dip_deg: float = 45.0,
    rake_deg: float = 90.0,
    # Source location
    xcen_km: float = 0.0,
    ycen_km: float = 0.0,
    depth_km: float = 10.0,
    # Grid parameters
    grid_size: Optional[int] = None,
    grid_extent_km: float = 50.0,
    grid_spacing_km: Optional[float] = None,
    # Elastic parameters
    nu: float = DEFAULT_POISSON_RATIO,
    mu: float = DEFAULT_SHEAR_MODULUS_PA,
    # InSAR geometry - can specify satellite or manual
    satellite: Optional[str] = None,
    orbit: str = "ascending",
    incidence_deg: Optional[float] = None,
    heading_deg: Optional[float] = None,
    wavelength_m: Optional[float] = None,
    # Noise parameters
    add_noise: bool = True,
    noise_amplitude_m: float = 0.005,
    add_orbital_ramp: bool = False,
    # Output options
    wrap: bool = True,
    seed: Optional[int] = None,
) -> Dict:
    """
    Generate synthetic InSAR data for earthquake deformation.

    This is the main generator function that creates a complete synthetic
    interferogram including displacement field, phase, and noise.
    Uses the Davis (1986) point source model.

    Parameters
    ----------
    Mw : float, optional
        Moment magnitude (provide either Mw or M0)
    M0 : float, optional
        Scalar seismic moment in N·m
    strike_deg : float
        Fault strike in degrees (0-360, clockwise from North)
    dip_deg : float
        Fault dip in degrees (0-90, from horizontal)
    rake_deg : float
        Slip rake in degrees (-180 to 180)
        - 0°: left-lateral strike-slip
        - 90°: thrust/reverse
        - ±180°: right-lateral strike-slip
        - -90°: normal fault
    xcen_km, ycen_km : float
        Epicenter location in km
    depth_km : float
        Source depth in km (positive downward)

    grid_size : int, optional
        Number of pixels for the grid (e.g., 128 for 128x128).
        If provided, grid_spacing_km is calculated automatically.
    grid_extent_km : float
        Half-width of the grid in km (default: 50 km, so 100 km total)
    grid_spacing_km : float, optional
        Grid spacing in km. If not provided and grid_size is given,
        calculated as: 2 * grid_extent_km / (grid_size - 1)

    nu : float
        Poisson's ratio (default: 0.25)
    mu : float
        Shear modulus in Pa (default: 30 GPa)

    satellite : str, optional
        Satellite name for automatic geometry. Options:
        'sentinel1', 'alos2', 'terrasar', 'cosmo', 'radarsat2', 'nisar', etc.
        If provided, overrides incidence_deg, heading_deg, wavelength_m
    orbit : str
        Orbit direction: 'ascending' or 'descending' (used with satellite)
    incidence_deg : float, optional
        Radar incidence angle in degrees (manual specification)
    heading_deg : float, optional
        Satellite heading in degrees (manual specification)
    wavelength_m : float, optional
        Radar wavelength in meters (manual specification)

    add_noise : bool
        Whether to add random noise
    noise_amplitude_m : float
        Noise standard deviation in meters (default: 5 mm)
    add_orbital_ramp : bool
        Whether to add orbital ramp artifact

    wrap : bool
        Whether to wrap phase to [-π, π]
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    result : dict
        Dictionary containing:
        - X_km, Y_km: coordinate grids (km)
        - Ue, Un, Uz: displacement components (m)
        - los_displacement: LOS displacement (m)
        - phase_unwrapped: unwrapped phase (rad)
        - phase_wrapped: wrapped phase (rad)
        - phase_noisy: wrapped phase with noise (rad)
        - metadata: source and processing parameters

    Examples
    --------
    >>> result = generate_synthetic_insar(
    ...     Mw=6.0,
    ...     strike_deg=45,
    ...     dip_deg=30,
    ...     rake_deg=90,
    ...     depth_km=10,
    ...     satellite='sentinel1',
    ...     orbit='ascending'
    ... )
    """
    if seed is not None:
        np.random.seed(seed)

    # Handle grid_size parameter
    if grid_size is not None:
        # Calculate grid_spacing_km from grid_size and grid_extent_km
        grid_spacing_km = (2 * grid_extent_km) / (grid_size - 1)
    elif grid_spacing_km is None:
        # Default spacing if neither grid_size nor grid_spacing_km provided
        grid_spacing_km = 0.5

    # Get seismic moment
    if M0 is None:
        if Mw is None:
            raise ValueError("Must provide either Mw or M0")
        M0 = mw_to_m0(Mw)
    else:
        Mw = m0_to_mw(M0)

    # Get satellite configuration or use manual parameters
    if satellite is not None:
        sat = get_satellite(satellite)
        if incidence_deg is None:
            incidence_deg = sat.incidence_deg
        if heading_deg is None:
            heading_deg = sat.get_heading(orbit)
        if wavelength_m is None:
            wavelength_m = sat.wavelength_m
        satellite_name = sat.name
    else:
        # Defaults if nothing specified
        if incidence_deg is None:
            incidence_deg = 33.0
        if heading_deg is None:
            heading_deg = -13.0
        if wavelength_m is None:
            wavelength_m = 0.05546  # Sentinel-1
        satellite_name = "custom"

    # Create coordinate grids
    if grid_size is not None:
        # Use linspace for exact grid_size
        xs_km = np.linspace(-grid_extent_km, grid_extent_km, grid_size)
        ys_km = np.linspace(-grid_extent_km, grid_extent_km, grid_size)
    else:
        # Use arange with spacing
        xs_km = np.arange(-grid_extent_km, grid_extent_km + grid_spacing_km, grid_spacing_km)
        ys_km = np.arange(-grid_extent_km, grid_extent_km + grid_spacing_km, grid_spacing_km)
    X_km, Y_km = np.meshgrid(xs_km, ys_km)

    # Convert to meters for physics calculations
    X_m = X_km * 1000.0
    Y_m = Y_km * 1000.0
    xcen_m = xcen_km * 1000.0
    ycen_m = ycen_km * 1000.0
    depth_m = depth_km * 1000.0

    # Build moment tensor
    Mxx, Myy, Mzz, Mxy, Myz, Mzx = double_couple_moment_tensor(
        strike_deg, dip_deg, rake_deg, M0
    )

    # Compute displacement using Davis point source
    Ue, Un, Uz = davis_point_source(
        X_m, Y_m, xcen_m, ycen_m, depth_m,
        Mxx, Myy, Mzz, Mxy, Myz, Mzx,
        nu=nu, mu=mu
    )

    # Compute LOS displacement
    d_los = compute_los_displacement(
        Ue, Un, Uz,
        incidence_deg=incidence_deg,
        heading_deg=heading_deg
    )

    # Convert to phase
    phase_unwrapped = displacement_to_phase(d_los, wavelength_m)
    phase_wrapped = wrap_phase(phase_unwrapped) if wrap else phase_unwrapped

    # Generate noise
    ny, nx = X_km.shape

    # Initialize noisy phase
    phase_noisy = phase_unwrapped.copy()

    if add_noise:
        # Random noise
        noise = generate_random_noise(
            (ny, nx),
            amplitude_m=noise_amplitude_m,
            seed=seed
        )
        noise_phase = displacement_to_phase(noise, wavelength_m)
        phase_noisy = phase_noisy + noise_phase

        # Orbital ramp
        if add_orbital_ramp:
            ramp = generate_orbital_ramp(
                (ny, nx),
                pixel_size_km=grid_spacing_km,
                seed=seed + 300 if seed else None
            )
            ramp_phase = displacement_to_phase(ramp, wavelength_m)
            phase_noisy = phase_noisy + ramp_phase

    # Wrap noisy phase
    if wrap:
        phase_noisy = wrap_phase(phase_noisy)

    # Package metadata
    metadata = {
        # Source parameters
        "Mw": Mw,
        "M0_Nm": M0,
        "strike_deg": strike_deg,
        "dip_deg": dip_deg,
        "rake_deg": rake_deg,
        "xcen_km": xcen_km,
        "ycen_km": ycen_km,
        "depth_km": depth_km,
        "source_type": "davis",
        # Grid parameters
        "grid_extent_km": grid_extent_km,
        "grid_spacing_km": grid_spacing_km,
        # Elastic parameters
        "nu": nu,
        "mu_Pa": mu,
        # InSAR geometry
        "satellite": satellite_name,
        "orbit": orbit,
        "incidence_deg": incidence_deg,
        "heading_deg": heading_deg,
        "wavelength_m": wavelength_m,
        # Noise parameters
        "noise_amplitude_m": noise_amplitude_m,
    }

    return {
        "X_km": X_km,
        "Y_km": Y_km,
        "Ue": Ue,
        "Un": Un,
        "Uz": Uz,
        "los_displacement": d_los,
        "phase_unwrapped": phase_unwrapped,
        "phase_wrapped": phase_wrapped,
        "phase_noisy": phase_noisy,
        "metadata": metadata,
    }
