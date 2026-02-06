"""
Time series generator for EQ-INSAR.

Generates sequences of InSAR frames with pre-event (noise only),
co-seismic (signal + noise), and post-event (noise only) phases.
Ideal for training ML models for earthquake detection and segmentation.
"""

import numpy as np
from typing import Dict, Optional

from ..constants import (
    DEFAULT_SHEAR_MODULUS_PA,
    DEFAULT_POISSON_RATIO,
)
from ..insar import (
    displacement_to_phase,
    wrap_phase,
    generate_random_noise,
)
from .single import generate_synthetic_insar


def generate_timeseries(
    # Source parameters
    Mw: Optional[float] = None,
    M0: Optional[float] = None,
    strike_deg: float = 0.0,
    dip_deg: float = 45.0,
    rake_deg: float = 90.0,
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
    # InSAR geometry
    satellite: Optional[str] = None,
    orbit: str = "ascending",
    incidence_deg: Optional[float] = None,
    heading_deg: Optional[float] = None,
    wavelength_m: Optional[float] = None,
    # Time series parameters
    n_pre: int = 5,
    n_event: int = 1,
    n_post: int = 5,
    # Noise parameters
    noise_amplitude_m: float = 0.005,
    # Output options
    wrap: bool = True,
    output_type: str = "phase",  # 'phase', 'displacement', 'both'
    deformation_threshold_m: float = 0.005,  # 5mm for label mask
    seed: Optional[int] = None,
) -> Dict:
    """
    Generate time series of synthetic InSAR data.

    Creates a sequence with:
    - Pre-event frames: noise only (no deformation signal)
    - Event frames: deformation signal + noise
    - Post-event frames: noise only

    This format is designed for training ML models to detect and
    segment earthquake deformation signals.

    Parameters
    ----------
    Mw : float, optional
        Moment magnitude
    M0 : float, optional
        Seismic moment in N·m
    strike_deg, dip_deg, rake_deg : float
        Fault geometry parameters
    xcen_km, ycen_km : float
        Epicenter location in km
    depth_km : float
        Source depth in km

    grid_size : int, optional
        Number of pixels for the grid (e.g., 128 for 128x128).
        If provided, grid_spacing_km is calculated automatically.
    grid_extent_km : float
        Half-width of the grid in km (default: 50 km)
    grid_spacing_km : float, optional
        Grid spacing in km. If not provided and grid_size is given,
        calculated as: 2 * grid_extent_km / (grid_size - 1)

    satellite : str, optional
        Satellite configuration ('sentinel1', 'alos2', etc.)
    orbit : str
        'ascending' or 'descending'

    n_pre : int
        Number of pre-event frames (noise only)
    n_event : int
        Number of event frames (signal + noise)
    n_post : int
        Number of post-event frames (noise only)

    noise_amplitude_m : float
        Noise standard deviation in meters

    wrap : bool
        Wrap phase to [-π, π]
    output_type : str
        'phase' (default), 'displacement', or 'both'
    deformation_threshold_m : float
        Threshold for creating binary deformation labels
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    result : dict
        Dictionary containing:
        - All fields from generate_synthetic_insar (static deformation)
        - timeseries: (n_total, ny, nx) array of frames
        - labels: (n_total, ny, nx) binary segmentation masks
        - metadata: includes n_pre, n_event, n_post

    Examples
    --------
    >>> result = generate_timeseries(
    ...     Mw=6.0,
    ...     satellite='sentinel1',
    ...     n_pre=5,
    ...     n_event=1,
    ...     n_post=5
    ... )
    >>> X = result['timeseries']  # Shape: (11, ny, nx)
    >>> y = result['labels']      # Binary masks
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate the static deformation signal (without noise)
    result = generate_synthetic_insar(
        Mw=Mw,
        M0=M0,
        strike_deg=strike_deg,
        dip_deg=dip_deg,
        rake_deg=rake_deg,
        xcen_km=xcen_km,
        ycen_km=ycen_km,
        depth_km=depth_km,
        grid_size=grid_size,
        grid_extent_km=grid_extent_km,
        grid_spacing_km=grid_spacing_km,
        nu=nu,
        mu=mu,
        satellite=satellite,
        orbit=orbit,
        incidence_deg=incidence_deg,
        heading_deg=heading_deg,
        wavelength_m=wavelength_m,
        add_noise=False,  # We'll add noise per-frame
        wrap=False,
        seed=seed,
    )

    ny, nx = result["X_km"].shape
    n_total = n_pre + n_event + n_post

    # Get wavelength for phase conversion
    wl = result["metadata"]["wavelength_m"]

    # Initialize output arrays
    timeseries = np.zeros((n_total, ny, nx), dtype=np.float32)

    # Create deformation labels (binary mask)
    labels = np.zeros((n_total, ny, nx), dtype=np.int32)
    deformation_mask = np.abs(result["los_displacement"]) > deformation_threshold_m

    # Generate frames
    for t in range(n_total):
        # Generate per-frame random noise (different each frame)
        noise = generate_random_noise(
            (ny, nx),
            amplitude_m=noise_amplitude_m,
            seed=seed + t * 100 if seed else None
        )

        # Check if this is an event frame
        is_event_frame = n_pre <= t < n_pre + n_event

        if is_event_frame:
            # Event frame: signal + noise
            if output_type == "displacement":
                timeseries[t] = result["los_displacement"] + noise
            else:  # phase
                signal_phase = result["phase_unwrapped"]
                noise_phase = displacement_to_phase(noise, wl)
                total_phase = signal_phase + noise_phase
                timeseries[t] = wrap_phase(total_phase) if wrap else total_phase

            # Set label
            labels[t] = deformation_mask.astype(np.int32)
        else:
            # Non-event frame: noise only
            if output_type == "displacement":
                timeseries[t] = noise
            else:  # phase
                noise_phase = displacement_to_phase(noise, wl)
                timeseries[t] = wrap_phase(noise_phase) if wrap else noise_phase
            # Labels remain 0

    # Add time series data to result
    result["timeseries"] = timeseries
    result["labels"] = labels

    # Update metadata
    result["metadata"]["n_pre"] = n_pre
    result["metadata"]["n_event"] = n_event
    result["metadata"]["n_post"] = n_post
    result["metadata"]["n_total"] = n_total
    result["metadata"]["output_type"] = output_type
    result["metadata"]["deformation_threshold_m"] = deformation_threshold_m

    return result
