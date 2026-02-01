"""
Batch generation functions for ML training data.

Provides functions to generate multiple synthetic InSAR samples
with randomized earthquake parameters for training machine learning models.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from .timeseries import generate_timeseries


def sample_earthquake_parameters(
    mw_range: Tuple[float, float] = (4.5, 7.0),
    depth_range_km: Tuple[float, float] = (5.0, 20.0),
    location_range_km: float = 30.0,
    fault_type: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict:
    """
    Sample random earthquake source parameters.

    Generates physically reasonable earthquake parameters with optional
    constraints on fault type.

    Parameters
    ----------
    mw_range : tuple (min, max)
        Range of moment magnitudes to sample from
    depth_range_km : tuple (min, max)
        Range of depths in km to sample from
    location_range_km : float
        Maximum offset from center for epicenter location
    fault_type : str, optional
        Constrain fault type: 'strike-slip', 'thrust', 'normal', or None (random)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    params : dict
        Dictionary with keys: Mw, strike_deg, dip_deg, rake_deg,
        xcen_km, ycen_km, depth_km
    """
    if seed is not None:
        np.random.seed(seed)

    # Sample magnitude
    Mw = np.random.uniform(*mw_range)

    # Sample location
    xcen_km = np.random.uniform(-location_range_km, location_range_km)
    ycen_km = np.random.uniform(-location_range_km, location_range_km)
    depth_km = np.random.uniform(*depth_range_km)

    # Sample fault geometry based on type
    strike_deg = np.random.uniform(0, 360)

    if fault_type is None:
        # Random fault type
        fault_type = np.random.choice(["strike-slip", "thrust", "normal"])

    if fault_type == "strike-slip":
        dip_deg = np.random.uniform(75, 90)  # Near-vertical
        rake_deg = np.random.choice([-1, 1]) * np.random.uniform(0, 30)  # Near-horizontal slip
    elif fault_type == "thrust" or fault_type == "reverse":
        dip_deg = np.random.uniform(15, 50)  # Shallow to moderate dip
        rake_deg = np.random.uniform(60, 120)  # Dominantly thrust
    elif fault_type == "normal":
        dip_deg = np.random.uniform(45, 70)  # Moderate to steep dip
        rake_deg = np.random.uniform(-120, -60)  # Dominantly normal
    else:
        # Fully random
        dip_deg = np.random.uniform(10, 80)
        rake_deg = np.random.uniform(-180, 180)

    return {
        "Mw": Mw,
        "strike_deg": strike_deg,
        "dip_deg": dip_deg,
        "rake_deg": rake_deg,
        "xcen_km": xcen_km,
        "ycen_km": ycen_km,
        "depth_km": depth_km,
    }


def generate_training_batch(
    n_samples: int,
    # Grid parameters
    grid_extent_km: float = 50.0,
    grid_spacing_km: float = 0.5,
    # Source parameters
    mw_range: Tuple[float, float] = (4.5, 7.0),
    depth_range_km: Tuple[float, float] = (5.0, 20.0),
    fault_type: Optional[str] = None,
    # InSAR geometry
    satellite: str = "sentinel1",
    orbit: str = "ascending",
    # Time series parameters
    n_pre: int = 3,
    n_event: int = 1,
    n_post: int = 3,
    # Noise parameters
    noise_range_m: Tuple[float, float] = (0.002, 0.008),
    # Output options
    wrap: bool = True,
    output_type: str = "phase",
    seed: Optional[int] = None,
    verbose: bool = True,
) -> List[Dict]:
    """
    Generate a batch of training samples with random parameters.

    Creates multiple synthetic InSAR time series with varied earthquake
    parameters, suitable for training ML models.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    grid_extent_km : float
        Half-width of the grid in km
    grid_spacing_km : float
        Grid spacing in km
    mw_range : tuple (min, max)
        Range of moment magnitudes
    depth_range_km : tuple (min, max)
        Range of depths in km
    fault_type : str, optional
        Constrain fault type: 'strike-slip', 'thrust', 'normal', or None
    satellite : str
        Satellite configuration name
    orbit : str
        'ascending' or 'descending'
    n_pre, n_event, n_post : int
        Time series structure
    noise_range_m : tuple (min, max)
        Range of noise amplitudes
    wrap : bool
        Whether to wrap phase
    output_type : str
        'phase' or 'displacement'
    seed : int, optional
        Random seed for reproducibility
    verbose : bool
        Print progress messages

    Returns
    -------
    samples : list of dict
        List of sample dictionaries from generate_timeseries

    Examples
    --------
    >>> batch = generate_training_batch(
    ...     n_samples=100,
    ...     mw_range=(5.0, 6.5),
    ...     satellite='sentinel1'
    ... )
    >>> X = np.stack([s['timeseries'] for s in batch])  # (100, T, H, W)
    >>> y = np.stack([s['labels'] for s in batch])      # (100, T, H, W)
    """
    samples = []

    for i in range(n_samples):
        sample_seed = seed + i if seed else None

        # Sample earthquake parameters
        params = sample_earthquake_parameters(
            mw_range=mw_range,
            depth_range_km=depth_range_km,
            location_range_km=grid_extent_km * 0.6,
            fault_type=fault_type,
            seed=sample_seed,
        )

        # Sample noise amplitude
        if sample_seed is not None:
            np.random.seed(sample_seed + 1000)
        noise_amplitude = np.random.uniform(*noise_range_m)

        # Generate time series
        result = generate_timeseries(
            **params,
            grid_extent_km=grid_extent_km,
            grid_spacing_km=grid_spacing_km,
            satellite=satellite,
            orbit=orbit,
            n_pre=n_pre,
            n_event=n_event,
            n_post=n_post,
            noise_amplitude_m=noise_amplitude,
            wrap=wrap,
            output_type=output_type,
            seed=sample_seed,
        )

        samples.append(result)

        if verbose and (i + 1) % max(1, n_samples // 10) == 0:
            print(f"Generated {i + 1}/{n_samples} samples")

    return samples


def batch_to_arrays(
    batch: List[Dict],
    include_metadata: bool = False,
) -> Dict:
    """
    Convert batch of samples to stacked numpy arrays for ML training.

    Parameters
    ----------
    batch : list of dict
        Output from generate_training_batch
    include_metadata : bool
        Include metadata as a list

    Returns
    -------
    data : dict
        Dictionary containing:
        - X: (N, T, H, W) timeseries array
        - y: (N, T, H, W) labels array
        - los: (N, H, W) static LOS displacement
        - metadata: list of metadata dicts (if include_metadata=True)
    """
    X = np.stack([s["timeseries"] for s in batch])
    y = np.stack([s["labels"] for s in batch])
    los = np.stack([s["los_displacement"] for s in batch])

    result = {
        "X": X,
        "y": y,
        "los_displacement": los,
    }

    if include_metadata:
        result["metadata"] = [s["metadata"] for s in batch]

    return result
