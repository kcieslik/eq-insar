"""
EQ-INSAR: Earthquake InSAR Synthetic Data Generator (Lite)
==========================================================

A lightweight, physics-based forward model for generating synthetic InSAR
deformation data from earthquake sources. Designed for machine learning
training, benchmarking, and sensitivity analysis.

Features
--------
- **Davis (1986) point source model**: Fast, accurate for small-to-moderate earthquakes
- **InSAR-native outputs**: LOS displacement, wrapped/unwrapped phase
- **Multiple satellites**: Sentinel-1, ALOS-2, TerraSAR-X, COSMO-SkyMed, etc.
- **Simple noise model**: Random Gaussian noise, orbital ramps
- **ML-ready**: Time series with pre/event/post frames, batch generation
- **Export formats**: GeoTIFF and NetCDF support

Quick Start
-----------
>>> from eq_insar import generate_synthetic_insar
>>>
>>> result = generate_synthetic_insar(
...     Mw=6.0,
...     strike_deg=30, dip_deg=45, rake_deg=90,
...     depth_km=10,
...     satellite='sentinel1',
...     orbit='ascending'
... )

For ML training data:
>>> from eq_insar import generate_timeseries, generate_training_batch
>>>
>>> result = generate_timeseries(Mw=6.0, n_pre=5, n_event=1, n_post=5)
>>> batch = generate_training_batch(n_samples=100, satellite='sentinel1')

Source Model
------------
**Davis (1986)**: Point source approximation in elastic half-space, suitable for
small-to-moderate earthquakes (Mw < 6.5) or far-field observations. Fast computation
makes it ideal for generating large ML training datasets.

References
----------
- Davis, P.M. (1986). Surface deformation associated with dipping hydrofracture. JGR.
- Aki, K. & Richards, P.G. (2002). Quantitative Seismology.
- Hanks, T.C. & Kanamori, H. (1979). A moment magnitude scale. JGR.

Author
------
Konrad Cieslik

License
-------
MIT License
"""

__version__ = "0.1.0"
__author__ = "Konrad Cieslik"

# =============================================================================
# Primary API - Main generators
# =============================================================================

from .generators import (
    generate_synthetic_insar,
    generate_timeseries,
    generate_training_batch,
    sample_earthquake_parameters,
    batch_to_arrays,
)

# =============================================================================
# Visualization (lazy imports to avoid matplotlib issues)
# =============================================================================

def __getattr__(name):
    """Lazy import for visualization functions to avoid matplotlib dependency at import time."""
    _viz_functions = {
        'plot_displacement_components',
        'plot_insar_products',
        'plot_timeseries_frames',
        'plot_timeseries_statistics',
        'plot_timeseries_at_points',
        'plot_timeseries_displacement_components',
        'plot_timeseries_profile',
        'plot_timeseries_difference',
    }
    if name in _viz_functions:
        from . import visualization
        return getattr(visualization, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# =============================================================================
# Satellite configurations
# =============================================================================

from .constants import (
    get_satellite,
    list_satellites,
    SatelliteConfig,
    SATELLITES,
)

# =============================================================================
# Core physics (for advanced users)
# =============================================================================

from .core import (
    # Moment tensor
    mw_to_m0,
    m0_to_mw,
    double_couple_moment_tensor,
    slip_from_moment,
    # Source models
    davis_point_source,
)

# =============================================================================
# InSAR functions (for advanced users)
# =============================================================================

from .insar import (
    # Projection
    compute_los_displacement,
    compute_los_vector,
    displacement_to_phase,
    wrap_phase,
    phase_to_displacement,
    # Noise models
    generate_random_noise,
    generate_orbital_ramp,
)

# =============================================================================
# I/O functions
# =============================================================================

from .io import (
    save_geotiff,
    save_displacement_geotiff,
    save_phase_geotiff,
    save_netcdf,
    save_timeseries_netcdf,
    load_netcdf,
)

# =============================================================================
# Constants
# =============================================================================

from .constants import (
    # Wavelengths
    SENTINEL1_WAVELENGTH_M,
    ALOS2_WAVELENGTH_M,
    TERRASAR_WAVELENGTH_M,
    COSMO_WAVELENGTH_M,
    # Elastic parameters
    DEFAULT_SHEAR_MODULUS_PA,
    DEFAULT_POISSON_RATIO,
)

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Version
    "__version__",
    "__author__",
    # Main generators (PRIMARY API)
    "generate_synthetic_insar",
    "generate_timeseries",
    "generate_training_batch",
    "sample_earthquake_parameters",
    "batch_to_arrays",
    # Visualization
    "plot_displacement_components",
    "plot_insar_products",
    "plot_timeseries_frames",
    "plot_timeseries_statistics",
    "plot_timeseries_at_points",
    "plot_timeseries_displacement_components",
    "plot_timeseries_profile",
    "plot_timeseries_difference",
    # Satellite configurations
    "get_satellite",
    "list_satellites",
    "SatelliteConfig",
    "SATELLITES",
    # Core physics
    "mw_to_m0",
    "m0_to_mw",
    "double_couple_moment_tensor",
    "slip_from_moment",
    "davis_point_source",
    # InSAR functions
    "compute_los_displacement",
    "compute_los_vector",
    "displacement_to_phase",
    "wrap_phase",
    "phase_to_displacement",
    "generate_random_noise",
    "generate_orbital_ramp",
    # I/O
    "save_geotiff",
    "save_displacement_geotiff",
    "save_phase_geotiff",
    "save_netcdf",
    "save_timeseries_netcdf",
    "load_netcdf",
    # Constants
    "SENTINEL1_WAVELENGTH_M",
    "ALOS2_WAVELENGTH_M",
    "TERRASAR_WAVELENGTH_M",
    "COSMO_WAVELENGTH_M",
    "DEFAULT_SHEAR_MODULUS_PA",
    "DEFAULT_POISSON_RATIO",
]
