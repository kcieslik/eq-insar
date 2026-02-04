# EQ-INSAR

**Earthquake InSAR Synthetic Data Generator**

A lightweight, physics-based forward model for generating synthetic InSAR surface deformation data from earthquake sources. Designed for machine learning training, benchmarking, and sensitivity analysis.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Davis (1986) Point Source Model**
  - Fast computation for generating large training datasets
  - Accurate for small-to-moderate earthquakes (Mw < 6.5)
  - Suitable for far-field observations

- **InSAR-Native Outputs**
  - Line-of-sight (LOS) displacement
  - Wrapped/unwrapped interferometric phase

- **Multiple Satellites**
  - Sentinel-1, ALOS-2, TerraSAR-X, COSMO-SkyMed, RADARSAT-2, NISAR, SAOCOM, ENVISAT, ICEYE
  - Automatic geometry (incidence, heading, wavelength)

- **Simple Noise Models**
  - Gaussian random noise
  - Orbital ramps

- **ML Training Ready**
  - Time series with pre-event/event/post-event frames
  - Binary segmentation labels
  - Batch generation with random parameters

- **Export Formats**
  - GeoTIFF (requires `rasterio`)
  - NetCDF (requires `netCDF4`)

- **Minimal Dependencies**
  - NumPy only for core computation
  - No SciPy requirement

## Requirements

- Python 3.8 or higher
- NumPy >= 1.20.0

Optional dependencies:
- `matplotlib` >= 3.3.0 (visualization)
- `rasterio` >= 1.2.0 (GeoTIFF export)
- `netCDF4` >= 1.5.0 (NetCDF export)

## Installation

### From PyPI

```bash
pip install eq-insar #TODO for now please install from source
```

### From Source

```bash
git clone https://github.com/kcieslik/eq-insar.git
cd eq-insar
pip install .
```

### Development Installation

For development with editable install:

```bash
git clone https://github.com/kcieslik/eq-insar.git
cd eq-insar
pip install -e ".[dev]"
```

### Optional Dependencies

Install optional features using extras:

```bash
# Visualization (matplotlib)
pip install eq-insar[viz]

# GeoTIFF export (rasterio)
pip install eq-insar[geotiff]

# NetCDF export (netCDF4)
pip install eq-insar[netcdf]

# All I/O formats (rasterio + netCDF4)
pip install eq-insar[io]

# Everything (all optional dependencies)
pip install eq-insar[all]

# Development (pytest, coverage)
pip install eq-insar[dev]
```

Or install individual packages manually:

```bash
pip install matplotlib    # For visualization
pip install rasterio      # For GeoTIFF export
pip install netCDF4       # For NetCDF export
```

## Quick Start

### Generate a Single Interferogram

```python
from eq_insar import generate_synthetic_insar, plot_displacement_components

# Mw 6.0 thrust earthquake with Sentinel-1 geometry
result = generate_synthetic_insar(
    Mw=6.0,
    strike_deg=30,
    dip_deg=45,
    rake_deg=90,          # thrust fault
    depth_km=10,
    satellite='sentinel1',
    orbit='ascending'
)

# Access the data
los_displacement = result['los_displacement']  # (height, width) array
wrapped_phase = result['wrapped_phase']        # (height, width) array
unwrapped_phase = result['unwrapped_phase']    # (height, width) array

# Visualize (requires matplotlib)
fig = plot_displacement_components(result)
```

### Generate Time Series for ML Training

```python
from eq_insar import generate_timeseries

result = generate_timeseries(
    Mw=6.0,
    satellite='sentinel1',
    n_pre=5,      # pre-event frames (noise only)
    n_event=1,    # event frames (signal + noise)
    n_post=5      # post-event frames (noise only)
)

# Access data
X = result['timeseries']  # (11, height, width)
y = result['labels']      # binary segmentation masks
```

### Batch Generation for ML Pipelines

```python
from eq_insar import generate_training_batch, batch_to_arrays

# Generate batch with randomized parameters
batch = generate_training_batch(
    n_samples=100,
    Mw_range=(5.0, 7.0),
    satellite='sentinel1',
    seed=42  # for reproducibility
)

# Convert to stacked arrays for PyTorch/TensorFlow
X, y = batch_to_arrays(batch)
# X: (100, T, H, W) - input time series
# y: (100, T, H, W) - segmentation labels
```

### Custom Earthquake Parameters

```python
from eq_insar import sample_earthquake_parameters, generate_synthetic_insar

# Sample random earthquake parameters
params = sample_earthquake_parameters(
    Mw_range=(5.5, 6.5),
    depth_range=(5, 20),
    seed=42
)

# Generate interferogram with sampled parameters
result = generate_synthetic_insar(**params, satellite='sentinel1')
```

## API Reference

### Main Functions

| Function | Description |
|----------|-------------|
| `generate_synthetic_insar()` | Generate a single interferogram |
| `generate_timeseries()` | Generate time series with pre/co/post-event frames |
| `generate_training_batch()` | Generate multiple samples with random parameters |
| `sample_earthquake_parameters()` | Sample random earthquake parameters |
| `batch_to_arrays()` | Convert batch to stacked NumPy arrays |

### Satellite Functions

| Function | Description |
|----------|-------------|
| `list_satellites()` | List all available satellite configurations |
| `get_satellite(name)` | Get configuration for a specific satellite |
| `SatelliteConfig` | Dataclass for satellite parameters |

### Core Physics Functions

| Function | Description |
|----------|-------------|
| `mw_to_m0()` | Convert moment magnitude to seismic moment |
| `m0_to_mw()` | Convert seismic moment to moment magnitude |
| `double_couple_moment_tensor()` | Create moment tensor from strike/dip/rake |
| `slip_from_moment()` | Calculate slip from seismic moment |
| `davis_point_source()` | Compute displacement using Davis (1986) model |

### InSAR Functions

| Function | Description |
|----------|-------------|
| `compute_los_vector()` | Compute LOS unit vector from geometry |
| `compute_los_displacement()` | Project 3D displacement to LOS |
| `displacement_to_phase()` | Convert displacement to interferometric phase |
| `wrap_phase()` | Wrap phase to [-pi, pi] |
| `phase_to_displacement()` | Convert phase back to displacement |
| `generate_random_noise()` | Generate Gaussian noise |
| `generate_orbital_ramp()` | Generate orbital ramp artifacts |

### I/O Functions

| Function | Description |
|----------|-------------|
| `save_geotiff()` | Save array as GeoTIFF |
| `save_displacement_geotiff()` | Save displacement components as GeoTIFFs |
| `save_phase_geotiff()` | Save phase data as GeoTIFF |
| `save_netcdf()` | Save single interferogram as NetCDF |
| `save_timeseries_netcdf()` | Save time series as NetCDF |
| `load_netcdf()` | Load data from NetCDF |

### Visualization Functions

| Function | Description |
|----------|-------------|
| `plot_displacement_components()` | Plot E/N/U displacement and LOS |
| `plot_insar_products()` | Plot wrapped/unwrapped phase |
| `plot_timeseries_frames()` | Plot time series frames |
| `plot_timeseries_statistics()` | Plot time series statistics |
| `plot_timeseries_at_points()` | Plot time series at specific locations |
| `plot_timeseries_displacement_components()` | Plot displacement components over time |
| `plot_timeseries_profile()` | Plot displacement profiles |
| `plot_timeseries_difference()` | Plot differences between frames |

## Supported Satellites

| Satellite | Band | Wavelength | Default Incidence | Agency |
|-----------|------|------------|-------------------|--------|
| Sentinel-1 | C | 5.5 cm | 33° | ESA |
| ALOS-2 | L | 22.9 cm | 35° | JAXA |
| TerraSAR-X | X | 3.1 cm | 35° | DLR |
| COSMO-SkyMed | X | 3.1 cm | 35° | ASI |
| RADARSAT-2 | C | 5.5 cm | 35° | CSA |
| NISAR | L | 23.8 cm | 35° | NASA/ISRO |
| SAOCOM | L | 23.5 cm | 35° | CONAE |
| ENVISAT | C | 5.6 cm | 23° | ESA |
| ICEYE | X | 3.1 cm | 30° | ICEYE |

```python
from eq_insar import list_satellites, get_satellite

# List all satellites
print(list_satellites())

# Get specific satellite configuration
sentinel1 = get_satellite('sentinel1')
print(f"Wavelength: {sentinel1.wavelength_m * 100:.2f} cm")
```

## Fault Geometry Convention

Uses Aki & Richards (2002) convention:

| Parameter | Range | Description |
|-----------|-------|-------------|
| Strike | 0-360° | Clockwise from North |
| Dip | 0-90° | From horizontal |
| Rake | -180 to 180° | Slip direction |

**Rake angle meanings:**
- 0°: Left-lateral strike-slip
- 90°: Thrust/reverse
- ±180°: Right-lateral strike-slip
- -90°: Normal fault

## Export Data

### GeoTIFF (requires rasterio)

```python
from eq_insar import generate_synthetic_insar, save_displacement_geotiff

result = generate_synthetic_insar(Mw=6.0, satellite='sentinel1')
save_displacement_geotiff(result, 'output/', prefix='eq_mw60')
# Creates: eq_mw60_east.tif, eq_mw60_north.tif, eq_mw60_up.tif, eq_mw60_los.tif
```

### NetCDF

```python
from eq_insar import save_netcdf, save_timeseries_netcdf, load_netcdf

# Single interferogram
result = generate_synthetic_insar(Mw=6.0, satellite='sentinel1')
save_netcdf(result, 'output/interferogram.nc')

# Time series
result_ts = generate_timeseries(Mw=6.0, satellite='sentinel1')
save_timeseries_netcdf(result_ts, 'output/timeseries.nc')

# Load back
data = load_netcdf('output/interferogram.nc')
```

## Examples

See the `examples/` directory for Jupyter notebooks:

- **[showcase.ipynb](examples/showcase.ipynb)**: Comprehensive tutorial covering:
  - Single interferogram generation
  - Fault type comparison (thrust, normal, strike-slip)
  - Magnitude scaling effects
  - Satellite comparison (C-band, L-band, X-band)
  - Noise effects visualization
  - Time series for ML training
  - Batch generation
  - Publication-quality figures

## Package Structure

```
eq-insar/
├── pyproject.toml              # Package configuration
├── README.md                   # This file
├── LICENSE                     # MIT License
├── src/
│   └── eq_insar/               # Main package
│       ├── __init__.py         # Public API with lazy imports
│       ├── constants.py        # Satellite configs, physical constants
│       ├── core/               # Seismic physics
│       │   ├── __init__.py
│       │   ├── davis.py        # Davis (1986) point source model
│       │   └── moment_tensor.py# Moment tensor, magnitude conversion
│       ├── generators/         # Synthetic data generation
│       │   ├── __init__.py
│       │   ├── single.py       # Single interferogram
│       │   ├── timeseries.py   # Time series generation
│       │   └── batch.py        # Batch generation for ML
│       ├── insar/              # InSAR processing
│       │   ├── __init__.py
│       │   ├── projection.py   # LOS projection, phase conversion
│       │   └── noise.py        # Noise models
│       ├── io/                 # Data export
│       │   ├── __init__.py
│       │   ├── geotiff.py      # GeoTIFF export
│       │   └── netcdf.py       # NetCDF export/load
│       └── visualization/      # Plotting functions
│           ├── __init__.py
│           ├── displacement.py # Displacement plots
│           └── timeseries.py   # Time series plots
├── tests/                      # Test suite
│   ├── __init__.py
│   └── test_generators.py      # Unit tests
└── examples/                   # Example notebooks
    └── showcase.ipynb          # Tutorial notebook
```

## Development

### Running Tests

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=eq_insar --cov-report=html

# Run specific test
pytest tests/test_generators.py::TestGenerateSyntheticInsar::test_basic_generation
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Make your changes
4. Run tests (`pytest tests/`)
5. Commit your changes (`git commit -am 'Add new feature'`)
6. Push to the branch (`git push origin feature/new-feature`)
7. Create a Pull Request

## Physics References

- **Davis, P.M. (1986)**. Surface deformation due to a dipping hydrofracture. *Journal of Geophysical Research*
- **Aki, K. & Richards, P.G. (2002)**. *Quantitative Seismology*, 2nd ed. University Science Books
- **Hanks, T.C. & Kanamori, H. (1979)**. A moment magnitude scale. *Journal of Geophysical Research*
- **Wells, D.L. & Coppersmith, K.J. (1994)**. New empirical relationships among magnitude, rupture length, rupture width, rupture area, and surface displacement. *Bulletin of the Seismological Society of America*

## Citation

If you use EQ-INSAR in your research, please cite:

```bibtex
@software{cieslik2026eqinsar,
  author = {Cieslik, Konrad and Milczarek, Wojciech},
  title = {EQ-INSAR: A Python Package for Generating Synthetic Earthquake InSAR Deformation Data},
  year = {2026},
  url = {https://github.com/kcieslik/eq-insar}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

This package was developed at the Wroclaw University of Science and Technology.
