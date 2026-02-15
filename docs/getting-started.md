# Getting Started

## Installation

### From PyPI

```bash
pip install eq-insar
```

### From Source

```bash
git clone https://github.com/kcieslik/eq-insar.git
cd eq-insar
pip install .
```

### Optional Dependencies

Install extras for additional features:

```bash
# Visualization (matplotlib)
pip install eq-insar[viz]

# GeoTIFF export (rasterio)
pip install eq-insar[geotiff]

# NetCDF export (netCDF4)
pip install eq-insar[netcdf]

# All optional dependencies
pip install eq-insar[all]

# Development (pytest, coverage)
pip install eq-insar[dev]
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

X = result['timeseries']  # (11, height, width)
y = result['labels']      # binary segmentation masks
```

### Batch Generation for ML Pipelines

```python
from eq_insar import generate_training_batch, batch_to_arrays

batch = generate_training_batch(
    n_samples=100,
    Mw_range=(5.0, 7.0),
    satellite='sentinel1',
    seed=42
)

X, y = batch_to_arrays(batch)
# X: (100, T, H, W) - input time series
# y: (100, T, H, W) - segmentation labels
```

## Fault Geometry Convention

EQ-INSAR uses the Aki & Richards (2002) convention:

| Parameter | Range | Description |
|-----------|-------|-------------|
| Strike | 0-360deg | Clockwise from North |
| Dip | 0-90deg | From horizontal |
| Rake | -180 to 180deg | Slip direction |

**Rake angle meanings:**

- **0deg** — Left-lateral strike-slip
- **90deg** — Thrust/reverse
- **+/-180deg** — Right-lateral strike-slip
- **-90deg** — Normal fault

## Supported Satellites

| Satellite | Band | Wavelength | Default Incidence | Agency |
|-----------|------|------------|-------------------|--------|
| Sentinel-1 | C | 5.5 cm | 33deg | ESA |
| ALOS-2 | L | 22.9 cm | 35deg | JAXA |
| TerraSAR-X | X | 3.1 cm | 35deg | DLR |
| COSMO-SkyMed | X | 3.1 cm | 35deg | ASI |
| RADARSAT-2 | C | 5.5 cm | 35deg | CSA |
| NISAR | L | 23.8 cm | 35deg | NASA/ISRO |
| SAOCOM | L | 23.5 cm | 35deg | CONAE |
| ENVISAT | C | 5.6 cm | 23deg | ESA |
| ICEYE | X | 3.1 cm | 30deg | ICEYE |

```python
from eq_insar import list_satellites, get_satellite

# List all satellites
print(list_satellites())

# Get specific satellite configuration
sentinel1 = get_satellite('sentinel1')
print(f"Wavelength: {sentinel1.wavelength_m * 100:.2f} cm")
```

## Next Steps

- Browse the [API Reference](api/generators.md) for detailed function documentation
- Check out the [Examples](examples.md) for Jupyter notebooks
