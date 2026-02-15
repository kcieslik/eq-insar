# EQ-INSAR

**Earthquake InSAR Synthetic Data Generator**

A lightweight, physics-based forward model for generating synthetic InSAR surface deformation data from earthquake sources. Designed for machine learning training, benchmarking, and sensitivity analysis.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/eq-insar.svg)](https://pypi.org/project/eq-insar/)

---

## Why EQ-INSAR?

Generating realistic synthetic InSAR data for earthquake studies typically requires heavy finite-element codes or full elastic dislocation models. **EQ-INSAR** provides a fast, lightweight alternative using the Davis (1986) point source model — ideal for producing large-scale training datasets for machine learning.

### Key Features

- **Fast Point Source Model** — Davis (1986) elastic half-space model, accurate for small-to-moderate earthquakes (Mw < 6.5)
- **InSAR-Native Outputs** — Line-of-sight displacement, wrapped/unwrapped interferometric phase
- **9 SAR Satellites** — Sentinel-1, ALOS-2, TerraSAR-X, COSMO-SkyMed, RADARSAT-2, NISAR, SAOCOM, ENVISAT, ICEYE
- **ML Training Ready** — Time series with pre/co/post-event frames, binary segmentation labels, batch generation
- **Minimal Dependencies** — Core computation requires only NumPy
- **Export Formats** — GeoTIFF and NetCDF support

## Quick Example

```python
from eq_insar import generate_synthetic_insar

# Mw 6.0 thrust earthquake with Sentinel-1 geometry
result = generate_synthetic_insar(
    Mw=6.0,
    strike_deg=30,
    dip_deg=45,
    rake_deg=90,
    depth_km=10,
    satellite='sentinel1',
    orbit='ascending'
)

los_displacement = result['los_displacement']   # (H, W) array
wrapped_phase = result['wrapped_phase']          # (H, W) array
```

## Physics

EQ-INSAR implements the **Davis (1986)** point source Green's functions for surface displacement in an elastic half-space, combined with:

- **Aki & Richards (2002)** moment tensor convention (strike/dip/rake)
- **Hanks & Kanamori (1979)** moment magnitude scale
- **Wells & Coppersmith (1994)** fault dimension scaling relations

## Installation

```bash
pip install eq-insar
```

See the [Getting Started](getting-started.md) guide for more options.

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

MIT License — see [LICENSE](https://github.com/kcieslik/eq-insar/blob/main/LICENSE) for details.

Developed at the Wroclaw University of Science and Technology & [trainai.io](https://trainai.io).
