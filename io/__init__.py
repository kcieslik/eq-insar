"""
EQ-INSAR I/O: Export synthetic data to common geospatial formats.

Supported formats:
- GeoTIFF: Standard raster format for GIS applications
- NetCDF: Self-describing scientific data format
- NumPy: Native Python array format
"""

from .geotiff import (
    save_geotiff,
    save_displacement_geotiff,
    save_phase_geotiff,
)
from .netcdf import (
    save_netcdf,
    save_timeseries_netcdf,
    load_netcdf,
)

__all__ = [
    # GeoTIFF
    "save_geotiff",
    "save_displacement_geotiff",
    "save_phase_geotiff",
    # NetCDF
    "save_netcdf",
    "save_timeseries_netcdf",
    "load_netcdf",
]
