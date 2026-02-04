"""
NetCDF export functions for EQ-INSAR.

NetCDF (Network Common Data Form) is a self-describing scientific data
format widely used in Earth sciences. It supports multi-dimensional arrays,
metadata, and compression.

Note: Requires netCDF4 or xarray. Functions gracefully handle missing
dependencies with informative error messages.
"""

import numpy as np
from typing import Dict, Optional, Union, List
from pathlib import Path
from datetime import datetime


def _check_netcdf4():
    """Check if netCDF4 is available."""
    try:
        import netCDF4
        return True
    except ImportError:
        return False


def save_netcdf(
    result: Dict,
    filepath: Union[str, Path],
    include_metadata: bool = True,
    compression_level: int = 4,
) -> None:
    """
    Save EQ-INSAR result to NetCDF format.

    Creates a CF-compliant NetCDF file with all displacement and
    phase fields, plus metadata.

    Parameters
    ----------
    result : dict
        Output from generate_synthetic_insar
    filepath : str or Path
        Output file path
    include_metadata : bool
        Include source parameters as global attributes
    compression_level : int
        Compression level (0-9, higher = smaller files)

    Raises
    ------
    ImportError
        If netCDF4 is not installed
    """
    if not _check_netcdf4():
        raise ImportError(
            "netCDF4 is required for NetCDF export. "
            "Install with: pip install netCDF4"
        )

    import netCDF4

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    X_km = result["X_km"]
    Y_km = result["Y_km"]
    meta = result.get("metadata", {})

    ny, nx = X_km.shape

    # Create file
    with netCDF4.Dataset(filepath, "w", format="NETCDF4") as nc:
        # Global attributes
        nc.title = "EQ-INSAR Synthetic InSAR Data"
        nc.institution = "EQ-INSAR Package"
        nc.source = "Synthetic earthquake deformation model"
        nc.history = f"Created {datetime.now().isoformat()}"
        nc.Conventions = "CF-1.8"

        if include_metadata:
            nc.earthquake_Mw = meta.get("Mw", np.nan)
            nc.earthquake_M0_Nm = meta.get("M0_Nm", np.nan)
            nc.earthquake_depth_km = meta.get("depth_km", np.nan)
            nc.earthquake_strike_deg = meta.get("strike_deg", np.nan)
            nc.earthquake_dip_deg = meta.get("dip_deg", np.nan)
            nc.earthquake_rake_deg = meta.get("rake_deg", np.nan)
            nc.earthquake_xcen_km = meta.get("xcen_km", np.nan)
            nc.earthquake_ycen_km = meta.get("ycen_km", np.nan)
            nc.source_type = meta.get("source_type", "unknown")
            nc.satellite = meta.get("satellite", "unknown")
            nc.wavelength_m = meta.get("wavelength_m", np.nan)
            nc.incidence_deg = meta.get("incidence_deg", np.nan)
            nc.heading_deg = meta.get("heading_deg", np.nan)

        # Create dimensions
        nc.createDimension("y", ny)
        nc.createDimension("x", nx)

        # Create coordinate variables
        x_var = nc.createVariable("x", "f4", ("x",))
        x_var.units = "km"
        x_var.long_name = "Easting"
        x_var.standard_name = "projection_x_coordinate"
        x_var[:] = X_km[0, :]

        y_var = nc.createVariable("y", "f4", ("y",))
        y_var.units = "km"
        y_var.long_name = "Northing"
        y_var.standard_name = "projection_y_coordinate"
        y_var[:] = Y_km[:, 0]

        # Compression settings
        comp = {"zlib": True, "complevel": compression_level}

        # Data variables
        variables = [
            ("los_displacement", "m", "LOS displacement", "positive toward satellite"),
            ("Ue", "m", "East displacement", "positive eastward"),
            ("Un", "m", "North displacement", "positive northward"),
            ("Uz", "m", "Vertical displacement", "positive upward"),
            ("phase_unwrapped", "rad", "Unwrapped interferometric phase", ""),
            ("phase_wrapped", "rad", "Wrapped interferometric phase", ""),
            ("phase_noisy", "rad", "Wrapped phase with noise", ""),
        ]

        for name, units, long_name, comment in variables:
            if name in result:
                var = nc.createVariable(name, "f4", ("y", "x"), **comp)
                var.units = units
                var.long_name = long_name
                if comment:
                    var.comment = comment
                var[:] = result[name]


def save_timeseries_netcdf(
    result: Dict,
    filepath: Union[str, Path],
    include_labels: bool = True,
    compression_level: int = 4,
) -> None:
    """
    Save EQ-INSAR time series to NetCDF format.

    Creates a NetCDF file with the time dimension for time series data,
    suitable for ML training datasets.

    Parameters
    ----------
    result : dict
        Output from generate_timeseries
    filepath : str or Path
        Output file path
    include_labels : bool
        Include segmentation labels
    compression_level : int
        Compression level (0-9)
    """
    if not _check_netcdf4():
        raise ImportError(
            "netCDF4 is required for NetCDF export. "
            "Install with: pip install netCDF4"
        )

    import netCDF4

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    X_km = result["X_km"]
    Y_km = result["Y_km"]
    meta = result.get("metadata", {})
    timeseries = result["timeseries"]

    nt, ny, nx = timeseries.shape
    n_pre = meta.get("n_pre", 0)
    n_event = meta.get("n_event", 1)
    n_post = meta.get("n_post", 0)

    with netCDF4.Dataset(filepath, "w", format="NETCDF4") as nc:
        # Global attributes
        nc.title = "EQ-INSAR Synthetic InSAR Time Series"
        nc.institution = "EQ-INSAR Package"
        nc.source = "Synthetic earthquake deformation model"
        nc.history = f"Created {datetime.now().isoformat()}"
        nc.Conventions = "CF-1.8"

        # Time series metadata
        nc.n_pre_event = n_pre
        nc.n_event = n_event
        nc.n_post_event = n_post
        nc.output_type = meta.get("output_type", "phase")

        # Source parameters
        nc.earthquake_Mw = meta.get("Mw", np.nan)
        nc.earthquake_depth_km = meta.get("depth_km", np.nan)
        nc.earthquake_strike_deg = meta.get("strike_deg", np.nan)
        nc.earthquake_dip_deg = meta.get("dip_deg", np.nan)
        nc.earthquake_rake_deg = meta.get("rake_deg", np.nan)

        # Create dimensions
        nc.createDimension("time", nt)
        nc.createDimension("y", ny)
        nc.createDimension("x", nx)

        # Coordinates
        time_var = nc.createVariable("time", "i4", ("time",))
        time_var.units = "frame number"
        time_var.long_name = "Time frame index"
        time_var[:] = np.arange(nt)

        x_var = nc.createVariable("x", "f4", ("x",))
        x_var.units = "km"
        x_var.long_name = "Easting"
        x_var[:] = X_km[0, :]

        y_var = nc.createVariable("y", "f4", ("y",))
        y_var.units = "km"
        y_var.long_name = "Northing"
        y_var[:] = Y_km[:, 0]

        # Frame type indicator
        frame_type = nc.createVariable("frame_type", "i1", ("time",))
        frame_type.long_name = "Frame type (0=pre, 1=event, 2=post)"
        frame_type.flag_values = np.array([0, 1, 2], dtype=np.int8)
        frame_type.flag_meanings = "pre_event event post_event"
        types = np.zeros(nt, dtype=np.int8)
        types[n_pre:n_pre + n_event] = 1
        types[n_pre + n_event:] = 2
        frame_type[:] = types

        # Compression settings
        comp = {"zlib": True, "complevel": compression_level}

        # Time series data
        ts_var = nc.createVariable("timeseries", "f4", ("time", "y", "x"), **comp)
        ts_var.units = "rad" if meta.get("output_type") == "phase" else "m"
        ts_var.long_name = "InSAR time series"
        ts_var[:] = timeseries

        # Labels
        if include_labels and "labels" in result:
            label_var = nc.createVariable("labels", "i1", ("time", "y", "x"), **comp)
            label_var.long_name = "Deformation mask (1=deformation, 0=no deformation)"
            label_var[:] = result["labels"]

        # Static fields (not time-varying)
        if "los_displacement" in result:
            los_var = nc.createVariable("los_displacement", "f4", ("y", "x"), **comp)
            los_var.units = "m"
            los_var.long_name = "LOS displacement (static, no noise)"
            los_var[:] = result["los_displacement"]


def load_netcdf(filepath: Union[str, Path]) -> Dict:
    """
    Load EQ-INSAR data from NetCDF file.

    Parameters
    ----------
    filepath : str or Path
        Path to NetCDF file

    Returns
    -------
    result : dict
        Dictionary with data arrays and metadata
    """
    if not _check_netcdf4():
        raise ImportError(
            "netCDF4 is required for NetCDF import. "
            "Install with: pip install netCDF4"
        )

    import netCDF4

    filepath = Path(filepath)

    result = {}
    metadata = {}

    with netCDF4.Dataset(filepath, "r") as nc:
        # Load coordinates
        x_km = nc.variables["x"][:]
        y_km = nc.variables["y"][:]
        X_km, Y_km = np.meshgrid(x_km, y_km)
        result["X_km"] = X_km
        result["Y_km"] = Y_km

        # Load data variables
        for var_name in nc.variables:
            if var_name not in ["x", "y", "time", "frame_type"]:
                result[var_name] = nc.variables[var_name][:]

        # Load global attributes as metadata
        for attr in nc.ncattrs():
            value = nc.getncattr(attr)
            if attr.startswith("earthquake_"):
                key = attr.replace("earthquake_", "")
                metadata[key] = value
            else:
                metadata[attr] = value

    result["metadata"] = metadata
    return result
