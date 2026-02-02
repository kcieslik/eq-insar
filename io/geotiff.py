"""
GeoTIFF export functions for EQ-INSAR.

Provides functions to export synthetic InSAR data to GeoTIFF format,
compatible with GIS software (QGIS, ArcGIS) and InSAR processing tools.

Note: Requires rasterio or GDAL. Functions gracefully handle missing
dependencies with informative error messages.
"""

import numpy as np
from typing import Dict, Optional, Union, Tuple
from pathlib import Path


def _check_rasterio():
    """Check if rasterio is available."""
    try:
        import rasterio
        return True
    except ImportError:
        return False


def save_geotiff(
    data: np.ndarray,
    filepath: Union[str, Path],
    bounds_km: Optional[Tuple[float, float, float, float]] = None,
    crs: str = "EPSG:32633",  # UTM zone 33N as default
    pixel_size_km: float = 0.5,
    nodata: float = np.nan,
    description: str = "",
    metadata: Optional[Dict] = None,
) -> None:
    """
    Save a 2D array to GeoTIFF format.

    Parameters
    ----------
    data : np.ndarray
        2D array to save (ny, nx)
    filepath : str or Path
        Output file path
    bounds_km : tuple, optional
        (xmin, ymin, xmax, ymax) in km. If None, centers data at origin.
    crs : str
        Coordinate reference system (default: UTM 33N)
    pixel_size_km : float
        Pixel size in km
    nodata : float
        NoData value
    description : str
        Band description
    metadata : dict, optional
        Additional metadata to include

    Raises
    ------
    ImportError
        If rasterio is not installed
    """
    if not _check_rasterio():
        raise ImportError(
            "rasterio is required for GeoTIFF export. "
            "Install with: pip install rasterio"
        )

    import rasterio
    from rasterio.transform import from_bounds

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    ny, nx = data.shape

    # Calculate bounds if not provided
    if bounds_km is None:
        half_x = nx * pixel_size_km / 2
        half_y = ny * pixel_size_km / 2
        bounds_km = (-half_x, -half_y, half_x, half_y)

    # Convert km to meters for geotransform
    xmin, ymin, xmax, ymax = [b * 1000 for b in bounds_km]

    # Create transform
    transform = from_bounds(xmin, ymin, xmax, ymax, nx, ny)

    # Prepare metadata
    profile = {
        "driver": "GTiff",
        "dtype": data.dtype,
        "width": nx,
        "height": ny,
        "count": 1,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "lzw",
    }

    # Write file
    with rasterio.open(filepath, "w", **profile) as dst:
        dst.write(data, 1)
        if description:
            dst.set_band_description(1, description)

        # Add custom metadata
        if metadata:
            dst.update_tags(**{str(k): str(v) for k, v in metadata.items()})


def save_displacement_geotiff(
    result: Dict,
    output_dir: Union[str, Path],
    prefix: str = "eq_insar",
    include_components: bool = True,
) -> Dict[str, Path]:
    """
    Save displacement fields from EQ-INSAR result to GeoTIFF files.

    Parameters
    ----------
    result : dict
        Output from generate_synthetic_insar or generate_timeseries
    output_dir : str or Path
        Output directory
    prefix : str
        Filename prefix
    include_components : bool
        If True, save Ue, Un, Uz components in addition to LOS

    Returns
    -------
    files : dict
        Dictionary mapping field names to output file paths
    """
    if not _check_rasterio():
        raise ImportError(
            "rasterio is required for GeoTIFF export. "
            "Install with: pip install rasterio"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {}
    meta = result.get("metadata", {})

    # Get bounds from coordinate grids
    X_km = result["X_km"]
    Y_km = result["Y_km"]
    bounds_km = (X_km.min(), Y_km.min(), X_km.max(), Y_km.max())
    pixel_size = meta.get("grid_spacing_km", 0.5)

    # Base metadata
    base_meta = {
        "source": "EQ-INSAR",
        "Mw": meta.get("Mw", "N/A"),
        "depth_km": meta.get("depth_km", "N/A"),
        "strike_deg": meta.get("strike_deg", "N/A"),
        "dip_deg": meta.get("dip_deg", "N/A"),
        "rake_deg": meta.get("rake_deg", "N/A"),
    }

    # Save LOS displacement
    filepath = output_dir / f"{prefix}_los_displacement_m.tif"
    save_geotiff(
        result["los_displacement"].astype(np.float32),
        filepath,
        bounds_km=bounds_km,
        pixel_size_km=pixel_size,
        description="LOS displacement (m, positive toward satellite)",
        metadata=base_meta,
    )
    files["los_displacement"] = filepath

    # Save displacement components
    if include_components:
        for comp, desc in [
            ("Ue", "East displacement (m)"),
            ("Un", "North displacement (m)"),
            ("Uz", "Vertical displacement (m, positive up)"),
        ]:
            if comp in result:
                filepath = output_dir / f"{prefix}_{comp}_m.tif"
                save_geotiff(
                    result[comp].astype(np.float32),
                    filepath,
                    bounds_km=bounds_km,
                    pixel_size_km=pixel_size,
                    description=desc,
                    metadata=base_meta,
                )
                files[comp] = filepath

    return files


def save_phase_geotiff(
    result: Dict,
    output_dir: Union[str, Path],
    prefix: str = "eq_insar",
    include_all: bool = True,
) -> Dict[str, Path]:
    """
    Save phase fields from EQ-INSAR result to GeoTIFF files.

    Parameters
    ----------
    result : dict
        Output from generate_synthetic_insar
    output_dir : str or Path
        Output directory
    prefix : str
        Filename prefix
    include_all : bool
        If True, save unwrapped, wrapped, and noisy phases

    Returns
    -------
    files : dict
        Dictionary mapping field names to output file paths
    """
    if not _check_rasterio():
        raise ImportError(
            "rasterio is required for GeoTIFF export. "
            "Install with: pip install rasterio"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {}
    meta = result.get("metadata", {})

    X_km = result["X_km"]
    Y_km = result["Y_km"]
    bounds_km = (X_km.min(), Y_km.min(), X_km.max(), Y_km.max())
    pixel_size = meta.get("grid_spacing_km", 0.5)

    base_meta = {
        "source": "EQ-INSAR",
        "wavelength_m": meta.get("wavelength_m", "N/A"),
    }

    phase_fields = [
        ("phase_wrapped", "Wrapped phase (rad)"),
    ]

    if include_all:
        phase_fields.extend([
            ("phase_unwrapped", "Unwrapped phase (rad)"),
            ("phase_noisy", "Wrapped phase with noise (rad)"),
        ])

    for field, desc in phase_fields:
        if field in result:
            filepath = output_dir / f"{prefix}_{field}.tif"
            save_geotiff(
                result[field].astype(np.float32),
                filepath,
                bounds_km=bounds_km,
                pixel_size_km=pixel_size,
                description=desc,
                metadata=base_meta,
            )
            files[field] = filepath

    return files
