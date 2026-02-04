"""
EQ-INSAR Visualization: Plotting functions for synthetic InSAR data.

Provides publication-quality visualization functions for:
- Displacement components (East, North, Up, LOS)
- InSAR products (wrapped phase)
- Time series analysis and animation
"""

from .displacement import (
    plot_displacement_components,
    plot_insar_products,
)
from .timeseries import (
    plot_timeseries_frames,
    plot_timeseries_statistics,
    plot_timeseries_at_points,
    plot_timeseries_displacement_components,
    plot_timeseries_profile,
    plot_timeseries_difference,
)

__all__ = [
    # Displacement plots
    "plot_displacement_components",
    "plot_insar_products",
    # Time series plots
    "plot_timeseries_frames",
    "plot_timeseries_statistics",
    "plot_timeseries_at_points",
    "plot_timeseries_displacement_components",
    "plot_timeseries_profile",
    "plot_timeseries_difference",
]
