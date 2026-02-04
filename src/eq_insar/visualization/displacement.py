"""
Visualization functions for displacement fields and InSAR products.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple


def plot_displacement_components(
    result: Dict,
    title_prefix: str = "",
    figsize: Tuple[float, float] = (14, 12),
    cmap: str = "RdBu_r",
    show_epicenter: bool = True,
) -> plt.Figure:
    """
    Plot all three displacement components and LOS.

    Creates a 2x2 grid showing East, North, Up, and LOS displacement.

    Parameters
    ----------
    result : dict
        Output from generate_synthetic_insar or generate_timeseries
    title_prefix : str
        Prefix for the plot title
    figsize : tuple
        Figure size in inches
    cmap : str
        Colormap for displacement
    show_epicenter : bool
        Show epicenter location as star

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    X = result["X_km"]
    Y = result["Y_km"]
    meta = result["metadata"]

    components = [
        ("Ue", "East displacement (cm)"),
        ("Un", "North displacement (cm)"),
        ("Uz", "Vertical displacement (cm)"),
        ("los_displacement", "LOS displacement (cm)"),
    ]

    for ax, (key, label) in zip(axes.flat, components):
        data = result[key] * 100  # Convert to cm
        vmax = np.percentile(np.abs(data), 99)
        vmax = max(vmax, 0.1)  # Minimum scale

        im = ax.contourf(
            X, Y, data, levels=50, cmap=cmap, vmin=-vmax, vmax=vmax
        )
        plt.colorbar(im, ax=ax, label=label)
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        ax.set_aspect("equal")

        if show_epicenter:
            ax.plot(meta["xcen_km"], meta["ycen_km"], "k*", markersize=12)

    # Build title
    title_parts = [title_prefix] if title_prefix else []
    title_parts.append(f"Mw={meta['Mw']:.1f}")
    title_parts.append(f"Strike={meta['strike_deg']:.0f}°")
    title_parts.append(f"Dip={meta['dip_deg']:.0f}°")
    title_parts.append(f"Rake={meta['rake_deg']:.0f}°")
    title_parts.append(f"Depth={meta['depth_km']:.1f}km")
    title_parts.append(f"({meta.get('source_type', 'unknown').upper()})")

    fig.suptitle(" | ".join(title_parts), fontsize=12)
    plt.tight_layout()
    return fig


def plot_insar_products(
    result: Dict,
    title_prefix: str = "",
    figsize: Tuple[float, float] = (14, 4),
    phase_cmap: str = "hsv",
) -> plt.Figure:
    """
    Plot InSAR-specific products: phase and LOS displacement.

    Parameters
    ----------
    result : dict
        Output from generate_synthetic_insar
    title_prefix : str
        Prefix for the plot title
    figsize : tuple
        Figure size in inches
    phase_cmap : str
        Colormap for wrapped phase (default: hsv for cyclic)

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    X = result["X_km"]
    Y = result["Y_km"]
    meta = result["metadata"]
    extent = [X.min(), X.max(), Y.min(), Y.max()]

    # Wrapped phase (clean)
    im0 = axes[0].imshow(
        result["phase_wrapped"],
        extent=extent,
        origin="lower",
        cmap=phase_cmap,
        vmin=-np.pi,
        vmax=np.pi,
    )
    plt.colorbar(im0, ax=axes[0], label="Phase (rad)")
    axes[0].set_title("Wrapped Phase (Clean)")
    axes[0].set_xlabel("X (km)")
    axes[0].set_ylabel("Y (km)")

    # Wrapped phase (noisy)
    im1 = axes[1].imshow(
        result["phase_noisy"],
        extent=extent,
        origin="lower",
        cmap=phase_cmap,
        vmin=-np.pi,
        vmax=np.pi,
    )
    plt.colorbar(im1, ax=axes[1], label="Phase (rad)")
    axes[1].set_title("Wrapped Phase (With Noise)")
    axes[1].set_xlabel("X (km)")

    # LOS displacement
    los_cm = result["los_displacement"] * 100
    vmax_los = np.percentile(np.abs(los_cm), 99)
    im2 = axes[2].imshow(
        los_cm,
        extent=extent,
        origin="lower",
        cmap="RdBu_r",
        vmin=-vmax_los,
        vmax=vmax_los,
    )
    plt.colorbar(im2, ax=axes[2], label="LOS (cm)")
    axes[2].set_title("LOS Displacement")
    axes[2].set_xlabel("X (km)")

    # Title with satellite info
    sat_info = f"{meta.get('satellite', 'N/A')} {meta.get('orbit', '')}"
    fig.suptitle(
        f"{title_prefix}Mw={meta['Mw']:.1f} | {sat_info} | "
        f"λ={meta['wavelength_m']*100:.1f}cm",
        fontsize=12,
    )
    plt.tight_layout()
    return fig
