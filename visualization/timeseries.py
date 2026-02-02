"""
Visualization functions for time series InSAR data.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional


def plot_timeseries_frames(
    result: Dict,
    n_cols: int = 4,
    figsize_per_frame: float = 3.0,
    phase_cmap: str = "hsv",
) -> plt.Figure:
    """
    Plot time series frames as spatial maps.

    Parameters
    ----------
    result : dict
        Output from generate_timeseries
    n_cols : int
        Number of columns in the grid
    figsize_per_frame : float
        Figure size per frame in inches
    phase_cmap : str
        Colormap for phase data

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    ts = result["timeseries"]
    n_frames = ts.shape[0]
    n_rows = int(np.ceil(n_frames / n_cols))

    figsize = (figsize_per_frame * n_cols, figsize_per_frame * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    meta = result["metadata"]
    n_pre = meta["n_pre"]
    n_event = meta["n_event"]

    for i in range(n_frames):
        if i < n_pre:
            title = f"Pre-event {i+1}"
            color = "blue"
        elif i < n_pre + n_event:
            title = f"Event {i - n_pre + 1}"
            color = "red"
        else:
            title = f"Post-event {i - n_pre - n_event + 1}"
            color = "green"

        im = axes[i].imshow(ts[i], cmap=phase_cmap, vmin=-np.pi, vmax=np.pi)
        axes[i].set_title(title, fontsize=10, color=color, fontweight="bold")
        axes[i].axis("off")

    for i in range(n_frames, len(axes)):
        axes[i].axis("off")

    plt.suptitle(
        f"Time Series Frames | Mw={meta['Mw']:.1f} | {meta['source_type'].upper()}",
        fontsize=12,
    )
    plt.tight_layout()
    return fig


def plot_timeseries_statistics(
    result: Dict,
    figsize: Tuple[float, float] = (14, 10),
) -> plt.Figure:
    """
    Plot time series statistics over time.

    Shows max, mean, std, and circular variance evolution through
    pre-event, event, and post-event phases.

    Parameters
    ----------
    result : dict
        Output from generate_timeseries
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    ts = result["timeseries"]
    meta = result["metadata"]
    n_frames = ts.shape[0]
    n_pre = meta["n_pre"]
    n_event = meta["n_event"]
    output_type = meta.get("output_type", "phase")

    # Compute statistics
    max_vals = np.array([np.max(np.abs(ts[t])) for t in range(n_frames)])
    mean_vals = np.array([np.mean(ts[t]) for t in range(n_frames)])
    std_vals = np.array([np.std(ts[t]) for t in range(n_frames)])

    # Circular variance for phase
    if output_type == "phase":
        signal_power = np.array(
            [1 - np.abs(np.mean(np.exp(1j * ts[t]))) for t in range(n_frames)]
        )
    else:
        signal_power = std_vals**2

    time_indices = np.arange(n_frames)
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Phase regions
    for ax in axes.flat:
        ax.axvspan(-0.5, n_pre - 0.5, alpha=0.2, color="blue", label="Pre-event")
        ax.axvspan(n_pre - 0.5, n_pre + n_event - 0.5, alpha=0.3, color="red", label="Event")
        ax.axvspan(n_pre + n_event - 0.5, n_frames - 0.5, alpha=0.2, color="green", label="Post-event")

    unit = " (rad)" if output_type == "phase" else " (m)"

    axes[0, 0].plot(time_indices, max_vals, "ko-", linewidth=2, markersize=8)
    axes[0, 0].set_ylabel(f"Max |value|{unit}")
    axes[0, 0].set_title("Maximum Absolute Value")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(time_indices, mean_vals, "ko-", linewidth=2, markersize=8)
    axes[0, 1].set_ylabel(f"Mean value{unit}")
    axes[0, 1].set_title("Mean Value")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(time_indices, std_vals, "ko-", linewidth=2, markersize=8)
    axes[1, 0].set_ylabel(f"Std dev{unit}")
    axes[1, 0].set_title("Standard Deviation")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(time_indices, signal_power, "ko-", linewidth=2, markersize=8)
    if output_type == "phase":
        axes[1, 1].set_ylabel("Circular variance")
        axes[1, 1].set_title("Phase Dispersion")
    else:
        axes[1, 1].set_ylabel("Variance (m²)")
        axes[1, 1].set_title("Signal Variance")
    axes[1, 1].grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("Frame")
        ax.set_xticks(time_indices)

    axes[0, 0].legend(loc="upper right", fontsize=9)
    fig.suptitle(f"Time Series Statistics | Mw={meta['Mw']:.1f}", fontsize=12)
    plt.tight_layout()
    return fig


def plot_timeseries_at_points(
    result: Dict,
    points_km: Optional[List[Tuple[float, float]]] = None,
    figsize: Tuple[float, float] = (14, 10),
) -> plt.Figure:
    """
    Plot time series signal at specific pixel locations.

    Parameters
    ----------
    result : dict
        Output from generate_timeseries
    points_km : list of tuples, optional
        List of (x_km, y_km) locations. If None, uses epicenter and nearby.
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    ts = result["timeseries"]
    X_km = result["X_km"]
    Y_km = result["Y_km"]
    meta = result["metadata"]
    output_type = meta.get("output_type", "phase")

    n_frames = ts.shape[0]
    n_pre = meta["n_pre"]
    n_event = meta["n_event"]

    # Default points
    if points_km is None:
        xcen = meta["xcen_km"]
        ycen = meta["ycen_km"]
        offset = 10.0
        points_km = [
            (xcen, ycen),
            (xcen + offset, ycen),
            (xcen - offset, ycen),
            (xcen, ycen + offset),
            (xcen, ycen - offset),
        ]
        point_labels = ["Epicenter", "East +10km", "West -10km", "North +10km", "South -10km"]
    else:
        point_labels = [f"({x:.1f}, {y:.1f})" for x, y in points_km]

    def km_to_pixel(x_km, y_km):
        dx = np.abs(X_km[0, :] - x_km)
        dy = np.abs(Y_km[:, 0] - y_km)
        return np.argmin(dy), np.argmin(dx)

    time_indices = np.arange(n_frames)
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, len(points_km)))

    # Time series plot
    ax1 = axes[0]
    for i, ((x_km, y_km), label) in enumerate(zip(points_km, point_labels)):
        iy, ix = km_to_pixel(x_km, y_km)
        signal = ts[:, iy, ix]
        ax1.plot(time_indices, signal, "o-", color=colors[i], linewidth=2,
                 markersize=6, label=label)

    ax1.axvspan(-0.5, n_pre - 0.5, alpha=0.15, color="blue")
    ax1.axvspan(n_pre - 0.5, n_pre + n_event - 0.5, alpha=0.2, color="red")
    ax1.axvspan(n_pre + n_event - 0.5, n_frames - 0.5, alpha=0.15, color="green")

    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Signal" + (" (rad)" if output_type == "phase" else " (m)"))
    ax1.set_title("Signal at Specific Locations")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(time_indices)

    # Map with points
    ax2 = axes[1]
    event_frame = ts[n_pre]
    extent = [X_km.min(), X_km.max(), Y_km.min(), Y_km.max()]

    if output_type == "phase":
        im = ax2.imshow(event_frame, extent=extent, origin="lower",
                        cmap="hsv", vmin=-np.pi, vmax=np.pi)
        plt.colorbar(im, ax=ax2, label="Phase (rad)")
    else:
        vmax = np.percentile(np.abs(event_frame), 99)
        im = ax2.imshow(event_frame, extent=extent, origin="lower",
                        cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax2, label="Displacement (m)")

    for i, ((x_km, y_km), label) in enumerate(zip(points_km, point_labels)):
        ax2.plot(x_km, y_km, "o", color=colors[i], markersize=12,
                 markeredgecolor="white", markeredgewidth=2)
        ax2.annotate(f"{i+1}", (x_km, y_km), color="white", fontsize=8,
                     ha="center", va="center", fontweight="bold")

    ax2.plot(meta["xcen_km"], meta["ycen_km"], "k*", markersize=15)
    ax2.set_xlabel("X (km)")
    ax2.set_ylabel("Y (km)")
    ax2.set_title(f"Event Frame (t={n_pre})")
    ax2.set_aspect("equal")

    fig.suptitle(f"Time Series at Points | Mw={meta['Mw']:.1f}", fontsize=12)
    plt.tight_layout()
    return fig


def plot_timeseries_displacement_components(
    result: Dict,
    point_km: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (16, 12),
) -> plt.Figure:
    """
    Plot displacement components (LOS, Uz, Ue, Un) over time at a point.

    Parameters
    ----------
    result : dict
        Output from generate_timeseries
    point_km : tuple (x_km, y_km), optional
        Location. If None, uses max |LOS| location.
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    X_km = result["X_km"]
    Y_km = result["Y_km"]
    meta = result["metadata"]

    n_pre = meta["n_pre"]
    n_event = meta["n_event"]
    n_total = meta["n_total"]

    # Get fields
    LOS_field = result["los_displacement"]
    Ue_field = result["Ue"]
    Un_field = result["Un"]
    Uz_field = result["Uz"]

    # Find point
    if point_km is None:
        max_idx = np.unravel_index(np.argmax(np.abs(LOS_field)), LOS_field.shape)
        iy, ix = max_idx
        x_km = X_km[iy, ix]
        y_km = Y_km[iy, ix]
        point_label = f"Max |LOS| ({x_km:.1f}, {y_km:.1f}) km"
    else:
        x_km, y_km = point_km
        dx = np.abs(X_km[0, :] - x_km)
        dy = np.abs(Y_km[:, 0] - y_km)
        ix = np.argmin(dx)
        iy = np.argmin(dy)
        x_km = X_km[iy, ix]
        y_km = Y_km[iy, ix]
        point_label = f"User-defined ({x_km:.1f}, {y_km:.1f}) km"

    # Extract values
    LOS_val = LOS_field[iy, ix]
    Ue_val = Ue_field[iy, ix]
    Un_val = Un_field[iy, ix]
    Uz_val = Uz_field[iy, ix]
    Uh_val = np.sqrt(Ue_val**2 + Un_val**2)

    # Build time series
    time_indices = np.arange(n_total)
    LOS_ts = np.zeros(n_total)
    Ue_ts = np.zeros(n_total)
    Un_ts = np.zeros(n_total)
    Uz_ts = np.zeros(n_total)
    Uh_ts = np.zeros(n_total)

    for t in range(n_total):
        if n_pre <= t < n_pre + n_event:
            LOS_ts[t] = LOS_val
            Ue_ts[t] = Ue_val
            Un_ts[t] = Un_val
            Uz_ts[t] = Uz_val
            Uh_ts[t] = Uh_val

    # Convert to cm
    LOS_cm = LOS_ts * 100
    Ue_cm = Ue_ts * 100
    Un_cm = Un_ts * 100
    Uz_cm = Uz_ts * 100
    Uh_cm = Uh_ts * 100

    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.3, wspace=0.3)

    ax_ts = fig.add_subplot(gs[0, :])
    ax_map = fig.add_subplot(gs[1, 0])
    ax_bar = fig.add_subplot(gs[1, 1])

    # Time series
    ax_ts.axvspan(-0.5, n_pre - 0.5, alpha=0.15, color="blue", label="Pre-event")
    ax_ts.axvspan(n_pre - 0.5, n_pre + n_event - 0.5, alpha=0.25, color="red", label="Event")
    ax_ts.axvspan(n_pre + n_event - 0.5, n_total - 0.5, alpha=0.15, color="green", label="Post-event")

    ax_ts.plot(time_indices, LOS_cm, "ko-", lw=2.5, ms=10, label=f"LOS ({LOS_val*100:.2f} cm)")
    ax_ts.plot(time_indices, Uz_cm, "s-", color="tab:blue", lw=2, ms=8, label=f"Uz ({Uz_val*100:.2f} cm)")
    ax_ts.plot(time_indices, Ue_cm, "^-", color="tab:orange", lw=2, ms=8, label=f"Ue ({Ue_val*100:.2f} cm)")
    ax_ts.plot(time_indices, Un_cm, "v-", color="tab:green", lw=2, ms=8, label=f"Un ({Un_val*100:.2f} cm)")
    ax_ts.plot(time_indices, Uh_cm, "d--", color="tab:purple", lw=2, ms=7, label=f"|Uh| ({Uh_val*100:.2f} cm)")

    ax_ts.axhline(0, color="gray", lw=0.5)
    ax_ts.set_xlabel("Frame")
    ax_ts.set_ylabel("Displacement (cm)")
    ax_ts.set_title(f"Displacement at {point_label}")
    ax_ts.legend(loc="upper right", fontsize=9)
    ax_ts.grid(True, alpha=0.3)
    ax_ts.set_xticks(time_indices)

    # Map
    extent = [X_km.min(), X_km.max(), Y_km.min(), Y_km.max()]
    vmax = np.percentile(np.abs(LOS_field), 99) * 100
    im = ax_map.imshow(LOS_field * 100, extent=extent, origin="lower",
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax_map, label="LOS (cm)")
    ax_map.plot(x_km, y_km, "ko", ms=15, mfc="yellow", mec="black", mew=2)
    ax_map.plot(meta["xcen_km"], meta["ycen_km"], "k*", ms=12)
    ax_map.set_xlabel("X (km)")
    ax_map.set_ylabel("Y (km)")
    ax_map.set_title("LOS Displacement")
    ax_map.set_aspect("equal")

    # Bar chart
    components = ["LOS", "Uz", "Ue", "Un", "|Uh|"]
    values = [LOS_val * 100, Uz_val * 100, Ue_val * 100, Un_val * 100, Uh_val * 100]
    colors_bar = ["black", "tab:blue", "tab:orange", "tab:green", "tab:purple"]
    bars = ax_bar.bar(components, values, color=colors_bar, edgecolor="black", lw=1.5)
    ax_bar.axhline(0, color="gray", lw=0.5)
    ax_bar.set_ylabel("Displacement (cm)")
    ax_bar.set_title("Components at Point")
    ax_bar.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, values):
        h = bar.get_height()
        ax_bar.annotate(f"{val:.2f}", xy=(bar.get_x() + bar.get_width()/2, h),
                       xytext=(0, 3 if h >= 0 else -12), textcoords="offset points",
                       ha="center", va="bottom" if h >= 0 else "top", fontsize=9, fontweight="bold")

    fig.suptitle(f"Displacement Time Series | Mw={meta['Mw']:.1f} | {meta['source_type'].upper()}", fontsize=13)
    plt.tight_layout()
    return fig


def plot_timeseries_profile(
    result: Dict,
    direction: str = "EW",
    position_km: float = 0.0,
    figsize: Tuple[float, float] = (14, 10),
) -> plt.Figure:
    """
    Plot space-time profile (kymograph).

    Parameters
    ----------
    result : dict
        Output from generate_timeseries
    direction : str
        'EW' (East-West) or 'NS' (North-South)
    position_km : float
        Position of profile line in km
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    ts = result["timeseries"]
    X_km = result["X_km"]
    Y_km = result["Y_km"]
    meta = result["metadata"]

    n_frames = ts.shape[0]
    n_pre = meta["n_pre"]
    n_event = meta["n_event"]

    if direction.upper() == "EW":
        iy = np.argmin(np.abs(Y_km[:, 0] - position_km))
        profile = ts[:, iy, :]
        distance = X_km[0, :]
        xlabel = "X (km)"
        profile_label = f"Y = {Y_km[iy, 0]:.1f} km"
    else:
        ix = np.argmin(np.abs(X_km[0, :] - position_km))
        profile = ts[:, :, ix]
        distance = Y_km[:, 0]
        xlabel = "Y (km)"
        profile_label = f"X = {X_km[0, ix]:.1f} km"

    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # Kymograph
    extent = [distance.min(), distance.max(), -0.5, n_frames - 0.5]
    im = axes[0].imshow(profile, extent=extent, aspect="auto", origin="lower",
                        cmap="hsv", vmin=-np.pi, vmax=np.pi)
    plt.colorbar(im, ax=axes[0], label="Phase (rad)")
    axes[0].axhline(n_pre - 0.5, color="white", ls="--", lw=2, label="Event start")
    axes[0].axhline(n_pre + n_event - 0.5, color="white", ls=":", lw=2, label="Event end")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Frame")
    axes[0].set_title(f"Space-Time Diagram ({profile_label})")
    axes[0].legend(loc="upper right")

    # Profiles
    colors = ["blue", "red", "green"]
    frames = [n_pre // 2, n_pre, n_pre + n_event + (n_frames - n_pre - n_event) // 2]
    labels = [f"Pre (t={frames[0]})", f"Event (t={frames[1]})", f"Post (t={frames[2]})"]

    for frame, label, color in zip(frames, labels, colors):
        axes[1].plot(distance, profile[frame], "-", lw=2, label=label, color=color)

    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Phase (rad)")
    axes[1].set_title("Profiles at Different Times")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-np.pi, np.pi)

    fig.suptitle(f"Space-Time Evolution | Mw={meta['Mw']:.1f} | {direction}", fontsize=12)
    plt.tight_layout()
    return fig


def plot_timeseries_difference(
    result: Dict,
    n_cols: int = 4,
    figsize_per_frame: float = 3.5,
) -> plt.Figure:
    """
    Plot frame-to-frame differences to highlight deformation onset.

    Parameters
    ----------
    result : dict
        Output from generate_timeseries
    n_cols : int
        Columns in grid
    figsize_per_frame : float
        Size per frame

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    ts = result["timeseries"]
    meta = result["metadata"]
    n_frames = ts.shape[0]
    n_pre = meta["n_pre"]
    n_event = meta["n_event"]

    # Compute differences
    diff = np.zeros((n_frames - 1, ts.shape[1], ts.shape[2]))
    for t in range(n_frames - 1):
        diff[t] = np.angle(np.exp(1j * (ts[t + 1] - ts[t])))

    n_diff = n_frames - 1
    n_rows = int(np.ceil(n_diff / n_cols))
    figsize = (figsize_per_frame * n_cols, figsize_per_frame * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i in range(n_diff):
        t_from, t_to = i, i + 1

        if t_to <= n_pre:
            title = f"Pre: {t_from}→{t_to}"
            color = "blue"
        elif t_from < n_pre <= t_to:
            title = f"ONSET: {t_from}→{t_to}"
            color = "red"
        elif t_from < n_pre + n_event <= t_to:
            title = f"END: {t_from}→{t_to}"
            color = "orange"
        elif n_pre <= t_from < n_pre + n_event:
            title = f"Event: {t_from}→{t_to}"
            color = "red"
        else:
            title = f"Post: {t_from}→{t_to}"
            color = "green"

        vmax = max(np.percentile(np.abs(diff[i]), 98), 0.1)
        axes[i].imshow(diff[i], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[i].set_title(title, fontsize=10, color=color, fontweight="bold")
        axes[i].axis("off")

    for i in range(n_diff, len(axes)):
        axes[i].axis("off")

    fig.suptitle(f"Frame Differences | Mw={meta['Mw']:.1f}", fontsize=12)
    plt.tight_layout()
    return fig
