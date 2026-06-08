"""
Noise models for synthetic InSAR data.

This module provides noise models:
- Uncorrelated Gaussian noise
- Spatially correlated power-law noise (atmospheric turbulence)
- Orbital/baseline ramp errors
"""

import numpy as np
from typing import Tuple, Optional


def generate_random_noise(
    shape: Tuple[int, int],
    amplitude_m: float = 0.005,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate simple random Gaussian noise.

    Parameters
    ----------
    shape : tuple (ny, nx)
        Output array shape
    amplitude_m : float
        Standard deviation of noise in meters (default: 5 mm)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    noise : np.ndarray
        Random noise field in meters
    """
    if seed is not None:
        np.random.seed(seed)

    return amplitude_m * np.random.randn(*shape)


def generate_correlated_noise(
    shape: Tuple[int, int],
    amplitude_m: float = 0.005,
    beta: float = 5 / 3,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate spatially correlated noise with a power-law spectrum.

    Models atmospheric turbulence in InSAR interferograms. The power spectral
    density follows P(f) ∝ f^(-β).  Kolmogorov atmospheric turbulence uses
    β = 5/3; β = 0 recovers uncorrelated white noise.

    Parameters
    ----------
    shape : tuple (ny, nx)
        Output array shape
    amplitude_m : float
        Standard deviation of the output noise field in meters (default: 5 mm)
    beta : float
        Power-law spectral index (default: 5/3 ≈ Kolmogorov turbulence).
        Higher values produce longer-range spatial correlations.
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    noise : np.ndarray
        Spatially correlated, zero-mean noise field in meters
    """
    rng = np.random.default_rng(seed)
    ny, nx = shape

    fy = np.fft.fftfreq(ny)
    fx = np.fft.fftfreq(nx)
    FX, FY = np.meshgrid(fx, fy)
    freq = np.sqrt(FX**2 + FY**2)
    freq[0, 0] = 1.0          # avoid divide-by-zero at DC

    power = freq ** (-beta / 2)
    power[0, 0] = 0.0          # zero mean: remove DC component

    white = rng.standard_normal((ny, nx)) + 1j * rng.standard_normal((ny, nx))
    field = np.real(np.fft.ifft2(power * white))

    std = field.std()
    if std > 0:
        field /= std

    return field * amplitude_m


def generate_orbital_ramp(
    shape: Tuple[int, int],
    ramp_east_m_per_km: float = 0.0,
    ramp_north_m_per_km: float = 0.0,
    offset_m: float = 0.0,
    pixel_size_km: float = 0.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate orbital/baseline error ramp.

    Parameters
    ----------
    shape : tuple (ny, nx)
        Output array shape
    ramp_east_m_per_km : float
        East-West gradient in meters per km
    ramp_north_m_per_km : float
        North-South gradient in meters per km
    offset_m : float
        Constant offset in meters
    pixel_size_km : float
        Pixel size in km
    seed : int, optional
        Random seed (for random ramp if gradients not specified)

    Returns
    -------
    ramp : np.ndarray
        Orbital ramp in meters
    """
    ny, nx = shape

    if ramp_east_m_per_km == 0 and ramp_north_m_per_km == 0 and seed is not None:
        np.random.seed(seed)
        ramp_east_m_per_km = np.random.uniform(-0.0005, 0.0005)
        ramp_north_m_per_km = np.random.uniform(-0.0005, 0.0005)

    x_km = np.arange(nx) * pixel_size_km - (nx - 1) * pixel_size_km / 2
    y_km = np.arange(ny) * pixel_size_km - (ny - 1) * pixel_size_km / 2
    X_km, Y_km = np.meshgrid(x_km, y_km)

    ramp = offset_m + ramp_east_m_per_km * X_km + ramp_north_m_per_km * Y_km

    return ramp
