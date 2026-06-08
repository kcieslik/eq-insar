"""
Tests for InSAR projection, phase conversion, and noise modules.
"""

import sys
import numpy as np
import pytest

from eq_insar.insar.projection import (
    compute_los_vector,
    compute_los_displacement,
    displacement_to_phase,
    phase_to_displacement,
    wrap_phase,
    fringe_count,
)
from eq_insar.insar.noise import (
    generate_random_noise,
    generate_correlated_noise,
    generate_orbital_ramp,
)
from eq_insar.constants import SENTINEL1_WAVELENGTH_M


class TestLOSVector:
    """Tests for compute_los_vector."""

    def test_vertical_incidence(self):
        """At 0 incidence (looking straight down), LOS = (0, 0, 1)."""
        le, ln, lu = compute_los_vector(incidence_deg=0.0, heading_deg=0.0)
        assert abs(le) < 1e-15
        assert abs(ln) < 1e-15
        assert abs(lu - 1.0) < 1e-15

    def test_unit_vector(self):
        """LOS vector must have unit length for any angles."""
        for inc in [0, 15, 33, 45, 60, 80]:
            for head in [-170, -90, -13, 0, 45, 90, 180]:
                le, ln, lu = compute_los_vector(inc, head)
                length = np.sqrt(le**2 + ln**2 + lu**2)
                assert abs(length - 1.0) < 1e-14, (
                    f"Non-unit LOS vector for inc={inc}, head={head}: length={length}"
                )

    def test_sentinel1_ascending_values(self):
        """Known Sentinel-1 ascending geometry values.

        Ascending S1 (heading ~-13° ≈ NNW, right-looking) has its satellite
        to the west-southwest: large negative E, small negative N, large U.
        Verified against LiCSAR geo_E/N/U look-vector files (E≈-0.54, N≈-0.10, U≈0.84).
        """
        le, ln, lu = compute_los_vector(incidence_deg=33.0, heading_deg=-13.0)
        # Vertical sensitivity dominates and is positive
        assert lu > 0.8
        # East sensitivity is large and negative (satellite to the west)
        assert le < -0.4
        # North sensitivity is small and negative
        assert -0.2 < ln < 0.0


class TestLOSDisplacement:
    """Tests for compute_los_displacement."""

    def test_pure_uplift(self):
        """Pure vertical uplift should give d_LOS ≈ cos(theta) * Uz."""
        Ue = np.zeros((5, 5))
        Un = np.zeros((5, 5))
        Uz = np.ones((5, 5)) * 0.1  # 10 cm uplift

        d_los = compute_los_displacement(
            Ue, Un, Uz, incidence_deg=33.0, heading_deg=-13.0
        )

        expected = 0.1 * np.cos(np.deg2rad(33.0))
        np.testing.assert_allclose(d_los, expected, atol=1e-10)

    def test_zero_displacement(self):
        """Zero displacement should give zero LOS."""
        zeros = np.zeros((10, 10))
        d_los = compute_los_displacement(
            zeros, zeros, zeros, incidence_deg=33, heading_deg=-13
        )
        np.testing.assert_array_equal(d_los, 0.0)

    def test_satellite_parameter(self):
        """Using satellite='sentinel1' should work."""
        Uz = np.ones((5, 5)) * 0.05
        zeros = np.zeros((5, 5))
        d_los = compute_los_displacement(
            zeros, zeros, Uz, satellite="sentinel1", orbit="ascending"
        )
        assert np.all(d_los > 0)  # Uplift → positive LOS (toward satellite)


class TestPhaseConversion:
    """Tests for displacement ↔ phase conversion."""

    def test_round_trip(self):
        """displacement → phase → displacement should recover original."""
        disp = np.array([0.01, 0.02, -0.03, 0.0, 0.05])
        phase = displacement_to_phase(disp)
        recovered = phase_to_displacement(phase)
        np.testing.assert_allclose(recovered, disp, atol=1e-15)

    def test_phase_sign_convention(self):
        """Positive LOS displacement (toward satellite) → negative phase."""
        disp = np.array([0.01])  # 1 cm toward satellite
        phase = displacement_to_phase(disp)
        assert phase[0] < 0, "Positive displacement should give negative phase"

    def test_one_fringe_displacement(self):
        """One fringe = lambda/2 of LOS displacement."""
        wl = SENTINEL1_WAVELENGTH_M
        # lambda/2 displacement should give exactly 2*pi phase
        disp = np.array([wl / 2])
        phase = displacement_to_phase(disp, wavelength_m=wl)
        assert abs(abs(phase[0]) - 2 * np.pi) < 1e-12

    def test_satellite_wavelength(self):
        """Phase conversion with satellite name should use correct wavelength."""
        disp = np.array([0.01])
        phase_s1 = displacement_to_phase(disp, satellite="sentinel1")
        phase_alos = displacement_to_phase(disp, satellite="alos2")
        # L-band (ALOS-2) has longer wavelength → smaller phase for same displacement
        assert abs(phase_alos[0]) < abs(phase_s1[0])


class TestPhaseWrapping:
    """Tests for wrap_phase."""

    def test_already_wrapped(self):
        """Phase in [-pi, pi] should not change."""
        phase = np.array([-np.pi, -1.0, 0.0, 1.0, np.pi - 0.001])
        wrapped = wrap_phase(phase)
        np.testing.assert_allclose(wrapped, phase, atol=1e-10)

    def test_wraps_to_range(self):
        """Wrapped phase must be in [-pi, pi]."""
        phase = np.linspace(-50, 50, 1000)
        wrapped = wrap_phase(phase)
        assert np.all(wrapped >= -np.pi)
        assert np.all(wrapped <= np.pi)

    def test_wrap_2pi_offset(self):
        """Adding 2*pi should not change wrapped phase."""
        phase = np.array([0.5, 1.0, -2.0])
        wrapped1 = wrap_phase(phase)
        wrapped2 = wrap_phase(phase + 2 * np.pi)
        np.testing.assert_allclose(wrapped1, wrapped2, atol=1e-10)


class TestFringeCount:
    """Tests for fringe_count."""

    def test_zero_displacement(self):
        disp = np.array([0.0])
        assert fringe_count(disp)[0] == 0.0

    def test_one_fringe(self):
        """lambda/2 displacement = 1 fringe."""
        wl = SENTINEL1_WAVELENGTH_M
        disp = np.array([wl / 2])
        np.testing.assert_allclose(fringe_count(disp, wl), 1.0, atol=1e-12)


class TestNoise:
    """Tests for noise generation functions."""

    def test_noise_shape(self):
        noise = generate_random_noise((100, 100), amplitude_m=0.005)
        assert noise.shape == (100, 100)

    def test_noise_amplitude(self):
        """Standard deviation should approximate the requested amplitude."""
        noise = generate_random_noise((1000, 1000), amplitude_m=0.01, seed=42)
        assert abs(np.std(noise) - 0.01) < 0.001

    def test_noise_reproducibility(self):
        n1 = generate_random_noise((50, 50), seed=123)
        n2 = generate_random_noise((50, 50), seed=123)
        np.testing.assert_array_equal(n1, n2)

    def test_noise_different_seeds(self):
        n1 = generate_random_noise((50, 50), seed=1)
        n2 = generate_random_noise((50, 50), seed=2)
        assert not np.array_equal(n1, n2)


class TestCorrelatedNoise:
    """Tests for generate_correlated_noise."""

    def test_shape(self):
        noise = generate_correlated_noise((64, 64), amplitude_m=0.005)
        assert noise.shape == (64, 64)

    def test_amplitude(self):
        """Standard deviation should approximate the requested amplitude."""
        noise = generate_correlated_noise((512, 512), amplitude_m=0.01, seed=42)
        assert abs(np.std(noise) - 0.01) < 0.001

    def test_zero_mean(self):
        """DC component is zeroed in the FFT — field must be zero-mean."""
        noise = generate_correlated_noise((256, 256), amplitude_m=0.005, seed=0)
        assert abs(np.mean(noise)) < 1e-10

    def test_reproducibility(self):
        n1 = generate_correlated_noise((50, 50), seed=99)
        n2 = generate_correlated_noise((50, 50), seed=99)
        np.testing.assert_array_equal(n1, n2)

    def test_different_seeds(self):
        n1 = generate_correlated_noise((50, 50), seed=1)
        n2 = generate_correlated_noise((50, 50), seed=2)
        assert not np.array_equal(n1, n2)

    def test_power_law_slope(self):
        """Radially averaged PSD slope should approximate -beta on a log-log scale."""
        beta = 5 / 3
        noise = generate_correlated_noise((256, 256), amplitude_m=0.01, beta=beta, seed=42)

        # Compute 2-D PSD
        F = np.fft.fft2(noise)
        P = np.abs(F) ** 2
        fy = np.fft.fftfreq(256)
        fx = np.fft.fftfreq(256)
        FX, FY = np.meshgrid(fx, fy)
        freq = np.sqrt(FX**2 + FY**2).ravel()
        power = P.ravel()

        # Bin into 32 frequency bands, excluding DC and Nyquist region
        f_bins = np.linspace(0.02, 0.45, 33)
        f_mid = 0.5 * (f_bins[:-1] + f_bins[1:])
        p_mean = np.array([
            power[(freq >= f_bins[i]) & (freq < f_bins[i + 1])].mean()
            for i in range(len(f_bins) - 1)
        ])

        # Fit log-log slope — guarded on Windows where polyfit can cause a
        # fatal C-level exception in certain NumPy/Anaconda builds
        if sys.platform != "win32":
            slope, _ = np.polyfit(np.log10(f_mid), np.log10(p_mean), 1)
            assert abs(slope - (-beta)) < 0.4, (
                f"Expected PSD slope ≈ -{beta:.2f}, got {slope:.2f}"
            )

    def test_white_noise_limit(self):
        """beta=0 should produce a nearly flat (white) spectrum."""
        noise = generate_correlated_noise((256, 256), amplitude_m=0.01, beta=0.0, seed=42)
        F = np.fft.fft2(noise)
        P = np.abs(F) ** 2
        fy = np.fft.fftfreq(256)
        fx = np.fft.fftfreq(256)
        FX, FY = np.meshgrid(fx, fy)
        freq = np.sqrt(FX**2 + FY**2).ravel()
        power = P.ravel()
        f_bins = np.linspace(0.05, 0.45, 17)
        f_mid = 0.5 * (f_bins[:-1] + f_bins[1:])
        p_mean = np.array([
            power[(freq >= f_bins[i]) & (freq < f_bins[i + 1])].mean()
            for i in range(len(f_bins) - 1)
        ])
        if sys.platform != "win32":
            slope, _ = np.polyfit(np.log10(f_mid), np.log10(p_mean), 1)
            assert abs(slope) < 0.5, f"beta=0 should give flat spectrum, slope={slope:.2f}"


class TestOrbitalRamp:
    """Tests for orbital ramp generation."""

    def test_ramp_shape(self):
        ramp = generate_orbital_ramp((100, 100))
        assert ramp.shape == (100, 100)

    def test_zero_gradient_zero_offset(self):
        """No gradients and no offset should give zeros."""
        ramp = generate_orbital_ramp(
            (50, 50),
            ramp_east_m_per_km=0.0,
            ramp_north_m_per_km=0.0,
            offset_m=0.0,
        )
        np.testing.assert_array_equal(ramp, 0.0)

    def test_constant_offset(self):
        """Pure offset, no gradient."""
        ramp = generate_orbital_ramp(
            (50, 50),
            ramp_east_m_per_km=0.0,
            ramp_north_m_per_km=0.0,
            offset_m=0.01,
        )
        np.testing.assert_allclose(ramp, 0.01, atol=1e-15)

    def test_east_gradient(self):
        """East gradient should produce left-right variation."""
        ramp = generate_orbital_ramp(
            (50, 50),
            ramp_east_m_per_km=0.001,
            ramp_north_m_per_km=0.0,
            offset_m=0.0,
        )
        # Left column should differ from right column
        assert ramp[25, 0] != ramp[25, -1]
        # All rows should be the same (no N-S gradient)
        np.testing.assert_array_equal(ramp[0, :], ramp[-1, :])
