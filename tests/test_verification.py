"""
Verification tests for the Davis (1986) point source implementation.

These tests reproduce the verification results reported in the EQ-INSAR paper
(Section 4: Verification). They exercise analytical properties of the Green's
functions and moment tensor construction.

Run with: pytest tests/test_verification.py -v
"""

import sys

import numpy as np
import pytest

from eq_insar.core.moment_tensor import double_couple_moment_tensor, mw_to_m0
from eq_insar.core.davis import davis_point_source


class TestDisplacementSymmetry:
    """
    Verify that displacement fields have the correct symmetries
    for known source geometries.

    Paper reference: Verification, "Symmetry of displacement patterns"
    """

    @pytest.fixture
    def grid(self):
        """201x201 grid covering +/-50 km at 10 km depth."""
        x = np.linspace(-50000, 50000, 201)
        y = np.linspace(-50000, 50000, 201)
        X, Y = np.meshgrid(x, y)
        return X, Y

    @pytest.fixture
    def M0(self):
        return mw_to_m0(6.0)

    def test_isotropic_source_radial_symmetry(self, grid, M0):
        """
        Isotropic source (Mxx = Myy = Mzz = M0/3): Uz must be radially
        symmetric — invariant under mirror flips and x<->y transpose.
        """
        X, Y = grid
        depth = 10000
        Ue, Un, Uz = davis_point_source(
            X, Y, 0, 0, depth,
            M0 / 3, M0 / 3, M0 / 3, 0, 0, 0
        )

        max_uz = np.max(np.abs(Uz))
        assert max_uz > 0, "Displacement should be non-zero"

        # Mirror about N-S axis (flip x)
        assert np.max(np.abs(Uz - Uz[:, ::-1])) == 0.0

        # Mirror about E-W axis (flip y)
        assert np.max(np.abs(Uz - Uz[::-1, :])) == 0.0

        # Transpose (swap x and y)
        assert np.max(np.abs(Uz - Uz.T)) == 0.0

    def test_strike_slip_antisymmetry(self, grid, M0):
        """
        Strike-slip on vertical N-striking fault (strike=0, dip=90, rake=0):
        Uz must be antisymmetric about both the E-W and N-S axes.
        """
        X, Y = grid
        depth = 10000
        Mxx, Myy, Mzz, Mxy, Myz, Mzx = double_couple_moment_tensor(0, 90, 0, M0)
        Ue, Un, Uz = davis_point_source(
            X, Y, 0, 0, depth, Mxx, Myy, Mzz, Mxy, Myz, Mzx
        )

        max_uz = np.max(np.abs(Uz))
        assert max_uz > 0

        # Antisymmetric about N-S axis: Uz(x,y) = -Uz(-x,y)
        residual_ns = np.max(np.abs(Uz + Uz[:, ::-1]))
        assert residual_ns / max_uz < 1e-14

        # Antisymmetric about E-W axis: Uz(x,y) = -Uz(x,-y)
        residual_ew = np.max(np.abs(Uz + Uz[::-1, :]))
        assert residual_ew / max_uz < 1e-14

        # Symmetric under 180-degree rotation: Uz(x,y) = Uz(-x,-y)
        residual_180 = np.max(np.abs(Uz - Uz[::-1, ::-1]))
        assert residual_180 / max_uz < 1e-14

    def test_dip_slip_ew_symmetry(self, grid, M0):
        """
        Dip-slip on vertical N-striking fault (strike=0, dip=90, rake=90):
        Uz must be symmetric about the E-W axis (the fault strike direction).
        """
        X, Y = grid
        depth = 10000
        Mxx, Myy, Mzz, Mxy, Myz, Mzx = double_couple_moment_tensor(0, 90, 90, M0)
        Ue, Un, Uz = davis_point_source(
            X, Y, 0, 0, depth, Mxx, Myy, Mzz, Mxy, Myz, Mzx
        )

        max_uz = np.max(np.abs(Uz))
        assert max_uz > 0

        # Symmetric about E-W axis: Uz(x,y) = Uz(x,-y)
        residual = np.max(np.abs(Uz - Uz[::-1, :]))
        assert residual / max_uz < 1e-14


class TestMomentTensorTrace:
    """
    Verify that the double-couple moment tensor has zero trace
    for all fault orientations.

    Paper reference: Verification, "Moment tensor zero-trace property"
    """

    def test_zero_trace_random_orientations(self):
        """
        tr(M) = Mee + Mnn + Muu = 0 for 10,000 random fault orientations.
        """
        np.random.seed(42)
        M0 = mw_to_m0(6.0)
        n_tests = 10000
        max_rel_trace = 0.0

        for _ in range(n_tests):
            strike = np.random.uniform(0, 360)
            dip = np.random.uniform(0, 90)
            rake = np.random.uniform(-180, 180)
            Mxx, Myy, Mzz, Mxy, Myz, Mzx = double_couple_moment_tensor(
                strike, dip, rake, M0
            )
            trace = abs(Mxx + Myy + Mzz)
            rel_trace = trace / M0
            if rel_trace > max_rel_trace:
                max_rel_trace = rel_trace

        assert max_rel_trace < 1e-14, (
            f"Max |tr(M)|/M0 = {max_rel_trace:.2e}, expected < 1e-14"
        )

    def test_zero_trace_specific_mechanisms(self):
        """Zero trace for canonical fault types."""
        M0 = mw_to_m0(6.0)

        mechanisms = [
            (0, 90, 0, "strike-slip"),
            (45, 30, 90, "thrust"),
            (180, 60, -90, "normal"),
            (270, 45, 45, "oblique"),
        ]

        for strike, dip, rake, name in mechanisms:
            Mxx, Myy, Mzz, Mxy, Myz, Mzx = double_couple_moment_tensor(
                strike, dip, rake, M0
            )
            trace = abs(Mxx + Myy + Mzz)
            assert trace / M0 < 1e-14, f"Non-zero trace for {name} mechanism"


class TestFarFieldDecay:
    """
    Verify that peak surface displacement decays as 1/d^2 with source depth.

    Paper reference: Verification, "Far-field 1/d^2 displacement decay"
    """

    def test_inverse_square_decay(self):
        """
        Peak |Uz| * d^2 must be constant across depths 5-80 km.
        Log-log slope must be -2.000.
        """
        M0 = mw_to_m0(6.0)
        Mxx, Myy, Mzz, Mxy, Myz, Mzx = double_couple_moment_tensor(45, 30, 90, M0)

        depths_km = [5, 10, 20, 40, 80]
        peak_uz = []

        for d_km in depths_km:
            d_m = d_km * 1000
            extent = d_km * 10 * 1000  # 10x depth
            x = np.linspace(-extent, extent, 401)
            y = np.linspace(-extent, extent, 401)
            X, Y = np.meshgrid(x, y)

            Ue, Un, Uz = davis_point_source(
                X, Y, 0, 0, d_m, Mxx, Myy, Mzz, Mxy, Myz, Mzx
            )
            peak_uz.append(np.max(np.abs(Uz)))

        # Check that |Uz| * d^2 is constant (relaxed to 1e-6 for cross-platform stability)
        products = [uz * (d * 1000) ** 2 for uz, d in zip(peak_uz, depths_km)]
        ref = products[0]
        for i, (d_km, prod) in enumerate(zip(depths_km, products)):
            ratio = prod / ref
            assert abs(ratio - 1.0) < 1e-6, (
                f"At depth={d_km}km: |Uz|*d^2 ratio = {ratio:.10f}, expected 1.0"
            )

        # Verify log-log slope is exactly -2
        # Skipped on Windows: np.polyfit triggers a fatal exception in some
        # NumPy builds (observed with NumPy 1.26.4 / Python 3.11 / Anaconda).
        # The core physics check above is the meaningful assertion.
        if sys.platform != "win32":
            log_d = np.log10([d * 1000 for d in depths_km])
            log_uz = np.log10(peak_uz)
            slope, _ = np.polyfit(log_d, log_uz, 1)
            assert abs(slope - (-2.0)) < 1e-6, (
                f"Power law exponent = {slope:.6f}, expected -2.000000"
            )
