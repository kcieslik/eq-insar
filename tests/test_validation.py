"""
Tests for input validation in generators.

Ensures that invalid parameters raise clear errors rather than producing
silent nonsense output.
"""

import pytest
import numpy as np
from eq_insar import generate_synthetic_insar, generate_training_batch
from eq_insar.generators.batch import sample_earthquake_parameters


class TestSourceParameterValidation:
    """Tests for earthquake source parameter validation."""

    def test_missing_magnitude(self):
        """Must provide either Mw or M0."""
        with pytest.raises(ValueError, match="Must provide either Mw or M0"):
            generate_synthetic_insar(strike_deg=0, dip_deg=45, rake_deg=90)

    def test_mw_too_small(self):
        with pytest.raises(ValueError, match="Mw=.*outside the valid range"):
            generate_synthetic_insar(Mw=-6.0, depth_km=10)

    def test_mw_too_large(self):
        with pytest.raises(ValueError, match="Mw=.*outside the valid range"):
            generate_synthetic_insar(Mw=11.0, depth_km=10)

    def test_negative_mw_valid(self):
        """Negative Mw is valid for microearthquakes."""
        result = generate_synthetic_insar(
            Mw=-2.0, depth_km=5, grid_size=32, add_noise=False
        )
        assert "los_displacement" in result

    def test_negative_m0(self):
        with pytest.raises(ValueError, match="M0 must be positive"):
            generate_synthetic_insar(M0=-1e18, depth_km=10)

    def test_negative_depth(self):
        with pytest.raises(ValueError, match="depth_km.*outside the valid range"):
            generate_synthetic_insar(Mw=6.0, depth_km=-5.0)

    def test_zero_depth(self):
        with pytest.raises(ValueError, match="depth_km.*outside the valid range"):
            generate_synthetic_insar(Mw=6.0, depth_km=0.0)

    def test_dip_out_of_range(self):
        with pytest.raises(ValueError, match="dip_deg.*outside the valid range"):
            generate_synthetic_insar(Mw=6.0, dip_deg=100, depth_km=10)

    def test_negative_dip(self):
        with pytest.raises(ValueError, match="dip_deg.*outside the valid range"):
            generate_synthetic_insar(Mw=6.0, dip_deg=-10, depth_km=10)

    def test_strike_out_of_range(self):
        with pytest.raises(ValueError, match="strike_deg.*outside the valid range"):
            generate_synthetic_insar(Mw=6.0, strike_deg=400, depth_km=10)

    def test_rake_out_of_range(self):
        with pytest.raises(ValueError, match="rake_deg.*outside the valid range"):
            generate_synthetic_insar(Mw=6.0, rake_deg=200, depth_km=10)

    def test_valid_boundary_values(self):
        """Boundary values should NOT raise errors."""
        # These are all valid edge cases
        result = generate_synthetic_insar(
            Mw=-5.0, strike_deg=0, dip_deg=0, rake_deg=-180,
            depth_km=0.1, grid_size=32, add_noise=False
        )
        assert "los_displacement" in result

        result = generate_synthetic_insar(
            Mw=10.0, strike_deg=360, dip_deg=90, rake_deg=180,
            depth_km=100, grid_size=32, add_noise=False
        )
        assert "los_displacement" in result


class TestBatchParameterValidation:
    """Tests for batch generation parameter validation."""

    def test_inverted_mw_range(self):
        with pytest.raises(ValueError, match="mw_range min.*> max"):
            sample_earthquake_parameters(mw_range=(7.0, 4.0))

    def test_inverted_depth_range(self):
        with pytest.raises(ValueError, match="depth_range_km min.*> max"):
            sample_earthquake_parameters(depth_range_km=(20.0, 5.0))

    def test_negative_depth_range(self):
        with pytest.raises(ValueError, match="depth_range_km min.*must be positive"):
            sample_earthquake_parameters(depth_range_km=(-5.0, 10.0))

    def test_valid_sample_parameters(self):
        """Valid ranges should work without errors."""
        params = sample_earthquake_parameters(
            mw_range=(4.5, 7.0),
            depth_range_km=(5.0, 20.0),
            seed=42
        )
        assert 4.5 <= params["Mw"] <= 7.0
        assert 5.0 <= params["depth_km"] <= 20.0
        assert 0 <= params["strike_deg"] <= 360
        assert 0 <= params["dip_deg"] <= 90
