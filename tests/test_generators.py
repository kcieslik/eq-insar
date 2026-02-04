"""Tests for EQ-INSAR generator functions."""

import numpy as np
import pytest


class TestGenerateSyntheticInsar:
    """Tests for generate_synthetic_insar function."""

    def test_basic_generation(self):
        """Test basic interferogram generation."""
        from eq_insar import generate_synthetic_insar

        result = generate_synthetic_insar(
            Mw=6.0,
            strike_deg=45,
            dip_deg=30,
            rake_deg=90,
            depth_km=10,
            satellite='sentinel1',
            add_noise=False,
            seed=42
        )

        # Check required keys exist
        assert 'los_displacement' in result
        assert 'phase_wrapped' in result
        assert 'phase_unwrapped' in result
        assert 'metadata' in result
        assert 'Ue' in result
        assert 'Un' in result
        assert 'Uz' in result

    def test_output_shapes(self):
        """Test that output arrays have correct shapes."""
        from eq_insar import generate_synthetic_insar

        result = generate_synthetic_insar(
            Mw=6.0,
            grid_extent_km=50,
            grid_spacing_km=0.5,
            add_noise=False
        )

        expected_size = int(2 * 50 / 0.5) + 1  # 201
        assert result['los_displacement'].shape == (expected_size, expected_size)
        assert result['phase_wrapped'].shape == (expected_size, expected_size)

    def test_magnitude_scaling(self):
        """Test that larger magnitude produces larger displacement."""
        from eq_insar import generate_synthetic_insar

        result_small = generate_synthetic_insar(Mw=5.0, add_noise=False)
        result_large = generate_synthetic_insar(Mw=6.0, add_noise=False)

        max_small = np.abs(result_small['los_displacement']).max()
        max_large = np.abs(result_large['los_displacement']).max()

        assert max_large > max_small

    def test_depth_effect(self):
        """Test that deeper source produces smaller surface displacement."""
        from eq_insar import generate_synthetic_insar

        result_shallow = generate_synthetic_insar(Mw=6.0, depth_km=5, add_noise=False)
        result_deep = generate_synthetic_insar(Mw=6.0, depth_km=20, add_noise=False)

        max_shallow = np.abs(result_shallow['los_displacement']).max()
        max_deep = np.abs(result_deep['los_displacement']).max()

        assert max_shallow > max_deep

    def test_satellite_configurations(self):
        """Test different satellite configurations."""
        from eq_insar import generate_synthetic_insar, list_satellites

        satellites = list(list_satellites().keys())

        for sat in satellites[:3]:  # Test first 3 satellites
            result = generate_synthetic_insar(
                Mw=6.0,
                satellite=sat,
                add_noise=False
            )
            assert result['metadata']['satellite'] == list_satellites()[sat].name

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same result."""
        from eq_insar import generate_synthetic_insar

        result1 = generate_synthetic_insar(Mw=6.0, seed=42)
        result2 = generate_synthetic_insar(Mw=6.0, seed=42)

        np.testing.assert_array_equal(
            result1['los_displacement'],
            result2['los_displacement']
        )

    def test_phase_wrapping(self):
        """Test that wrapped phase is within [-pi, pi]."""
        from eq_insar import generate_synthetic_insar

        result = generate_synthetic_insar(Mw=6.5, add_noise=False)

        assert result['phase_wrapped'].min() >= -np.pi
        assert result['phase_wrapped'].max() <= np.pi


class TestGenerateTimeseries:
    """Tests for generate_timeseries function."""

    def test_basic_timeseries(self):
        """Test basic time series generation."""
        from eq_insar import generate_timeseries

        result = generate_timeseries(
            Mw=6.0,
            n_pre=3,
            n_event=1,
            n_post=3,
            seed=42
        )

        assert 'timeseries' in result
        assert 'labels' in result

        # Check shape: n_pre + n_event + n_post frames
        n_frames = 3 + 1 + 3
        assert result['timeseries'].shape[0] == n_frames
        assert result['labels'].shape[0] == n_frames

    def test_labels_structure(self):
        """Test that labels are only non-zero during event."""
        from eq_insar import generate_timeseries

        result = generate_timeseries(
            Mw=6.0,
            n_pre=3,
            n_event=1,
            n_post=3,
            seed=42
        )

        labels = result['labels']

        # Pre-event frames should have no deformation labels
        for i in range(3):
            assert labels[i].sum() == 0, f"Pre-event frame {i} should have no labels"

        # Event frame should have labels
        assert labels[3].sum() > 0, "Event frame should have labels"

        # Post-event frames should have no labels
        for i in range(4, 7):
            assert labels[i].sum() == 0, f"Post-event frame {i} should have no labels"


class TestBatchGeneration:
    """Tests for generate_training_batch function."""

    def test_batch_size(self):
        """Test that batch generates correct number of samples."""
        from eq_insar import generate_training_batch

        batch = generate_training_batch(
            n_samples=5,
            seed=42
        )

        assert len(batch) == 5

    def test_batch_magnitude_range(self):
        """Test that batch respects magnitude range."""
        from eq_insar import generate_training_batch

        batch = generate_training_batch(
            n_samples=10,
            mw_range=(5.5, 6.5),
            seed=42
        )

        for sample in batch:
            mw = sample['metadata']['Mw']
            assert 5.5 <= mw <= 6.5


class TestCoreFunctions:
    """Tests for core physics functions."""

    def test_mw_to_m0_conversion(self):
        """Test moment magnitude to scalar moment conversion."""
        from eq_insar import mw_to_m0

        # Mw 6.0 should give M0 ~ 1.26e18 N*m
        m0 = mw_to_m0(6.0)
        assert 1e18 < m0 < 2e18

        # Mw 7.0 should be ~31.6x larger (10^1.5)
        m0_7 = mw_to_m0(7.0)
        ratio = m0_7 / m0
        assert 30 < ratio < 33

    def test_double_couple_moment_tensor(self):
        """Test moment tensor construction."""
        from eq_insar import double_couple_moment_tensor, mw_to_m0

        M0 = mw_to_m0(6.0)
        Mxx, Myy, Mzz, Mxy, Myz, Mzx = double_couple_moment_tensor(
            strike_deg=0, dip_deg=90, rake_deg=0, M0=M0
        )

        # For vertical strike-slip, should have Mxy = M0
        assert np.abs(Mxy) > 0

        # Moment tensor should be traceless (double-couple)
        trace = Mxx + Myy + Mzz
        assert np.abs(trace) < 1e10  # Effectively zero


class TestSatelliteConfig:
    """Tests for satellite configuration."""

    def test_list_satellites(self):
        """Test that satellite list is not empty."""
        from eq_insar import list_satellites

        satellites = list_satellites()
        assert len(satellites) > 0

    def test_get_satellite(self):
        """Test getting specific satellite config."""
        from eq_insar import get_satellite

        s1 = get_satellite('sentinel1')

        assert s1.name == 'Sentinel-1'
        assert s1.band == 'C'
        assert 0.05 < s1.wavelength_m < 0.06  # C-band ~5.5 cm

    def test_satellite_heading(self):
        """Test satellite heading for ascending/descending."""
        from eq_insar import get_satellite

        s1 = get_satellite('sentinel1')

        heading_asc = s1.get_heading('ascending')
        heading_desc = s1.get_heading('descending')

        # Ascending and descending should have different headings
        assert heading_asc != heading_desc
