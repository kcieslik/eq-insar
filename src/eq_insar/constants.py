"""
DC-INSAR: Physical constants and satellite parameters.

This module contains:
- Radar wavelengths for various SAR satellites
- Default elastic parameters for crustal deformation
- Satellite orbital and imaging geometry parameters
"""

from dataclasses import dataclass
from typing import Dict

# =============================================================================
# Radar Wavelengths (meters)
# =============================================================================

# C-band (~5.6 cm)
SENTINEL1_WAVELENGTH_M = 0.05546  # Sentinel-1 A/B
RADARSAT2_WAVELENGTH_M = 0.05546  # RADARSAT-2
ENVISAT_WAVELENGTH_M = 0.05624   # ENVISAT ASAR
ERS_WAVELENGTH_M = 0.05666       # ERS-1/2

# L-band (~23.6 cm)
ALOS2_WAVELENGTH_M = 0.2291      # ALOS-2 PALSAR-2
ALOS_WAVELENGTH_M = 0.2360       # ALOS PALSAR
NISAR_L_WAVELENGTH_M = 0.238     # NISAR L-band
SAOCOM_WAVELENGTH_M = 0.2350     # SAOCOM-1A/1B

# X-band (~3.1 cm)
TERRASAR_WAVELENGTH_M = 0.03106  # TerraSAR-X / TanDEM-X
COSMO_WAVELENGTH_M = 0.03125     # COSMO-SkyMed
PAZ_WAVELENGTH_M = 0.03106       # PAZ
ICEYE_WAVELENGTH_M = 0.03106     # ICEYE

# S-band (~9.4 cm)
NISAR_S_WAVELENGTH_M = 0.094     # NISAR S-band

# =============================================================================
# Elastic Parameters
# =============================================================================

DEFAULT_SHEAR_MODULUS_PA = 3.0e10    # ~30 GPa, typical upper crust
DEFAULT_POISSON_RATIO = 0.25         # Typical crustal value
DEFAULT_LAME_LAMBDA_PA = 3.0e10      # λ = μ for ν = 0.25


# =============================================================================
# Satellite Configuration Dataclass
# =============================================================================

@dataclass
class SatelliteConfig:
    """Configuration for a SAR satellite."""
    name: str
    wavelength_m: float
    band: str
    incidence_deg: float        # Typical/default incidence angle
    incidence_range: tuple      # (min, max) incidence angle range
    heading_asc_deg: float      # Ascending orbit heading
    heading_desc_deg: float     # Descending orbit heading
    revisit_days: int           # Nominal revisit period
    resolution_m: float         # Typical ground resolution
    agency: str

    def get_heading(self, orbit: str = "ascending") -> float:
        """Get satellite heading for orbit direction."""
        if orbit.lower() in ["asc", "ascending"]:
            return self.heading_asc_deg
        elif orbit.lower() in ["desc", "descending"]:
            return self.heading_desc_deg
        else:
            raise ValueError(f"Unknown orbit: {orbit}. Use 'ascending' or 'descending'.")


# =============================================================================
# Pre-defined Satellite Configurations
# =============================================================================

SATELLITES: Dict[str, SatelliteConfig] = {
    # C-band satellites
    "sentinel1": SatelliteConfig(
        name="Sentinel-1",
        wavelength_m=SENTINEL1_WAVELENGTH_M,
        band="C",
        incidence_deg=33.0,
        incidence_range=(29.0, 46.0),
        heading_asc_deg=-13.0,
        heading_desc_deg=-167.0,
        revisit_days=6,
        resolution_m=5.0,
        agency="ESA"
    ),
    "radarsat2": SatelliteConfig(
        name="RADARSAT-2",
        wavelength_m=RADARSAT2_WAVELENGTH_M,
        band="C",
        incidence_deg=35.0,
        incidence_range=(20.0, 49.0),
        heading_asc_deg=-10.0,
        heading_desc_deg=-170.0,
        revisit_days=24,
        resolution_m=3.0,
        agency="CSA"
    ),
    "envisat": SatelliteConfig(
        name="ENVISAT",
        wavelength_m=ENVISAT_WAVELENGTH_M,
        band="C",
        incidence_deg=23.0,
        incidence_range=(15.0, 45.0),
        heading_asc_deg=-16.0,
        heading_desc_deg=-164.0,
        revisit_days=35,
        resolution_m=30.0,
        agency="ESA"
    ),

    # L-band satellites
    "alos2": SatelliteConfig(
        name="ALOS-2",
        wavelength_m=ALOS2_WAVELENGTH_M,
        band="L",
        incidence_deg=35.0,
        incidence_range=(8.0, 70.0),
        heading_asc_deg=-10.0,
        heading_desc_deg=-170.0,
        revisit_days=14,
        resolution_m=3.0,
        agency="JAXA"
    ),
    "nisar": SatelliteConfig(
        name="NISAR",
        wavelength_m=NISAR_L_WAVELENGTH_M,
        band="L",
        incidence_deg=35.0,
        incidence_range=(33.0, 47.0),
        heading_asc_deg=-12.0,
        heading_desc_deg=-168.0,
        revisit_days=12,
        resolution_m=7.0,
        agency="NASA/ISRO"
    ),
    "saocom": SatelliteConfig(
        name="SAOCOM",
        wavelength_m=SAOCOM_WAVELENGTH_M,
        band="L",
        incidence_deg=35.0,
        incidence_range=(20.0, 50.0),
        heading_asc_deg=-12.0,
        heading_desc_deg=-168.0,
        revisit_days=16,
        resolution_m=10.0,
        agency="CONAE"
    ),

    # X-band satellites
    "terrasar": SatelliteConfig(
        name="TerraSAR-X",
        wavelength_m=TERRASAR_WAVELENGTH_M,
        band="X",
        incidence_deg=35.0,
        incidence_range=(20.0, 55.0),
        heading_asc_deg=-10.0,
        heading_desc_deg=-170.0,
        revisit_days=11,
        resolution_m=1.0,
        agency="DLR"
    ),
    "cosmo": SatelliteConfig(
        name="COSMO-SkyMed",
        wavelength_m=COSMO_WAVELENGTH_M,
        band="X",
        incidence_deg=35.0,
        incidence_range=(25.0, 50.0),
        heading_asc_deg=-10.0,
        heading_desc_deg=-170.0,
        revisit_days=1,  # Constellation
        resolution_m=1.0,
        agency="ASI"
    ),
    "iceye": SatelliteConfig(
        name="ICEYE",
        wavelength_m=ICEYE_WAVELENGTH_M,
        band="X",
        incidence_deg=30.0,
        incidence_range=(15.0, 45.0),
        heading_asc_deg=-10.0,
        heading_desc_deg=-170.0,
        revisit_days=1,  # Constellation
        resolution_m=1.0,
        agency="ICEYE"
    ),
}


def get_satellite(name: str) -> SatelliteConfig:
    """
    Get satellite configuration by name.

    Parameters
    ----------
    name : str
        Satellite name (case-insensitive). Options:
        'sentinel1', 'radarsat2', 'envisat', 'alos2', 'nisar',
        'saocom', 'terrasar', 'cosmo', 'iceye'

    Returns
    -------
    SatelliteConfig
        Satellite configuration object
    """
    name_lower = name.lower().replace("-", "").replace("_", "").replace(" ", "")

    # Handle common aliases
    aliases = {
        "s1": "sentinel1",
        "sentinel": "sentinel1",
        "rs2": "radarsat2",
        "radarsat": "radarsat2",
        "alos": "alos2",
        "palsar2": "alos2",
        "tsx": "terrasar",
        "tdx": "terrasar",
        "csk": "cosmo",
        "cosmoskymed": "cosmo",
    }

    name_lower = aliases.get(name_lower, name_lower)

    if name_lower not in SATELLITES:
        available = ", ".join(SATELLITES.keys())
        raise ValueError(f"Unknown satellite: {name}. Available: {available}")

    return SATELLITES[name_lower]


def list_satellites() -> Dict[str, SatelliteConfig]:
    """Return dictionary of all available satellite configurations."""
    return SATELLITES.copy()
