"""
EQ-INSAR Generators: Create synthetic InSAR data for earthquakes.

Main functions:
- generate_synthetic_insar(): Single interferogram
- generate_timeseries(): Time series with pre/event/post frames
- generate_training_batch(): Multiple samples for ML training
"""

from .single import generate_synthetic_insar
from .timeseries import generate_timeseries
from .batch import generate_training_batch, sample_earthquake_parameters, batch_to_arrays

__all__ = [
    "generate_synthetic_insar",
    "generate_timeseries",
    "generate_training_batch",
    "sample_earthquake_parameters",
    "batch_to_arrays",
]
