"""Synthetic killer–target motility inference and validation tools."""

from .simulator import PARAMETER_NAMES, SimConfig, from_mantishrimp_config, load_dataset, proximity, save_dataset, simulate

__all__ = ["PARAMETER_NAMES", "SimConfig", "simulate", "proximity", "save_dataset", "load_dataset", "from_mantishrimp_config"]
