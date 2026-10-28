"""MantiShrimp: killer immune cell-target cell agent-based modelling."""

from .config import (
    CytotoxicityConfig,
    DomainConfig,
    MechanicsConfig,
    MotilityConfig,
    OutputConfig,
    SimulationConfig,
)
from .results import SimulationResult
from .simulation import Simulation, simulate

__all__ = [
    "CytotoxicityConfig",
    "DomainConfig",
    "MechanicsConfig",
    "MotilityConfig",
    "OutputConfig",
    "Simulation",
    "SimulationConfig",
    "SimulationResult",
    "simulate",
]

__version__ = "0.3.0"
