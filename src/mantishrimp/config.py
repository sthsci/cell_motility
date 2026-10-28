"""Typed configuration for the MantiShrimp agent-based model.

The simulator uses one consistent system of user-defined spatial and time units.
The Szonja baseline interprets these as micrometres and minutes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any, Literal

BoundaryCondition = Literal["periodic", "confined"]


@dataclass(frozen=True, slots=True)
class DomainConfig:
    """Two-dimensional simulation domain."""

    x_min: float = -600.0
    x_max: float = 600.0
    y_min: float = -600.0
    y_max: float = 600.0
    boundary: BoundaryCondition = "periodic"

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    def validate(self) -> None:
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("domain maxima must be greater than minima")
        if self.boundary not in {"periodic", "confined"}:
            raise ValueError("boundary must be 'periodic' or 'confined'")


@dataclass(frozen=True, slots=True)
class MotilityConfig:
    """OU polarity and movement, with optional signed proximity responses.

    The delta parameters are log near/far ratios at gate=1. Defaults retain
    the original behavior. Changing persistence rescales polarity noise too,
    keeping its local stationary variance constant.
    """

    killer_polarity_decay: float = 2.5
    target_polarity_decay: float = 2.5
    killer_polarity_noise: float = 0.75
    target_polarity_noise: float = 0.2
    killer_speed: float = 150.0
    target_speed: float = 50.0
    killer_translation_noise: float = 0.75
    target_translation_noise: float = 0.2
    target_interaction_length: float = 10.0
    killer_delta_speed: float = 0.0
    killer_delta_tau: float = 0.0
    killer_delta_jitter: float = 0.0

    def validate(self) -> None:
        values = asdict(self)
        signed = {"killer_delta_speed", "killer_delta_tau", "killer_delta_jitter"}
        if any(not isfinite(value) for value in values.values()):
            raise ValueError("motility parameters must be finite")
        if any(value < 0 for name, value in values.items() if name not in signed):
            raise ValueError("motility parameters must be non-negative")
        if self.target_interaction_length <= 0:
            raise ValueError("target_interaction_length must be positive")


@dataclass(frozen=True, slots=True)
class MechanicsConfig:
    """Hookean repulsion and stochastic killer-target synapse mechanics."""

    repulsion_kk: float = 250.0
    repulsion_tt: float = 250.0
    repulsion_kt: float = 250.0
    adhesion_kt: float = 25.0
    binding_rate: float = 3.0
    unbinding_rate: float = 0.5
    capture_radius_factor: float = 1.5
    contact_tolerance_factor: float = 0.05

    def validate(self) -> None:
        values = asdict(self)
        if any(value < 0 for value in values.values()):
            raise ValueError("mechanical parameters must be non-negative")
        if self.capture_radius_factor < 1.0:
            raise ValueError("capture_radius_factor must be at least 1")


@dataclass(frozen=True, slots=True)
class CytotoxicityConfig:
    """Killing, cumulative target damage, recovery, and exhaustion rules."""

    killing_rate_min: float = 0.0
    killing_rate_max: float = 1.0
    death_threshold: float = 1.0
    recovery_rate: float = 0.0
    kill_probability: float = 1.0
    kill_probability_sd: float = 0.0
    kill_probability_decay_per_synapse: float = 0.0
    initial_killer_state: float = 1.0
    killer_state_sd: float = 0.0
    initial_target_state: float = 1.0
    target_state_sd: float = 0.0
    initial_death_factor: float = 0.0
    initial_death_factor_sd: float = 0.0
    exhaustion_rate_min: float = 0.001
    exhaustion_rate_max: float = 0.1
    instant_kill: bool = False

    def validate(self) -> None:
        if self.killing_rate_min < 0 or self.killing_rate_max < self.killing_rate_min:
            raise ValueError("killing rates must satisfy 0 <= min <= max")
        if self.death_threshold <= 0:
            raise ValueError("death_threshold must be positive")
        if self.recovery_rate < 0:
            raise ValueError("recovery_rate must be non-negative")
        if not 0 <= self.kill_probability <= 1:
            raise ValueError("kill_probability must lie in [0, 1]")
        if self.kill_probability_sd < 0:
            raise ValueError("kill_probability_sd must be non-negative")
        if self.kill_probability_decay_per_synapse < 0:
            raise ValueError("kill_probability_decay_per_synapse must be non-negative")
        for name in ("initial_killer_state", "initial_target_state"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        for name in ("killer_state_sd", "target_state_sd", "initial_death_factor_sd"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.initial_death_factor < 0:
            raise ValueError("initial_death_factor must be non-negative")
        if self.exhaustion_rate_min < 0 or self.exhaustion_rate_max < self.exhaustion_rate_min:
            raise ValueError("exhaustion rates must satisfy 0 <= min <= max")


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """Controls the frequency and content of recorded trajectories."""

    record_every: int = 1

    def validate(self) -> None:
        if self.record_every < 1:
            raise ValueError("record_every must be at least 1")


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Complete configuration of one MantiShrimp simulation."""

    n_killers: int = 25
    n_targets: int = 100
    killer_radius: float = 10.0
    target_radius: float = 12.0
    duration: float = 25.0
    max_dt: float = 1.0 / 25.0
    max_drift_displacement: float = 1.0
    seed: int | None = None
    initial_separation_factor: float = 1.0
    max_initialisation_attempts: int = 100_000
    domain: DomainConfig = field(default_factory=DomainConfig)
    motility: MotilityConfig = field(default_factory=MotilityConfig)
    mechanics: MechanicsConfig = field(default_factory=MechanicsConfig)
    cytotoxicity: CytotoxicityConfig = field(default_factory=CytotoxicityConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def szonja_baseline(cls, **overrides: Any) -> "SimulationConfig":
        """Return the Hookean baseline used as the package's reference profile."""

        parameters: dict[str, Any] = {"duration": 80.0}
        parameters.update(overrides)
        return cls(**parameters)

    def validate(self) -> None:
        if self.n_killers < 1 or self.n_targets < 1:
            raise ValueError("n_killers and n_targets must both be positive")
        if self.killer_radius <= 0 or self.target_radius <= 0:
            raise ValueError("cell radii must be positive")
        if self.duration <= 0 or self.max_dt <= 0:
            raise ValueError("duration and max_dt must be positive")
        if self.max_drift_displacement <= 0:
            raise ValueError("max_drift_displacement must be positive")
        if self.initial_separation_factor < 1:
            raise ValueError("initial_separation_factor must be at least 1")
        if self.max_initialisation_attempts < 1:
            raise ValueError("max_initialisation_attempts must be positive")
        self.domain.validate()
        self.motility.validate()
        self.mechanics.validate()
        self.cytotoxicity.validate()
        self.output.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
