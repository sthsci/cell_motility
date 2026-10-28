"""Shared, explicit prior for synthetic inference experiments.

Units in the benchmark are micrometres and minutes. These are simulation design
choices, not measurements or estimates for a biological cell population.
"""
from __future__ import annotations

import numpy as np

PARAMETER_NAMES = (
    "log_speed", "log_tau", "log_jitter",
    "delta_speed", "delta_tau", "delta_jitter",
)
PRIOR_LOW = np.array([np.log(2.0), np.log(0.8), np.log(0.4), -1.2, -1.2, -0.5])
PRIOR_HIGH = np.array([np.log(5.0), np.log(3.0), np.log(1.2), 0.5, 0.5, 1.2])
# Convenient aliases for external benchmarking scripts.
LOW, HIGH = PRIOR_LOW, PRIOR_HIGH
BASE_THETA = np.array([np.log(3.2), np.log(1.6), np.log(0.7), 0.0, 0.0, 0.0])


def sample_prior(n: int, seed: int = 0) -> np.ndarray:
    if n < 1:
        raise ValueError("n must be positive")
    return np.random.default_rng(seed).uniform(PRIOR_LOW, PRIOR_HIGH, (n, 6))


def log_prior(theta: np.ndarray) -> float:
    theta = np.asarray(theta)
    if theta.shape != (6,) or not np.all(np.isfinite(theta)):
        return -np.inf
    if np.any(theta <= PRIOR_LOW) or np.any(theta >= PRIOR_HIGH):
        return -np.inf
    return -float(np.log(PRIOR_HIGH - PRIOR_LOW).sum())


def scenario_theta(name: str) -> np.ndarray:
    """Prespecified alternatives; exp(delta) is the effect at surface contact."""
    theta = BASE_THETA.copy()
    effects = {
        "null": (0.0, 0.0, 0.0),
        "slowdown": (-0.8, 0.0, 0.0),
        "decorrelation": (0.0, -0.8, 0.0),
        "jitter": (0.0, 0.0, 0.8),
        "combined": (-0.8, -0.8, 0.8),
        "weak": (-0.3, -0.3, 0.3),
    }
    if name not in effects:
        raise ValueError(f"unknown scenario: {name}")
    theta[3:] = effects[name]
    return theta
