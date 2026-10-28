"""Synthetic count generation for validating the Bayesian inference layer."""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

PopulationMode = Literal["homogeneous", "heterogeneous"]


def gamma_shape_rate_from_mean_sd(mean: float, sd: float) -> tuple[float, float]:
    """Convert Gamma mean/SD parameterisation to shape/rate."""

    mean, sd = float(mean), float(sd)
    if mean <= 0:
        raise ValueError("mean must be positive")
    if sd < 0:
        raise ValueError("sd must be non-negative")
    if sd == 0:
        return np.inf, np.inf
    return (mean / sd) ** 2, mean / sd**2


def sample_rates(
    n_cells: int,
    *,
    mode: PopulationMode = "homogeneous",
    mean_rate: float,
    rate_sd: float = 0.0,
    zero_fraction: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Sample homogeneous or zero-inflated Gamma cell-specific rates."""

    if n_cells < 1:
        raise ValueError("n_cells must be positive")
    if mean_rate <= 0 or rate_sd < 0:
        raise ValueError("mean_rate must be positive and rate_sd non-negative")
    if not 0 <= zero_fraction <= 1:
        raise ValueError("zero_fraction must lie in [0, 1]")
    if mode not in {"homogeneous", "heterogeneous"}:
        raise ValueError("mode must be 'homogeneous' or 'heterogeneous'")

    rng = np.random.default_rng(seed)
    if mode == "homogeneous" or rate_sd == 0:
        rates = np.full(int(n_cells), float(mean_rate), dtype=float)
    else:
        shape, rate = gamma_shape_rate_from_mean_sd(mean_rate, rate_sd)
        rates = rng.gamma(shape=shape, scale=1.0 / rate, size=int(n_cells))
    if zero_fraction:
        rates[rng.random(int(n_cells)) < zero_fraction] = 0.0
    return rates


def process_times(
    rate: float,
    duration: float,
    *,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate event times from a homogeneous Poisson process."""

    rate, duration = float(rate), float(duration)
    if rate < 0 or not np.isfinite(rate):
        raise ValueError("rate must be finite and non-negative")
    if duration < 0:
        raise ValueError("duration must be non-negative")
    if rate == 0 or duration == 0:
        return np.array([], dtype=float)
    generator = rng if rng is not None else np.random.default_rng(seed)
    time = 0.0
    events: list[float] = []
    while True:
        time += float(generator.exponential(1.0 / rate))
        if time > duration:
            break
        events.append(time)
    return np.asarray(events, dtype=float)


def simulate_single_cell(
    rate: float,
    duration: float = 1.0,
    *,
    seed: int | None = None,
) -> dict[str, np.ndarray | int]:
    """Simulate one cell and return event times and waiting-time arrays."""

    times = process_times(rate, duration, seed=seed)
    return {
        "times": times,
        "inter_event_times": np.diff(times),
        "waiting_times": np.diff(np.concatenate(([0.0], times))),
        "n_events": int(times.size),
    }


def simulate_population(
    n_cells: int,
    duration: float = 1.0,
    *,
    rates: Sequence[float] | None = None,
    mode: PopulationMode = "homogeneous",
    mean_rate: float = 1.0,
    rate_sd: float = 0.0,
    zero_fraction: float = 0.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate cell-specific rates and Poisson event counts."""

    if duration <= 0:
        raise ValueError("duration must be positive")
    rng = np.random.default_rng(seed)
    if rates is None:
        resolved_rates = sample_rates(
            n_cells,
            mode=mode,
            mean_rate=mean_rate,
            rate_sd=rate_sd,
            zero_fraction=zero_fraction,
            seed=seed,
        )
    else:
        resolved_rates = np.asarray(rates, dtype=float)
        if resolved_rates.shape != (int(n_cells),):
            raise ValueError(f"rates must have shape ({int(n_cells)},)")
        if np.any(resolved_rates < 0) or np.any(~np.isfinite(resolved_rates)):
            raise ValueError("rates must be finite and non-negative")
    counts = rng.poisson(resolved_rates * float(duration)).astype(int)
    return pd.DataFrame(
        {
            "cell_id": np.arange(int(n_cells), dtype=int),
            "lambda_i": resolved_rates,
            "count": counts,
            "exposure": float(duration),
        }
    )


def simulate_zero_inflated_gamma_poisson(
    n_cells: int,
    observation_time: float,
    mean_rate: float,
    rate_sd: float,
    zero_fraction: float,
    seed: int | None = None,
) -> pd.DataFrame:
    """Compatibility wrapper for the Orca synthetic-validation notebook."""

    return simulate_population(
        n_cells,
        observation_time,
        mode="heterogeneous",
        mean_rate=mean_rate,
        rate_sd=rate_sd,
        zero_fraction=zero_fraction,
        seed=seed,
    )


def sample_lambda(
    n_cells: int,
    mode: PopulationMode = "homogeneous",
    seed: int | None = None,
    *,
    mu_lambda: float | None = None,
    p0_lambda: float | None = None,
    sd_lambda: float | None = None,
    Dist_mode: str = "gamma",
) -> np.ndarray:
    """Orca-compatible wrapper around :func:`sample_rates`."""

    if Dist_mode.lower() != "gamma":
        raise ValueError("Dist_mode must be 'gamma'")
    if mu_lambda is None:
        raise ValueError("mu_lambda is required")
    if mode == "heterogeneous" and (sd_lambda is None or p0_lambda is None):
        raise ValueError("heterogeneous mode requires sd_lambda and p0_lambda")
    return sample_rates(
        n_cells,
        mode=mode,
        mean_rate=mu_lambda,
        rate_sd=0.0 if sd_lambda is None else sd_lambda,
        zero_fraction=0.0 if p0_lambda is None else p0_lambda,
        seed=seed,
    )


def simulate_SingleCell(
    lambda_rate: float,
    T: float = 1.0,
    seed: int | None = None,
) -> dict[str, Any]:
    """Orca-compatible single-cell output schema."""

    result = simulate_single_cell(lambda_rate, T, seed=seed)
    return {
        "times": result["times"],
        "dt": result["inter_event_times"],
        "dt_full": result["waiting_times"],
        "n_events": result["n_events"],
    }


def simulate_Population(
    n_cells: int,
    T: float = 1.0,
    *,
    rates: Sequence[float] | None = None,
    mode: PopulationMode = "homogeneous",
    seed: int | None = None,
    mu_lambda: float | None = None,
    sd_lambda: float | None = None,
    p0_lambda: float | None = None,
    Dist_mode: str = "gamma",
) -> dict[str, Any]:
    """Orca-compatible population output schema."""

    if Dist_mode.lower() != "gamma":
        raise ValueError("Dist_mode must be 'gamma'")
    if rates is None and mu_lambda is None:
        raise ValueError("mu_lambda is required when rates are not supplied")
    frame = simulate_population(
        n_cells,
        T,
        rates=rates,
        mode=mode,
        mean_rate=1.0 if mu_lambda is None else mu_lambda,
        rate_sd=0.0 if sd_lambda is None else sd_lambda,
        zero_fraction=0.0 if p0_lambda is None else p0_lambda,
        seed=seed,
    )
    return {
        "n_cells": int(n_cells),
        "max_time": float(T),
        "rates": frame["lambda_i"].to_numpy(),
        "n_events": frame["count"].to_numpy(dtype=int),
    }


__all__ = [
    "gamma_shape_rate_from_mean_sd",
    "process_times",
    "sample_lambda",
    "sample_rates",
    "simulate_Population",
    "simulate_SingleCell",
    "simulate_population",
    "simulate_single_cell",
    "simulate_zero_inflated_gamma_poisson",
]
