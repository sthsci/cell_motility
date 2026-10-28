"""Fixed, observed-data-only summaries for likelihood-free migration inference.

All distance stratification uses the *start* of an interval. Future contact or
eventual successful encounter is never used to select an interval. Latent truth,
velocity, cell random effects and simulator parameter values are never read.
Summary definitions must be held fixed between training and application.
"""
from __future__ import annotations

import numpy as np

LAGS = (1, 2, 4, 8, 16)
STRATA = ("all", "contact_near", "intermediate", "far")


def observed_geometry(data: dict) -> dict[str, np.ndarray]:
    """Nearest-target surface geometry under the configured periodic domain."""
    x = np.asarray(data["positions"], dtype=float)
    target = np.asarray(data["targets"], dtype=float)
    cfg = data["config"]
    if x.ndim != 3 or x.shape[-1] != 2 or target.ndim != 3 or target.shape[-1] != 2:
        raise ValueError("positions and targets must have shape (cells, frames, 2)")
    if target.shape[0] == 0 or target.shape[1] != x.shape[1]:
        raise ValueError("at least one synchronized target trajectory is required")
    box = float(cfg["box_size"])
    contact = float(cfg.get("contact_distance", cfg.get("killer_radius", 6) + cfg.get("target_radius", 6)))
    length = float(cfg["interaction_length"])
    if box <= 0 or contact < 0 or length <= 0:
        raise ValueError("invalid geometric parameters")
    rel = target.transpose(1, 0, 2)[None] - x[:, :, None]
    rel -= box * np.round(rel / box)
    dist = np.linalg.norm(rel, axis=-1)
    finite = np.isfinite(dist)
    safe = np.where(finite, dist, np.inf)
    nearest = np.argmin(safe, axis=-1)
    min_dist = np.take_along_axis(safe, nearest[..., None], axis=-1)[..., 0]
    vector = np.take_along_axis(rel, nearest[..., None, None], axis=2)[:, :, 0]
    unit = np.divide(vector, min_dist[..., None], out=np.zeros_like(vector), where=(min_dist[..., None] > 0) & np.isfinite(min_dist[..., None]))
    gap = min_dist - contact
    valid = np.isfinite(min_dist) & np.isfinite(x).all(axis=-1)
    gate = np.exp(-np.maximum(gap, 0) / length)
    gate[~valid] = np.nan
    gap[~valid] = np.nan
    unit[~valid] = np.nan
    return {"gate": gate, "gap": gap, "unit": unit, "nearest": nearest, "valid": valid}


def _mean(values: np.ndarray, mask: np.ndarray, fallback: float = 0.0) -> float:
    good = mask & np.isfinite(values)
    return float(values[good].mean()) if good.any() else fallback


def _log_mean(values: np.ndarray, mask: np.ndarray) -> float:
    return float(np.log(max(_mean(values, mask), 1e-8)))


def _dwell_lengths(mask: np.ndarray) -> np.ndarray:
    """Observed run lengths; movie-end censoring is shared by train/test data."""
    lengths = []
    for row in mask:
        changes = np.diff(np.r_[False, row, False].astype(np.int8))
        lengths.extend((np.flatnonzero(changes == -1) - np.flatnonzero(changes == 1)).tolist())
    return np.asarray(lengths, dtype=float)


def summarize(data: dict, *, return_names: bool = False):
    """Return finite multiscale summaries of observed paired trajectories.

    Fixed frame-count lags require the same cadence and observation protocol as
    the simulation bank. Missing points are excluded, not interpolated. If a
    stratum is absent its moments use finite placeholders; explicit exposure
    fractions tell the neural posterior/ABC that it contains no information.
    """
    x = np.asarray(data["positions"], dtype=float)
    times = np.asarray(data["times"], dtype=float)
    if x.ndim != 3 or x.shape[-1] != 2 or x.shape[1] < max(LAGS) + 2:
        raise ValueError("at least 18 observation frames are required")
    if times.shape != (x.shape[1],) or not np.all(np.diff(times) > 0):
        raise ValueError("strictly increasing synchronized times are required")
    if not np.allclose(np.diff(times), np.diff(times)[0], rtol=1e-6, atol=1e-10):
        raise ValueError("SBI summaries require a regular observation cadence")
    dt = float(np.diff(times)[0])
    geometry = observed_geometry(data)
    gap, gate = geometry["gap"], geometry["gate"]
    length = float(data["config"]["interaction_length"])
    valid = geometry["valid"]
    masks = [valid, valid & (gap <= 0.5 * length), valid & (gap > 0.5 * length) & (gap <= 2 * length), valid & (gap > 2 * length)]
    out, names = [], []

    def add(name, value):
        names.append(name)
        out.append(float(value))

    dx = np.diff(x, axis=1)
    step2 = (dx ** 2).sum(axis=-1)
    finite_step = np.isfinite(dx).all(axis=-1)
    for label, mask in zip(STRATA, masks):
        add(f"{label}.fraction", float(mask.mean()))
        for lag in LAGS:
            inc = x[:, lag:] - x[:, :-lag]
            squared = (inc ** 2).sum(axis=-1)
            add(f"{label}.log_msd_lag{lag}", _log_mean(squared, mask[:, :-lag]))
        usable = mask[:, :-1] & finite_step
        add(f"{label}.log_step_fourth", _log_mean(step2 ** 2, usable))
        for lag in (1, 2, 4, 8):
            dot = (dx[:, :-lag] * dx[:, lag:]).sum(axis=-1)
            denominator = np.sqrt(step2[:, :-lag] * step2[:, lag:])
            cosine = np.divide(dot, denominator, out=np.zeros_like(dot), where=denominator > 1e-10)
            m = mask[:, :-(lag + 1)] & np.isfinite(dot) & (denominator > 1e-10)
            add(f"{label}.cosine_lag{lag}", _mean(cosine, m))
            # Covariance divided by local second moment distinguishes velocity
            # persistence from pure translational/localization noise.
            corr = _mean(dot, m) / max(_mean(step2[:, :-lag], m), 1e-8)
            add(f"{label}.autocorrelation_lag{lag}", np.clip(corr, -3, 3))
        dot1 = (dx[:, :-1] * dx[:, 1:]).sum(axis=-1)
        add(f"{label}.reversal_fraction", _mean((dot1 < 0).astype(float), mask[:, :-2] & np.isfinite(dot1)))
        radial = (dx * geometry["unit"][:, :-1]).sum(axis=-1) / dt
        add(f"{label}.radial_velocity", _mean(radial, usable))

    # Contrasts retain within-cell proximity information while marginalizing
    # heterogeneous baseline motility across independent simulated datasets.
    near, far = masks[1][:, :-1], masks[3][:, :-1]
    per_cell_msd = np.array([_mean(step2[i], finite_step[i]) for i in range(x.shape[0])])
    log_cell = np.log(np.maximum(per_cell_msd, 1e-8))
    for q in (0.1, 0.5, 0.9):
        add(f"cell.log_msd_quantile{q}", np.quantile(log_cell, q))
    contrasts = []
    for i in range(x.shape[0]):
        if np.sum(near[i] & finite_step[i]) >= 5 and np.sum(far[i] & finite_step[i]) >= 5:
            contrasts.append(_log_mean(step2[i], near[i]) - _log_mean(step2[i], far[i]))
    add("cell.near_far_eligible_fraction", len(contrasts) / x.shape[0])
    for q in (0.1, 0.5, 0.9):
        add(f"cell.near_far_log_msd_quantile{q}", np.quantile(contrasts, q) if contrasts else 0)
    occupancy = np.nanmean(gate, axis=1)
    for q in (0.1, 0.5, 0.9):
        add(f"cell.exposure_quantile{q}", np.nanquantile(occupancy, q))
    dwell = _dwell_lengths(masks[1]) * dt
    for q in (0.5, 0.9):
        add(f"near.dwell_quantile{q}", np.quantile(dwell, q) if dwell.size else 0)
    add("near.episodes_per_cell", dwell.size / x.shape[0])
    add("valid.fraction", float(np.isfinite(x).all(axis=-1).mean()))
    values = np.nan_to_num(np.asarray(out), nan=0.0, posinf=0.0, neginf=0.0)
    return (values, names) if return_names else values


def exposure_diagnostics(data: dict) -> dict:
    geometry = observed_geometry(data)
    g = geometry["gate"]
    return {
        "mean_gate": float(np.nanmean(g)),
        "near_fraction": float(np.nanmean(g >= np.exp(-0.5))),
        "far_fraction": float(np.nanmean(g < np.exp(-2.0))),
        "within_cell_gate_sd": float(np.nanmean(np.nanstd(g, axis=1))),
        "observed_fraction": float(np.isfinite(g).mean()),
    }
