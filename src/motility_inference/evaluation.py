"""Dataset-level recovery metrics, uncertainty, and predeclared decisions."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .priors import PARAMETER_NAMES

EFFECT_ROPE = float(np.log(1.2))


def wilson_interval(successes: int, total: int, confidence: float = 0.95):
    if total < 1:
        return [float("nan"), float("nan")]
    z = norm.ppf(0.5 + confidence / 2)
    p = successes / total
    denominator = 1 + z * z / total
    midpoint = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / denominator
    return [float(midpoint - half), float(midpoint + half)]


def posterior_metrics(samples, truth):
    """One independent dataset; draws may have leading chain axes."""
    samples = np.asarray(samples).reshape(-1, 6)
    truth = np.asarray(truth)
    if not np.all(np.isfinite(samples)) or truth.shape != (6,):
        raise ValueError("finite posterior samples and a six-vector truth required")
    lo, median, hi = np.quantile(samples, [0.05, 0.5, 0.95], axis=0)
    signs = np.mean(samples > 0, axis=0)
    # Directional claims correspond to the prespecified trapped/jiggly hypothesis.
    directional = np.array([
        np.mean(samples[:, 3] < -EFFECT_ROPE),
        np.mean(samples[:, 4] < -EFFECT_ROPE),
        np.mean(samples[:, 5] > EFFECT_ROPE),
    ])
    return {
        "truth": truth.tolist(), "median": median.tolist(),
        "lower90": lo.tolist(), "upper90": hi.tolist(),
        "covered90": ((lo <= truth) & (truth <= hi)).tolist(),
        "error": (median - truth).tolist(),
        "rank_fraction": np.mean(samples < truth, axis=0).tolist(),
        "prob_positive": signs.tolist(),
        "prob_practical_hypothesis": directional.tolist(),
        "detected": (directional > 0.95).tolist(),
        "any_detected": bool(np.any(directional > 0.95)),
        "prob_equivalent": np.mean(np.abs(samples[:, 3:]) < EFFECT_ROPE, axis=0).tolist(),
        "coverage": {
            str(level): (
                (np.quantile(samples, (1 - level) / 2, axis=0) <= truth)
                & (truth <= np.quantile(samples, (1 + level) / 2, axis=0))
            ).tolist()
            for level in (0.5, 0.8, 0.9, 0.95)
        },
    }


def aggregate_metrics(records):
    """Monte Carlo uncertainty is across datasets, never across frames."""
    if not records:
        return {"n": 0}
    n = len(records)
    covered = np.asarray([r["covered90"] for r in records], dtype=int)
    errors = np.asarray([r["error"] for r in records])
    widths = np.asarray([np.subtract(r["upper90"], r["lower90"]) for r in records])
    detection = np.asarray([r["detected"] for r in records], dtype=int)
    any_detection = sum(r["any_detected"] for r in records)
    return {
        "n": n,
        "parameter_names": PARAMETER_NAMES,
        "coverage90": covered.mean(axis=0).tolist(),
        "coverage90_wilson": [wilson_interval(int(s), n) for s in covered.sum(axis=0)],
        "rmse": np.sqrt(np.mean(errors**2, axis=0)).tolist(),
        "bias": errors.mean(axis=0).tolist(),
        "mean_width90": widths.mean(axis=0).tolist(),
        "detection_rate": detection.mean(axis=0).tolist(),
        "detection_wilson": [wilson_interval(int(s), n) for s in detection.sum(axis=0)],
        "any_detection_rate": any_detection / n,
        "any_detection_wilson": wilson_interval(any_detection, n),
        "coverage_curve": {
            str(level): np.mean([r["coverage"][str(level)] for r in records], axis=0).tolist()
            for level in (0.5, 0.8, 0.9, 0.95)
        },
    }


def exposure_diagnostics(data):
    from .simulator import proximity
    cfg = data["config"]
    gate, gap, _, _ = proximity(
        data["positions"], data["targets"], cfg["box_size"],
        cfg["contact_distance"], cfg["interaction_length"],
    )
    near = gate[:, :-1] > 0.5
    far = gate[:, :-1] < 0.2
    return {
        "near_fraction": float(near.mean()), "far_fraction": float(far.mean()),
        "cells_with_near_and_far": int(np.sum(near.any(axis=1) & far.any(axis=1))),
        "n_cells": int(gate.shape[0]),
        "warning": bool(near.sum() < 30 or far.sum() < 30),
    }
