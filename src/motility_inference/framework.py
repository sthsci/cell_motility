"""Guarded inference entry points for paired killer/target trajectories.

These functions consume observed ``positions``, ``targets``, ``times`` and
protocol ``config`` only. Simulator truth labels are neither required nor read.
The low-level inference functions remain available for deliberate stress tests.

``infer_neural`` checks the actual arrays and the observation protocol against
the simulation bank. A protocol shift is rejected by default. Opting in yields
exploratory posterior probabilities and an explicit abstention, not a trusted
effect classification. Even a matching protocol and unremarkable summary
diagnostics do not establish calibration on experimental data.

``infer_state_space`` permits a changed *known* localization noise scale, but
requires convergence before supporting an effect. Known mechanics or
spontaneous temporal changes omitted from its likelihood also cause abstention.
Zero-valued simulator options are assumptions, not evidence that such mechanisms
are absent from real cells. A supported effect is conditional on the fitted
movement model and does not establish intrinsic biological regulation.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .evaluation import EFFECT_ROPE, exposure_diagnostics
from .priors import PARAMETER_NAMES


_REQUIRED_PROTOCOL = (
    "n_killers", "n_targets", "n_frames", "frame_dt", "box_size",
    "contact_distance", "interaction_length", "localization_sd", "heterogeneity_sd",
)
_MECHANISM_DEFAULTS = {
    "substeps": 1, "target_speed": 0.0, "target_tau": 2.0,
    "repulsion": 0.0, "binding_rate": 0.0, "unbinding_rate": 0.5,
    "adhesion": 0.0, "capture_radius_factor": 1.5,
    "spontaneous_sd": 0.0, "spontaneous_tau": 5.0,
}
_OMITTED_SSM = ("repulsion", "adhesion", "binding_rate", "spontaneous_sd")
_INTERPRETATION = (
    "Effects describe movement under the fitted model. Target-associated movement "
    "changes do not by themselves establish intrinsic regulation: mechanical "
    "restriction, adhesion, spontaneous changes and localization error can alter "
    "the same observations. Zero-valued protocol options do not establish that "
    "these mechanisms are absent in experimental data."
)


def _as_config(value: Any) -> dict:
    if is_dataclass(value):
        value = asdict(value)
    if not isinstance(value, dict):
        raise TypeError("config must be a dictionary or dataclass")
    return dict(value)


def validate_observations(data: dict, *, max_missing_fraction: float = 0.2) -> dict:
    """Validate actual coordinates/timestamps and consistency with their config.

    A frame is valid only when both spatial coordinates are finite. NaNs denote
    missing observations; infinity is invalid. Each killer and each target must
    have at least three valid frames and no more than the allowed missing
    fraction. Every frame must contain at least one observed target. These are
    input-quality checks, not proof that a dataset contains enough information.
    """
    if not isinstance(data, dict):
        raise TypeError("data must be a dictionary")
    for name in ("positions", "targets", "times", "config"):
        if name not in data:
            raise ValueError(f"observed dataset is missing {name}")
    x = np.asarray(data["positions"], dtype=float)
    target = np.asarray(data["targets"], dtype=float)
    times = np.asarray(data["times"], dtype=float)
    if x.ndim != 3 or x.shape[0] < 1 or x.shape[1] < 3 or x.shape[2] != 2:
        raise ValueError("positions must be [at least one killer, at least 3 frames, 2]")
    if target.ndim != 3 or target.shape[0] < 1 or target.shape[1:] != x.shape[1:]:
        raise ValueError("targets must be [at least one target, matching frames, 2]")
    if np.any(np.isinf(x)) or np.any(np.isinf(target)):
        raise ValueError("infinite coordinates are invalid; use NaN for missing positions")
    if times.shape != (x.shape[1],) or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must be finite, strictly increasing, and aligned with frames")
    if not 0 <= max_missing_fraction < 1:
        raise ValueError("max_missing_fraction must lie in [0,1)")
    finite_k = np.isfinite(x).all(axis=-1)
    finite_t = np.isfinite(target).all(axis=-1)
    for name, valid in (("killer", finite_k), ("target", finite_t)):
        if np.any(valid.sum(axis=1) < 3):
            raise ValueError(f"every {name} needs at least 3 finite position frames")
        if np.any(1-valid.mean(axis=1) > max_missing_fraction+1e-12):
            raise ValueError(f"a {name} track exceeds {max_missing_fraction:.0%} missing observations")
    if not finite_t.any(axis=0).all():
        raise ValueError("every frame needs at least one finite target position")
    config = _as_config(data["config"])
    absent = [key for key in _REQUIRED_PROTOCOL if key not in config]
    if absent:
        raise ValueError("explicit observation protocol is missing: "+", ".join(absent))
    for key in _REQUIRED_PROTOCOL:
        if not np.isscalar(config[key]) or not np.isfinite(config[key]):
            raise ValueError(f"config {key} must be finite")
    actual = {"n_killers": x.shape[0], "n_targets": target.shape[0], "n_frames": x.shape[1]}
    for key, value in actual.items():
        if int(config[key]) != value or float(config[key]) != value:
            raise ValueError(f"config {key}={config[key]} disagrees with actual arrays ({value})")
    intervals = np.diff(times)
    if not np.allclose(intervals, config["frame_dt"], rtol=1e-6, atol=1e-10):
        raise ValueError("timestamps disagree with frame_dt; this guarded workflow requires regular recorded frames")
    for key in ("frame_dt", "box_size", "contact_distance", "interaction_length"):
        if config[key] <= 0:
            raise ValueError(f"config {key} must be positive")
    for key in ("localization_sd", "heterogeneity_sd"):
        if config[key] < 0:
            raise ValueError(f"config {key} must be nonnegative")
    for key in _MECHANISM_DEFAULTS:
        if key in config and (not np.isscalar(config[key]) or not np.isfinite(config[key]) or config[key] < 0):
            raise ValueError(f"config {key} must be finite and nonnegative")
    actual["frame_dt"] = float(intervals[0])
    return {
        "config": config, "actual_design": actual,
        "killer_missing_fraction": float(1-finite_k.mean()),
        "target_missing_fraction": float(1-finite_t.mean()),
        "killer_valid_counts": finite_k.sum(axis=1).tolist(),
        "target_valid_counts": finite_t.sum(axis=1).tolist(),
        "has_missing": bool(not finite_k.all() or not finite_t.all()),
    }


def _training_protocol(checkpoint_path, protocol_path, training_config):
    if training_config is not None and protocol_path is not None:
        raise ValueError("supply either training_config or protocol_path, not both")
    if training_config is not None:
        config = _as_config(training_config)
        source = "explicit training_config"
    else:
        path = Path(protocol_path) if protocol_path is not None else Path(checkpoint_path).parent/"bank_metadata.json"
        if not path.is_file():
            raise ValueError("training protocol is required; provide training_config or the bank_metadata.json path")
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("training protocol file must contain a JSON object")
        config = _as_config(metadata.get("config", metadata))
        source = str(path)
    absent = [key for key in _REQUIRED_PROTOCOL if key not in config]
    if absent:
        raise ValueError("training protocol is missing: "+", ".join(absent))
    for key in _REQUIRED_PROTOCOL:
        if not np.isscalar(config[key]) or not np.isfinite(config[key]):
            raise ValueError(f"training protocol {key} must be finite")
    return config, source


def _reference_generator_mismatch(data: dict) -> bool:
    """Known independent-generator provenance; absence is not a validation."""
    return (data.get("generator") == "mantishrimp_euler_polarity"
            or data.get("reference_substeps") is not None)


def _exposure_check(data: dict) -> dict:
    result = exposure_diagnostics(data)
    result["warning"] = bool(result["warning"] or result["cells_with_near_and_far"] == 0)
    result["criteria"] = {
        "near_gate": ">0.5", "far_gate": "<0.2",
        "minimum_intervals_per_stratum": 30,
        "minimum_cells_with_both_strata": 1,
        "interpretation": "A heuristic information guard, not a power calculation or calibration guarantee.",
    }
    return result


def effect_probabilities(samples, *, rope: float = EFFECT_ROPE, probability_threshold: float = 0.95) -> dict:
    """Truth-free marginal posterior summaries for prespecified directions.

    The practical hypothesis is lower active velocity SD, shorter persistence,
    or larger positional jitter. Other signs are also reported. The threshold
    is a posterior decision rule, not a frequentist family-wise error guarantee.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim < 2 or samples.shape[-1] != 6 or not np.isfinite(samples).all() or samples.size == 0:
        raise ValueError("finite posterior draws with final dimension six are required")
    if not np.isfinite(rope) or rope <= 0 or not 0.5 < probability_threshold <= 1:
        raise ValueError("rope must be positive and threshold must lie in (0.5,1]")
    samples = samples.reshape(-1, 6)
    result = {}
    for j in range(3, 6):
        values = samples[:, j]
        probability = float(np.mean(values < -rope) if j < 5 else np.mean(values > rope))
        result[PARAMETER_NAMES[j]] = {
            "median": float(np.median(values)), "ci90": np.quantile(values, [.05, .95]).tolist(),
            "prob_positive": float(np.mean(values > 0)), "prob_negative": float(np.mean(values < 0)),
            "prob_in_rope": float(np.mean(np.abs(values) < rope)),
            "prob_practical_hypothesis": probability,
            "direction": "decrease" if j < 5 else "increase",
            "exceeds_posterior_threshold": bool(probability > probability_threshold),
            "contact_ratio_ci90": np.exp(np.quantile(values, [.05, .95])).tolist(),
        }
    return result


def infer_neural(data: dict, checkpoint_path: str | Path, *, protocol_path: str | Path | None = None,
                 training_config=None, allow_protocol_shift: bool = False,
                 n_draws: int = 2000, seed: int = 0, probability_threshold: float = 0.95) -> dict:
    """Apply a neural posterior only after checking its training protocol.

    By default, ``bank_metadata.json`` beside the checkpoint supplies the
    training design. ``training_config`` is an explicit alternative. Changing
    localization noise, cadence, counts, geometry, or known simulator mechanisms
    is a protocol shift. Missingness is also unvalidated because this bank's
    observation model contains no missingness mechanism.

    If a shift is explicitly allowed, ``decision`` is always
    ``protocol_shift_unvalidated`` and ``abstained`` is True. For matching inputs,
    results remain approximate posterior advisories, not validated scientific
    tests. Marginal summary warnings trigger caution/abstention; lack of a warning
    does not establish that a dataset is in the training distribution.
    """
    from .sbi import NeuralPosterior
    from .summaries import summarize

    quality = validate_observations(data)
    observed = quality["config"]
    expected, source = _training_protocol(checkpoint_path, protocol_path, training_config)
    shifts = []
    for key in _REQUIRED_PROTOCOL+tuple(_MECHANISM_DEFAULTS):
        default = _MECHANISM_DEFAULTS.get(key)
        a, b = observed.get(key, default), expected.get(key, default)
        if not np.isclose(a, b, rtol=1e-6, atol=1e-10):
            shifts.append({"field": key, "observed": float(a), "training": float(b)})
    if quality["has_missing"]:
        shifts.append({"field": "missing_observations", "reason": "no missingness model in the training protocol"})
    if _reference_generator_mismatch(data):
        shifts.append({"field": "generator", "reason": "independent MantiShrimp Euler-polarity generator differs from the trained reduced OU model"})
    if shifts and not allow_protocol_shift:
        names = ", ".join(item["field"] for item in shifts)
        raise ValueError("observation protocol differs from neural training: "+names+
                         ". Retrain for this protocol, or explicitly allow_protocol_shift=True for an abstaining stress analysis.")
    model = NeuralPosterior.load(checkpoint_path)
    summary = summarize(data)
    inputs = model.input_diagnostics(summary)
    samples = model.sample(summary, n=n_draws, seed=seed)
    effects = effect_probabilities(samples, probability_threshold=probability_threshold)
    exposure = _exposure_check(data)
    candidates = [name for name, value in effects.items() if value["exceeds_posterior_threshold"]]
    cautions = [
        "The neural posterior is approximate and conditional on the simulation prior and summary statistics.",
        "Protocol agreement and unremarkable summary diagnostics do not establish experimental calibration or absence of model misspecification.",
        "The posterior threshold is not a guarantee of a 5% false-positive rate across three effects.",
    ]
    omitted = [key for key in _OMITTED_SSM if observed.get(key, 0) > 0 and expected.get(key, 0) == 0]
    if omitted:
        cautions.append("Known mechanisms absent from training: "+", ".join(omitted)+".")
    if shifts:
        decision, abstained = "protocol_shift_unvalidated", True
        cautions.append("All effect probabilities are exploratory because the observation protocol differs from training.")
    elif inputs.get("warning", False):
        decision, abstained = "summary_support_caution", True
        cautions.append("Summary diagnostics warn of unusual input; do not classify effects without validation.")
    elif exposure["warning"]:
        decision, abstained = "insufficient_proximity_contrast", True
        cautions.append("Near/far exposure fails a heuristic information guard; do not classify distance effects from this dataset.")
    else:
        decision = "advisory_effect" if candidates else "inconclusive"
        abstained = not bool(candidates)
    return {
        "method": "neural_posterior", "decision": decision, "abstained": abstained,
        "effect_probabilities": effects, "candidate_effects": candidates,
        "supported_effects": [], "samples": samples,
        "probability_threshold": float(probability_threshold), "rope": EFFECT_ROPE,
        "diagnostics": {"data_quality": quality, "input_summary": inputs,
                        "protocol_shifts": shifts, "protocol_source": source,
                        "known_omitted_components": omitted, "exposure": exposure},
        "cautions": cautions, "interpretation": _INTERPRETATION,
    }


def infer_state_space(data: dict, *, probability_threshold: float = 0.95, **fit_kwargs) -> dict:
    """Fit the hierarchical state-space model with convergence-based abstention.

    ``fit_kwargs`` are forwarded to :func:`state_space.fit`. Changed known
    localization noise is allowed; it enters that model's measurement variance.
    Its accuracy is still an assumption requiring independent calibration.
    ``supported_effect`` means conditional posterior support under this model,
    not proof of biological regulation or physical trapping. A failed diagnostic
    gate or known omitted mechanism always gives ``inconclusive``.
    """
    from .state_space import fit

    quality = validate_observations(data)
    config = quality["config"]
    # The likelihood currently requires complete target covariates. Partial
    # killer observations can be skipped, with approximate interpolated gates.
    if not np.isfinite(np.asarray(data["targets"])).all():
        raise ValueError("state-space inference currently requires complete target trajectories")
    if not np.isfinite(np.asarray(data["positions"])[:, 0]).all():
        raise ValueError("state-space inference requires finite first-frame killer positions")
    result = fit(data, **fit_kwargs)
    effects = effect_probabilities(result["samples"], probability_threshold=probability_threshold)
    candidates = [name for name, value in effects.items() if value["exceeds_posterior_threshold"]]
    omitted = [key for key in _OMITTED_SSM if config.get(key, 0) > 0]
    generator_mismatch = _reference_generator_mismatch(data)
    exposure = _exposure_check(data)
    passed = bool(result["diagnostics"].get("converged", False))
    eligible = passed and not omitted and not generator_mismatch and not exposure["warning"]
    supported = candidates if eligible else []
    cautions = list(result["diagnostics"].get("observation_model_limitations", []))
    cautions.append("Measurement noise and population heterogeneity scales are treated as known, not inferred.")
    cautions.append("Zero-valued simulator mechanism settings are assumptions, not experimental evidence that confounders are absent.")
    if not passed:
        cautions.append("MCMC did not pass the requested convergence/ESS gate; probabilities are exploratory.")
    if omitted:
        cautions.append("Known omitted components can be misattributed to motility effects: "+", ".join(omitted)+".")
    if generator_mismatch:
        cautions.append("Known independent-generator mismatch: the MantiShrimp Euler-polarity transition and initialization differ from the fitted model.")
    if exposure["warning"]:
        cautions.append("Near/far exposure fails a heuristic information guard; distance effects remain inconclusive.")
    if quality["has_missing"]:
        cautions.append("Missing killer positions are interpolated only for distance covariates; their observation updates are skipped.")
    return {
        "method": "hierarchical_state_space", "decision": "supported_effect" if supported else "inconclusive",
        "abstained": not bool(supported), "effect_probabilities": effects,
        "candidate_effects": candidates, "supported_effects": supported,
        "samples": result["samples"], "fit": result,
        "probability_threshold": float(probability_threshold), "rope": EFFECT_ROPE,
        "diagnostics": {"data_quality": quality, "sampling": result["diagnostics"],
                        "known_omitted_components": omitted, "known_generator_mismatch": generator_mismatch,
                        "exposure": exposure},
        "cautions": cautions, "interpretation": _INTERPRETATION,
    }
