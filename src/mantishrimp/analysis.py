"""Reusable posterior, evidence, sweep, and persistence helpers."""

from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import arviz as az
import numpy as np
import pandas as pd

from .inference import FitResult, fit_count_model

MODEL_TO_COMMON_PARAMETER: dict[str, dict[str, str]] = {
    "homo": {"mu_lambda": "lambda"},
    "Z2P": {"mu_lambda": "lambda", "p_zero": "p_zero"},
    "Dis2P": {"mu_lambda": "mu_lambda", "sigma_lambda": "sigma_lambda"},
    "hetero3": {
        "mu_lambda": "mu_lambda",
        "sigma_lambda": "sigma_lambda",
        "p_zero": "p_zero",
    },
}


def safe_slug(text: str) -> str:
    """Return a filesystem-safe, stable slug."""

    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text).strip()).strip("-._")
    return slug or "result"


def scenario_directory(root: str | Path, scenario: Mapping[str, Any]) -> Path:
    """Construct the notebook-compatible output directory for a scenario."""

    number = scenario.get("number", scenario.get("scenario", "scenario"))
    name = scenario.get("name", number)
    return Path(root) / f"{number}_{safe_slug(str(name))}"


def config_value(config: Mapping[str, Any], key: str) -> Any:
    """Return a config value with JSON lists normalised for comparisons."""

    value = config.get(key)
    return tuple(value) if isinstance(value, list) else value


def ground_truth_for_model(
    scenario: Mapping[str, Any], model_name: str
) -> dict[str, float]:
    """Map a synthetic scenario onto one model's parameter names."""

    mu = float(scenario["mu_lambda"])
    sigma = float(scenario["sigma_lambda"])
    p_zero = float(scenario["p_zero"])
    if model_name == "homo":
        return {"lambda": mu}
    if model_name == "Z2P":
        return {"lambda": mu, "p_zero": p_zero}
    if model_name == "Dis2P":
        return {"mu_lambda": mu, "sigma_lambda": sigma}
    if model_name == "hetero3":
        return {"mu_lambda": mu, "sigma_lambda": sigma, "p_zero": p_zero}
    raise ValueError(f"unknown model_name {model_name!r}")


def scenario_ground_truth_common(scenario: Mapping[str, Any]) -> dict[str, float]:
    """Return a scenario in the shared posterior-comparison parameter space."""

    return {
        "mu_lambda": float(scenario["mu_lambda"]),
        "sigma_lambda": float(scenario["sigma_lambda"]),
        "p_zero": float(scenario["p_zero"]),
    }


def scenario_title(
    scenario: Mapping[str, Any],
    *,
    scenario_labels: Mapping[str, str] | None = None,
) -> str:
    """Format a compact title for synthetic-validation scenarios."""

    scenario_id = str(scenario.get("scenario", scenario.get("number", "")))
    label = (scenario_labels or {}).get(scenario_id, scenario_id)
    return (
        rf"({label}) $\mu_\lambda={float(scenario['mu_lambda']):g},\ "
        rf"\sigma_\lambda={float(scenario['sigma_lambda']):g},\ "
        rf"\phi_0={float(scenario['p_zero']):g}$"
    )


def save_idata(idata: az.InferenceData, path: str | Path) -> Path:
    """Save an ArviZ inference data object as NetCDF."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(output)
    return output


def load_idata(path: str | Path) -> az.InferenceData:
    return az.from_netcdf(Path(path))


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_log_evidence_table(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path))


def posterior_values(
    idata: az.InferenceData,
    parameter: str,
    *,
    sample_size: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Flatten finite posterior draws and optionally subsample them."""

    if parameter not in idata.posterior:
        return np.array([], dtype=float)
    values = np.asarray(idata.posterior[parameter].values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if sample_size is not None and values.size > int(sample_size):
        rng = np.random.default_rng(seed)
        values = values[rng.choice(values.size, int(sample_size), replace=False)]
    return values


def get_common_posterior_samples(
    idata: az.InferenceData,
    *,
    model_name: str,
    common_parameter: str,
    sample_size: int | None = 6000,
    seed: int | None = None,
) -> np.ndarray:
    """Map model-specific parameter names onto a shared comparison space."""

    source = MODEL_TO_COMMON_PARAMETER.get(model_name, {}).get(common_parameter)
    if source is None:
        return np.array([], dtype=float)
    return posterior_values(idata, source, sample_size=sample_size, seed=seed)


def paired_draw_frame(
    model_name: str,
    idata: az.InferenceData,
) -> pd.DataFrame:
    """Return aligned posterior draws under common public parameter names."""

    frame = pd.DataFrame()
    for public_name, source_name in MODEL_TO_COMMON_PARAMETER.get(model_name, {}).items():
        values = posterior_values(idata, source_name)
        if values.size:
            frame[public_name] = values
    frame["model_name"] = model_name
    return frame


def bayes_factor_pairs(
    log_evidence_by_model: Mapping[str, float],
) -> pd.DataFrame:
    """Create a tidy pairwise Bayes-factor table from log evidence."""

    rows: list[dict[str, float | str]] = []
    for first, second in combinations(log_evidence_by_model, 2):
        difference = float(log_evidence_by_model[first] - log_evidence_by_model[second])
        rows.append(
            {
                "model_1": first,
                "model_2": second,
                "log_evidence_1": float(log_evidence_by_model[first]),
                "log_evidence_2": float(log_evidence_by_model[second]),
                "delta_log_evidence": difference,
                "log10_bayes_factor": difference / np.log(10.0),
                "bayes_factor": float(np.exp(difference)) if difference < 709 else np.inf,
            }
        )
    return pd.DataFrame(rows).sort_values(
        "delta_log_evidence", ascending=False, ignore_index=True
    )


def evidence_from_fits(fits: Sequence[FitResult]) -> dict[str, float]:
    return {fit.model_name: float(fit.log_evidence) for fit in fits}


def select_sweep(data: pd.DataFrame, parameter: str) -> pd.DataFrame:
    """Select rows belonging to a named comma-delimited parameter sweep."""

    if "sweep_membership" not in data:
        raise KeyError("data must contain a 'sweep_membership' column")
    membership = data["sweep_membership"].fillna("").astype(str)
    mask = membership.str.split(",").map(
        lambda names: parameter in {name.strip() for name in names}
    )
    subset = data.loc[mask].copy()
    if subset.empty:
        raise ValueError(f"no rows belong to sweep {parameter!r}")
    return subset


def summarize_replicates(
    data: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    value_column: str,
    error_style: str = "sd",
) -> pd.DataFrame:
    """Summarise replicate means with SD, SEM, or no error."""

    summary = (
        data.groupby(list(group_columns), as_index=False)[value_column]
        .agg(mean="mean", sd="std", n_replicates="count")
        .sort_values(list(group_columns))
    )
    summary["sd"] = summary["sd"].fillna(0.0)
    if error_style == "sd":
        summary["error"] = summary["sd"]
    elif error_style == "sem":
        summary["error"] = summary["sd"] / np.sqrt(
            summary["n_replicates"].clip(lower=1)
        )
    elif error_style in {"none", None}:
        summary["error"] = 0.0
    else:
        raise ValueError("error_style must be 'sd', 'sem', or 'none'")
    return summary


def summarize_bf_replicates(
    data: pd.DataFrame,
    *,
    x_column: str = "n_cells",
    value_column: str = "log10_bf_model_vs_true",
    model_column: str = "model",
    error_style: str = "sd",
) -> pd.DataFrame:
    return summarize_replicates(
        data,
        group_columns=[model_column, x_column],
        value_column=value_column,
        error_style=error_style,
    )


def sample_size_ticks(sample_sizes: Sequence[int], xscale: str = "log") -> list[int]:
    """Choose readable sample-size ticks for posterior summaries."""

    values = sorted({int(value) for value in sample_sizes})
    if xscale != "log" or len(values) <= 7:
        return values
    indices = np.unique(np.round(np.linspace(0, len(values) - 1, 7)).astype(int))
    return [values[index] for index in indices]


def top_models(
    log_evidence_by_model: Mapping[str, float],
    *,
    n: int = 2,
) -> list[str]:
    return [
        name
        for name, _ in sorted(
            log_evidence_by_model.items(), key=lambda item: item[1], reverse=True
        )[: int(n)]
    ]


def max_common_parameters(models: Sequence[str]) -> list[str]:
    """Return the union of comparable parameters for selected models."""

    preferred = ["mu_lambda", "sigma_lambda", "p_zero"]
    available = set().union(
        *(MODEL_TO_COMMON_PARAMETER.get(model, {}) for model in models)
    )
    return [parameter for parameter in preferred if parameter in available]


def run_or_load_inference(
    *,
    model_name: str,
    counts_per_cell: Sequence[int],
    exposure: float | Sequence[float],
    output_path: str | Path | None = None,
    force: bool = False,
    **fit_kwargs,
) -> FitResult | az.InferenceData:
    """Fit a model, optionally caching or loading its InferenceData."""

    path = Path(output_path) if output_path is not None else None
    if path is not None and path.exists() and not force:
        return load_idata(path)
    fit = fit_count_model(model_name, counts_per_cell, exposure, **fit_kwargs)
    if path is not None:
        save_idata(fit.idata, path)
    return fit


__all__ = [
    "MODEL_TO_COMMON_PARAMETER",
    "bayes_factor_pairs",
    "config_value",
    "evidence_from_fits",
    "get_common_posterior_samples",
    "ground_truth_for_model",
    "load_config",
    "load_idata",
    "load_log_evidence_table",
    "max_common_parameters",
    "paired_draw_frame",
    "posterior_values",
    "run_or_load_inference",
    "safe_slug",
    "sample_size_ticks",
    "save_idata",
    "scenario_directory",
    "scenario_ground_truth_common",
    "scenario_title",
    "select_sweep",
    "summarize_bf_replicates",
    "summarize_replicates",
    "top_models",
]
