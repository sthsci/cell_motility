"""Reusable plotting functions extracted and consolidated from the Orca notebooks."""

from __future__ import annotations

from typing import Mapping, Sequence

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.ticker import NullLocator

from .analysis import (
    get_common_posterior_samples,
    max_common_parameters,
    posterior_values,
    summarize_bf_replicates,
    summarize_replicates,
    top_models,
)
from .inference import FitResult, MODEL_LABELS, MODEL_ORDER

PARAMETER_LABELS = {
    "lambda": r"$\lambda$",
    "mu_lambda": r"$\mu_\lambda$",
    "sigma_lambda": r"$\sigma_\lambda$",
    "p_zero": r"$\phi_0$",
}
MODEL_COLORS = {
    "homo": "#9BD7FF",
    "Z2P": "#56B4E9",
    "Dis2P": "#2F80ED",
    "hetero3": "#0B1F5B",
}
BF_BAND_COLORS = {
    "anecdotal": "#F7F7F7",
    "moderate": "#FFF3B0",
    "strong": "#FFD28A",
    "extreme": "#E76F51",
}


def param_display_label(parameter: str) -> str:
    """Return the standard mathematical label for one model parameter."""

    return PARAMETER_LABELS.get(parameter, parameter)


def to_even_bf_axis(
    values: Sequence[float], *, clip_log10_bf: float = 3.0
) -> np.ndarray:
    """Map signed log10 Bayes factors to equal-width evidence bands."""

    raw = np.asarray(values, dtype=float)
    sign = np.sign(raw)
    magnitude = np.abs(raw)
    bf3, bf10, bf100 = np.log10(3.0), 1.0, 2.0
    clip = max(float(clip_log10_bf), bf100 + 1e-6)
    display = np.zeros_like(magnitude)
    first = magnitude <= bf3
    display[first] = magnitude[first] / bf3
    second = (magnitude > bf3) & (magnitude <= bf10)
    display[second] = 1.0 + (magnitude[second] - bf3) / (bf10 - bf3)
    third = (magnitude > bf10) & (magnitude <= bf100)
    display[third] = 2.0 + (magnitude[third] - bf10) / (bf100 - bf10)
    fourth = magnitude > bf100
    display[fourth] = 3.0 + np.minimum(
        (magnitude[fourth] - bf100) / (clip - bf100), 1.0
    )
    return sign * display


def to_banded_bf_axis(
    log10_bf_model_vs_best: Sequence[float],
    *,
    clip_log10_evidence: float = 3.0,
) -> np.ndarray:
    """Map non-positive model-vs-best values onto equal-width BF bands."""

    values = np.asarray(log10_bf_model_vs_best, dtype=float)
    return -np.abs(to_even_bf_axis(values, clip_log10_bf=clip_log10_evidence))


def setup_even_bf_axis(ax, *, clip_log10_bf: float = 3.0) -> None:
    """Configure ticks for values transformed by :func:`to_even_bf_axis`."""

    clip = float(clip_log10_bf)
    ax.set_ylim(-4.0, 4.0)
    ax.set_yticks(np.arange(-4, 5))
    ax.set_yticklabels(
        [
            rf"$\leq-{clip:g}$",
            r"$-2$",
            r"$-1$",
            r"$-\log_{10}3$",
            "$0$",
            r"$\log_{10}3$",
            "$1$",
            "$2$",
            rf"$\geq{clip:g}$",
        ]
    )


def apply_param_ticks(
    ax,
    *,
    x_parameter: str | None = None,
    y_parameter: str | None = None,
    parameter_ticks: Mapping[str, Sequence[float]] | None = None,
    parameter_ticklabels: Mapping[str, Sequence[str]] | None = None,
) -> None:
    """Apply validated parameter-specific ticks to one Matplotlib axis."""

    if not parameter_ticks:
        return
    for axis_name, parameter in (("x", x_parameter), ("y", y_parameter)):
        if parameter not in parameter_ticks:
            continue
        ticks = list(parameter_ticks[parameter])
        getattr(ax, f"set_{axis_name}ticks")(ticks)
        if parameter_ticklabels and parameter in parameter_ticklabels:
            labels = list(parameter_ticklabels[parameter])
            if len(labels) != len(ticks):
                raise ValueError(f"tick labels for {parameter} do not match tick count")
            getattr(ax, f"set_{axis_name}ticklabels")(labels)


def _truth_for_label(ground_truth, label: str, parameter: str) -> float | None:
    if ground_truth is None:
        return None
    if parameter in ground_truth and np.isscalar(ground_truth[parameter]):
        return float(ground_truth[parameter])
    if label in ground_truth and parameter in ground_truth[label]:
        return float(ground_truth[label][parameter])
    return None


def plot_posteriors(
    idatas: Sequence[tuple[str, az.InferenceData]],
    *,
    parameters: Sequence[str],
    ground_truth: Mapping | None = None,
    parameter_labels: Mapping[str, str] | None = None,
    colors: Mapping[str, str] | Sequence[str] | None = None,
    hdi_prob: float = 0.95,
    sample_size: int | None = 20_000,
    seed: int | None = None,
    limits: Mapping[str, tuple[float, float]] | None = None,
    parameter_ticks: Mapping[str, Sequence[float]] | None = None,
    parameter_ticklabels: Mapping[str, Sequence[str]] | None = None,
    bins: int = 30,
    dpi: int = 200,
):
    """Plot lower-triangle joint/marginal posterior comparisons."""

    if not idatas:
        raise ValueError("idatas must not be empty")
    params = list(parameters)
    labels = dict(PARAMETER_LABELS)
    if parameter_labels:
        labels.update(parameter_labels)
    if colors is None:
        palette = plt.colormaps["viridis"](
            np.linspace(0.3, 0.9, max(len(idatas), 2))
        )
        color_map = {label: palette[index] for index, (label, _) in enumerate(idatas)}
    elif isinstance(colors, Mapping):
        color_map = dict(colors)
    else:
        values = list(colors)
        if len(values) != len(idatas):
            raise ValueError("colors must align with idatas")
        color_map = {label: values[index] for index, (label, _) in enumerate(idatas)}

    rng = np.random.default_rng(seed)
    samples: dict[str, dict[str, np.ndarray]] = {}
    for label, idata in idatas:
        samples[label] = {}
        for parameter in params:
            values = posterior_values(idata, parameter)
            if sample_size is not None and values.size > sample_size:
                values = values[rng.choice(values.size, sample_size, replace=False)]
            samples[label][parameter] = values

    n_parameters = len(params)
    fig, axes = plt.subplots(
        n_parameters,
        n_parameters,
        figsize=(5 * n_parameters, 5 * n_parameters),
        dpi=dpi,
        squeeze=False,
    )
    for row, row_parameter in enumerate(params):
        for column, column_parameter in enumerate(params):
            ax = axes[row, column]
            ax.xaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_minor_locator(NullLocator())
            if column > row:
                ax.axis("off")
                continue
            for label, _ in idatas:
                color = color_map.get(label, "black")
                if row == column:
                    values = samples[label][row_parameter]
                    if not values.size:
                        continue
                    sns.histplot(
                        values,
                        bins=bins,
                        stat="density",
                        element="step",
                        fill=False,
                        linewidth=1.8,
                        color=color,
                        label=label if row == 0 else None,
                        ax=ax,
                    )
                    lower, upper = az.hdi(values, hdi_prob=hdi_prob)
                    ax.axvspan(lower, upper, color=color, alpha=0.08)
                    truth = _truth_for_label(ground_truth, label, row_parameter)
                    if truth is not None:
                        ax.axvline(truth, color="tab:orange", linestyle="--", linewidth=2)
                else:
                    x_values = samples[label][column_parameter]
                    y_values = samples[label][row_parameter]
                    size = min(x_values.size, y_values.size)
                    if size < 8:
                        continue
                    sns.kdeplot(
                        x=x_values[:size],
                        y=y_values[:size],
                        levels=6,
                        fill=False,
                        color=color,
                        linewidths=1.3,
                        ax=ax,
                    )
                    x_truth = _truth_for_label(ground_truth, label, column_parameter)
                    y_truth = _truth_for_label(ground_truth, label, row_parameter)
                    if x_truth is not None:
                        ax.axvline(x_truth, color="tab:orange", linestyle="--")
                    if y_truth is not None:
                        ax.axhline(y_truth, color="tab:orange", linestyle="--")
            ax.set_xlabel(labels.get(column_parameter, column_parameter))
            ax.set_ylabel(
                "Density" if row == column else labels.get(row_parameter, row_parameter)
            )
            if limits and column_parameter in limits:
                ax.set_xlim(*limits[column_parameter])
            if limits and row != column and row_parameter in limits:
                ax.set_ylim(*limits[row_parameter])
            apply_param_ticks(
                ax,
                x_parameter=column_parameter,
                y_parameter=None if row == column else row_parameter,
                parameter_ticks=parameter_ticks,
                parameter_ticklabels=parameter_ticklabels,
            )
            ax.grid(alpha=0.2)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, legend_labels, loc="upper center", ncol=len(handles))
    fig.tight_layout()
    return fig, axes


def plot_model_posteriors(
    fits: Sequence[FitResult],
    *,
    parameters: Sequence[str] = ("mu_lambda", "sigma_lambda", "p_zero"),
    ground_truth: Mapping[str, float] | None = None,
    sample_size: int = 6000,
    seed: int | None = None,
    **plot_kwargs,
):
    """Compare all model posteriors in a shared parameter space."""

    idatas = []
    converted: dict[str, az.InferenceData] = {}
    for fit_index, fit in enumerate(fits):
        posterior = {}
        for parameter in parameters:
            values = get_common_posterior_samples(
                fit.idata,
                model_name=fit.model_name,
                common_parameter=parameter,
                sample_size=sample_size,
                seed=None if seed is None else seed + fit_index,
            )
            if values.size:
                posterior[parameter] = values[None, :]
        if posterior:
            converted[fit.model_name] = az.from_dict(posterior=posterior)
            idatas.append((fit.model_label, converted[fit.model_name]))
    color_map = {
        MODEL_LABELS[name]: MODEL_COLORS[name]
        for name in converted
    }
    return plot_posteriors(
        idatas,
        parameters=parameters,
        ground_truth=ground_truth,
        colors=color_map,
        sample_size=sample_size,
        seed=seed,
        **plot_kwargs,
    )


def plot_hdi_summary(
    idatas_by_sample_size: Mapping[int, az.InferenceData],
    *,
    parameters: Sequence[str],
    ground_truth: Mapping[str, float] | None = None,
    hdi_prob: float = 0.95,
    xscale: str = "log",
    parameter_labels: Mapping[str, str] | None = None,
    dpi: int = 180,
):
    """Plot posterior medians and HDIs against simulated sample size."""

    labels = dict(PARAMETER_LABELS)
    if parameter_labels:
        labels.update(parameter_labels)
    sizes = sorted(idatas_by_sample_size)
    fig, axes = plt.subplots(
        len(parameters), 1, figsize=(8, 3.8 * len(parameters)), dpi=dpi, squeeze=False
    )
    for axis, parameter in zip(axes[:, 0], parameters):
        medians, lowers, uppers, valid_sizes = [], [], [], []
        for size in sizes:
            values = posterior_values(idatas_by_sample_size[size], parameter)
            if not values.size:
                continue
            lower, upper = az.hdi(values, hdi_prob=hdi_prob)
            valid_sizes.append(size)
            medians.append(float(np.median(values)))
            lowers.append(float(lower))
            uppers.append(float(upper))
        valid_sizes = np.asarray(valid_sizes)
        medians = np.asarray(medians)
        axis.errorbar(
            valid_sizes,
            medians,
            yerr=np.vstack([medians - lowers, np.asarray(uppers) - medians]),
            marker="o",
            capsize=4,
        )
        if ground_truth and parameter in ground_truth:
            axis.axhline(ground_truth[parameter], color="tab:orange", linestyle="--")
        axis.set_xscale(xscale)
        axis.set_xlabel("Number of cells")
        axis.set_ylabel(labels.get(parameter, parameter))
        axis.grid(alpha=0.2)
    fig.tight_layout()
    return fig, axes[:, 0]


def add_bf_bands(
    ax,
    *,
    limit: float = 3.0,
    alpha: float = 0.22,
    colors: Mapping[str, str] = BF_BAND_COLORS,
    label_bands: bool = False,
) -> None:
    """Add conventional base-10 Bayes-factor evidence bands."""

    bf3 = float(np.log10(3.0))
    bands = [
        (-limit, -2.0, "extreme"),
        (-2.0, -1.0, "strong"),
        (-1.0, -bf3, "moderate"),
        (-bf3, bf3, "anecdotal"),
        (bf3, 1.0, "moderate"),
        (1.0, 2.0, "strong"),
        (2.0, limit, "extreme"),
    ]
    for low, high, label in bands:
        if high <= -limit or low >= limit:
            continue
        low, high = max(low, -limit), min(high, limit)
        ax.axhspan(low, high, color=colors[label], alpha=alpha, linewidth=0)
        if label_bands and label != "anecdotal":
            ax.text(
                0.985,
                (low + high) / 2,
                label.capitalize(),
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                alpha=0.55,
            )
    for value in (-2.0, -1.0, -bf3, 0.0, bf3, 1.0, 2.0):
        if -limit <= value <= limit:
            ax.axhline(
                value,
                color="black",
                linestyle="--" if value == 0 else ":",
                linewidth=1.3 if value == 0 else 0.8,
                alpha=0.6,
            )
    ax.set_ylim(-limit, limit)


def plot_bayes_factor(
    log_evidence_by_model: Mapping[str, float],
    *,
    reference_model: str | None = None,
    ax=None,
    colors: Mapping[str, str] = MODEL_COLORS,
):
    """Plot log10 Bayes factors against the best or selected reference model."""

    if not log_evidence_by_model:
        raise ValueError("log_evidence_by_model must not be empty")
    if reference_model is None:
        reference_model = max(log_evidence_by_model, key=log_evidence_by_model.get)
    if reference_model not in log_evidence_by_model:
        raise KeyError(reference_model)
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(8, 4.8), dpi=180)
    models = list(log_evidence_by_model)
    reference = float(log_evidence_by_model[reference_model])
    values = [
        (reference - float(log_evidence_by_model[model])) / np.log(10.0)
        for model in models
    ]
    ax.barh(
        [MODEL_LABELS.get(model, model) for model in models],
        values,
        color=[colors.get(model, "gray") for model in models],
        edgecolor="black",
    )
    ax.set_xlabel(
        rf"$\log_{{10}}\mathrm{{BF}}({reference_model}/\mathcal{{M}})$"
    )
    ax.grid(axis="x", alpha=0.25)
    return ax.figure, ax


def plot_model_sweep(
    ax,
    data: pd.DataFrame,
    *,
    x_column: str,
    value_column: str = "log10_bf_model_vs_reference",
    model_column: str = "model",
    model_order: Sequence[str] = MODEL_ORDER,
    error_style: str = "sd",
    clip: float = 4.0,
) -> None:
    """Plot replicate-aggregated Bayes factors across a parameter sweep."""

    add_bf_bands(ax, limit=clip)
    summary = summarize_replicates(
        data,
        group_columns=[model_column, x_column],
        value_column=value_column,
        error_style=error_style,
    )
    for model in model_order:
        rows = summary[summary[model_column] == model].sort_values(x_column)
        if rows.empty:
            continue
        y = np.clip(rows["mean"].to_numpy(), -clip, clip)
        error = rows["error"].to_numpy()
        ax.errorbar(
            rows[x_column],
            y,
            yerr=error,
            marker="o",
            capsize=3,
            color=MODEL_COLORS.get(model, "gray"),
            label=MODEL_LABELS.get(model, model),
        )
    ax.set_xlabel(x_column)
    ax.set_ylabel(r"$\log_{10}\mathrm{BF}$")


def plot_bf_trajectory(
    scenario_id: str,
    *,
    data: pd.DataFrame,
    ax=None,
    scenario_column: str = "scenario",
    sample_size_column: str = "n_cells",
    value_column: str = "log10_bf_model_vs_true",
    model_column: str = "model",
    true_model_column: str = "true_model",
    error_style: str = "sd",
    clip: float = 3.0,
    xscale: str = "log",
):
    """Plot model-evidence trajectories against in-silico sample size."""

    subset = data[data[scenario_column] == scenario_id].copy()
    if subset.empty:
        raise ValueError(f"no rows found for scenario {scenario_id!r}")
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(9, 5.2), dpi=180)
    add_bf_bands(ax, limit=clip, label_bands=True)
    summary = summarize_bf_replicates(
        subset,
        x_column=sample_size_column,
        value_column=value_column,
        model_column=model_column,
        error_style=error_style,
    )
    true_model = str(subset.iloc[0][true_model_column])
    for model in MODEL_ORDER:
        rows = summary[summary[model_column] == model]
        if rows.empty:
            continue
        y = np.clip(rows["mean"].to_numpy(), -clip, clip)
        ax.errorbar(
            rows[sample_size_column],
            y,
            yerr=rows["error"],
            marker="o",
            capsize=3,
            linewidth=3 if model == true_model else 2,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model] + (" (true)" if model == true_model else ""),
        )
    ax.set_xscale(xscale)
    ax.set_xlabel("Number of in-silico killer cells")
    ax.set_ylabel(r"$\log_{10}\mathrm{BF}(\mathcal{M}/\mathcal{M}_{true})$")
    return ax.figure, ax


def draw_half_violin(
    ax,
    values: Sequence[float],
    *,
    position: float,
    side: str,
    color: str,
    width: float = 0.8,
    alpha: float = 0.55,
) -> None:
    """Draw one left or right half of a violin distribution."""

    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    parts = ax.violinplot(
        [np.asarray(values, dtype=float)],
        positions=[position],
        widths=width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in parts["bodies"]:
        vertices = body.get_paths()[0].vertices
        if side == "left":
            vertices[:, 0] = np.minimum(vertices[:, 0], position)
        else:
            vertices[:, 0] = np.maximum(vertices[:, 0], position)
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(alpha)


def plot_two_model_half_violins(
    first: FitResult,
    second: FitResult,
    *,
    parameters: Sequence[str] | None = None,
    colors: Mapping[str, str] = MODEL_COLORS,
):
    """Compare two fitted models with paired half violins."""

    if parameters is None:
        parameters = max_common_parameters([first.model_name, second.model_name])
    fig, axes = plt.subplots(1, len(parameters), figsize=(5 * len(parameters), 4.5))
    axes = np.atleast_1d(axes)
    for ax, parameter in zip(axes, parameters):
        first_values = get_common_posterior_samples(
            first.idata, model_name=first.model_name, common_parameter=parameter
        )
        second_values = get_common_posterior_samples(
            second.idata, model_name=second.model_name, common_parameter=parameter
        )
        if first_values.size:
            draw_half_violin(
                ax,
                first_values,
                position=0,
                side="left",
                color=colors[first.model_name],
            )
        if second_values.size:
            draw_half_violin(
                ax,
                second_values,
                position=0,
                side="right",
                color=colors[second.model_name],
            )
        ax.set_xticks([])
        ax.set_ylabel(PARAMETER_LABELS.get(parameter, parameter))
    handles = [
        Patch(color=colors[first.model_name], label=first.model_label, alpha=0.55),
        Patch(color=colors[second.model_name], label=second.model_label, alpha=0.55),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2)
    fig.tight_layout()
    return fig, axes


def plot_best_two_model_posteriors(
    fits: Sequence[FitResult],
    **kwargs,
):
    """Select the two highest-evidence models and compare their posteriors."""

    evidence = {fit.model_name: fit.log_evidence for fit in fits}
    selected_names = top_models(evidence, n=2)
    selected = [next(fit for fit in fits if fit.model_name == name) for name in selected_names]
    return plot_two_model_half_violins(selected[0], selected[1], **kwargs)


# Notebook-facing aliases retained for recognisable migration names.
plot_four_model_posteriors = plot_model_posteriors


__all__ = [
    "BF_BAND_COLORS",
    "MODEL_COLORS",
    "PARAMETER_LABELS",
    "add_bf_bands",
    "apply_param_ticks",
    "draw_half_violin",
    "param_display_label",
    "plot_bayes_factor",
    "plot_best_two_model_posteriors",
    "plot_bf_trajectory",
    "plot_four_model_posteriors",
    "plot_hdi_summary",
    "plot_model_posteriors",
    "plot_model_sweep",
    "plot_posteriors",
    "plot_two_model_half_violins",
    "setup_even_bf_axis",
    "to_banded_bf_axis",
    "to_even_bf_axis",
]
