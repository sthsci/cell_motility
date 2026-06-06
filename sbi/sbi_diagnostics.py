import argparse
import csv
import json
import math
import os
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_PARAMS = ("gamma0", "gamma1", "sd1")
CENTRAL_LEVELS = (0.5, 0.8, 0.9, 0.95)


PARAM_ALIASES = {
    "gamma0": ("gamma0", lambda x: x),
    "log_gamma0": ("gamma0", lambda x: np.log(-x)),
    "log_rate": ("gamma0", lambda x: np.log(-x)),
    "rate": ("gamma0", lambda x: -x),
    "gamma1": ("gamma1", lambda x: x),
    "log_gamma1": ("gamma1", np.log),
    "log_coupling": ("gamma1", np.log),
    "coupling": ("gamma1", lambda x: x),
    "sd0": ("sd0", lambda x: x),
    "sigma_p": ("sd0", lambda x: x),
    "log_sd0": ("sd0", np.log),
    "log_sigma_p": ("sd0", np.log),
    "sd1": ("sd1", lambda x: x),
    "sigma_r": ("sd1", lambda x: x),
    "log_sd1": ("sd1", np.log),
    "log_sigma_r": ("sd1", np.log),
}


TRUTH_DATASET_MAP = {
    "gamma0": "gamma1_set",
    "gamma1": "gamma2_set",
    "sd0": "sigma_p_set",
    "sd1": "sigma_r_set",
}


PER_CELL_TRUTH_DATASET_MAP = {
    "gamma0": "gamma1",
    "gamma1": "gamma2",
    "sd0": "sigma_p",
    "sd1": "sigma_r",
}


LABELS = {
    "gamma0": r"$\gamma_0$",
    "log_gamma0": r"$\log(-\gamma_0)$",
    "log_rate": r"$\log(-\gamma_0)$",
    "rate": r"$-\gamma_0$",
    "gamma1": r"$\gamma_1$",
    "log_gamma1": r"$\log(\gamma_1)$",
    "log_coupling": r"$\log(\gamma_1)$",
    "coupling": r"$\gamma_1$",
    "sd0": r"$\sigma_p$",
    "sigma_p": r"$\sigma_p$",
    "log_sd0": r"$\log(\sigma_p)$",
    "log_sigma_p": r"$\log(\sigma_p)$",
    "sd1": r"$\sigma_r$",
    "sigma_r": r"$\sigma_r$",
    "log_sd1": r"$\log(\sigma_r)$",
    "log_sigma_r": r"$\log(\sigma_r)$",
}


def parse_param_list(x):
    if x is None:
        return list(DEFAULT_PARAMS)
    if isinstance(x, str):
        out = []
        for part in x.replace(",", " ").split():
            if part:
                out.append(part)
        return out
    out = []
    for v in x:
        out.extend(parse_param_list(v))
    return out


def finite_rows(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    return x[np.isfinite(x).all(axis=1)]


def attr_text(x):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def theta_raw_columns(f, d):
    if "theta_raw_columns" in f.attrs:
        return [c.strip() for c in attr_text(f.attrs["theta_raw_columns"]).split(",")]
    return [f"theta_raw:{j}" for j in range(d)]


def base_param(name):
    if name not in PARAM_ALIASES:
        return name
    return PARAM_ALIASES[name][0]


def param_transform(name):
    if name not in PARAM_ALIASES:
        return lambda x: x
    return PARAM_ALIASES[name][1]


def transform_from_base(base, target, x):
    x = np.asarray(x, dtype=np.float64)

    if target == base:
        return x

    if base == "gamma0":
        if target in ("log_gamma0", "log_rate"):
            return np.log(-x)
        if target == "rate":
            return -x

    if base == "gamma1":
        if target in ("log_gamma1", "log_coupling"):
            return np.log(x)
        if target == "coupling":
            return x

    if base == "sd0":
        if target in ("log_sd0", "log_sigma_p"):
            return np.log(x)
        if target == "sigma_p":
            return x

    if base == "sd1":
        if target in ("log_sd1", "log_sigma_r"):
            return np.log(x)
        if target == "sigma_r":
            return x

    raise KeyError(
        f"Cannot transform stored base parameter {base!r} to requested parameter {target!r}."
    )


def read_sample_column(group, name):
    if name in group:
        return np.asarray(group[name][:], dtype=np.float64).reshape(-1)

    base = base_param(name)

    if base in group:
        raw = np.asarray(group[base][:], dtype=np.float64).reshape(-1)
        return transform_from_base(base, name, raw)

    if base == "gamma0":
        for stored in ("log_gamma0", "log_rate"):
            if stored in group:
                raw = -np.exp(
                    np.asarray(group[stored][:], dtype=np.float64).reshape(-1)
                )
                return transform_from_base("gamma0", name, raw)
        if "rate" in group:
            raw = -np.asarray(group["rate"][:], dtype=np.float64).reshape(-1)
            return transform_from_base("gamma0", name, raw)

    if base == "gamma1":
        for stored in ("log_gamma1", "log_coupling"):
            if stored in group:
                raw = np.exp(np.asarray(group[stored][:], dtype=np.float64).reshape(-1))
                return transform_from_base("gamma1", name, raw)
        if "coupling" in group:
            raw = np.asarray(group["coupling"][:], dtype=np.float64).reshape(-1)
            return transform_from_base("gamma1", name, raw)

    if base == "sd0":
        for stored in ("log_sd0", "log_sigma_p"):
            if stored in group:
                raw = np.exp(np.asarray(group[stored][:], dtype=np.float64).reshape(-1))
                return transform_from_base("sd0", name, raw)
        if "sigma_p" in group:
            raw = np.asarray(group["sigma_p"][:], dtype=np.float64).reshape(-1)
            return transform_from_base("sd0", name, raw)

    if base == "sd1":
        for stored in ("log_sd1", "log_sigma_r"):
            if stored in group:
                raw = np.exp(np.asarray(group[stored][:], dtype=np.float64).reshape(-1))
                return transform_from_base("sd1", name, raw)
        if "sigma_r" in group:
            raw = np.asarray(group["sigma_r"][:], dtype=np.float64).reshape(-1)
            return transform_from_base("sd1", name, raw)

    return None


def read_raw_column(raw, names, name):
    if name in names:
        return raw[:, names.index(name)]

    base = base_param(name)

    if base in names:
        return transform_from_base(base, name, raw[:, names.index(base)])

    if base == "gamma0":
        for stored in ("log_gamma0", "log_rate"):
            if stored in names:
                phys = -np.exp(raw[:, names.index(stored)])
                return transform_from_base("gamma0", name, phys)
        if "rate" in names:
            phys = -raw[:, names.index("rate")]
            return transform_from_base("gamma0", name, phys)

    if base == "gamma1":
        for stored in ("log_gamma1", "log_coupling"):
            if stored in names:
                phys = np.exp(raw[:, names.index(stored)])
                return transform_from_base("gamma1", name, phys)
        if "coupling" in names:
            phys = raw[:, names.index("coupling")]
            return transform_from_base("gamma1", name, phys)

    if base == "sd0":
        for stored in ("log_sd0", "log_sigma_p"):
            if stored in names:
                phys = np.exp(raw[:, names.index(stored)])
                return transform_from_base("sd0", name, phys)
        if "sigma_p" in names:
            phys = raw[:, names.index("sigma_p")]
            return transform_from_base("sd0", name, phys)

    if base == "sd1":
        for stored in ("log_sd1", "log_sigma_r"):
            if stored in names:
                phys = np.exp(raw[:, names.index(stored)])
                return transform_from_base("sd1", name, phys)
        if "sigma_r" in names:
            phys = raw[:, names.index("sigma_r")]
            return transform_from_base("sd1", name, phys)

    raise KeyError(f"Parameter {name!r} not found in theta_raw columns {names}.")


def load_posterior_samples(path, params, n=None, seed=0, prefer_raw=False):
    rng = np.random.default_rng(seed)

    with h5py.File(path, "r") as f:
        cols = []

        if not prefer_raw and "samples" in f:
            g = f["samples"]
            for p in params:
                col = read_sample_column(g, p)
                if col is None:
                    cols = []
                    break
                cols.append(col)

        if cols:
            x = np.column_stack(cols)
        elif "theta_raw" in f:
            raw = np.asarray(f["theta_raw"][:], dtype=np.float64)
            names = theta_raw_columns(f, raw.shape[1])
            x = np.column_stack([read_raw_column(raw, names, p) for p in params])
        else:
            raise KeyError(
                f"No usable 'samples' group or 'theta_raw' dataset in {path}."
            )

        weights = None
        if "weights" in f:
            weights = np.asarray(f["weights"][:], dtype=np.float64).reshape(-1)

    finite = np.isfinite(x).all(axis=1)
    x = x[finite]
    if weights is not None and len(weights) == len(finite):
        weights = weights[finite]

    if x.shape[0] == 0:
        raise ValueError(f"No finite posterior samples in {path}.")

    if weights is not None and len(weights) == x.shape[0]:
        weights = np.clip(weights, 0.0, np.inf)
        if np.isfinite(weights).all() and weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = None
    else:
        weights = None

    if n is not None and x.shape[0] > int(n):
        idx = rng.choice(
            x.shape[0],
            size=int(n),
            replace=False if weights is None else True,
            p=weights,
        )
        x = x[idx]
    elif n is not None and x.shape[0] < int(n):
        idx = rng.choice(x.shape[0], size=int(n), replace=True, p=weights)
        x = x[idx]

    return x


def select_class_state_value(arr, class_id, state_index):
    arr = np.asarray(arr)

    if arr.ndim == 0:
        return arr
    if arr.ndim == 1:
        return arr[int(class_id)]

    return arr[int(class_id), int(state_index)]


def select_per_cell_state_values(arr, idx, state_index):
    arr = np.asarray(arr)[idx]

    if arr.ndim >= 2:
        return arr[:, int(state_index)]

    return arr


def truth_from_simulation_file(simulation_file, class_id, params, state_index=-1):
    class_id = int(class_id)
    vals = []

    with h5py.File(simulation_file, "r") as f:
        has_class_truth = all(
            key in f for key in ("gamma1_set", "gamma2_set", "sigma_r_set")
        )

        if has_class_truth:
            for name in params:
                base = base_param(name)
                transform = param_transform(name)
                key = TRUTH_DATASET_MAP.get(base, base)

                if key not in f:
                    raise KeyError(
                        f"Simulation file is missing truth dataset {key!r} "
                        f"for requested parameter {name!r}."
                    )

                raw_val = select_class_state_value(f[key], class_id, state_index)
                vals.append(float(transform(np.asarray(raw_val, dtype=np.float64))))

            return np.asarray(vals, dtype=np.float64)

        needed = ("labels", "gamma1", "gamma2", "sigma_r")
        missing = [k for k in needed if k not in f]

        if missing:
            raise KeyError(
                "Simulation file lacks per-class gamma*_set truth and is missing "
                + ", ".join(missing)
                + " for per-cell truth fallback."
            )

        labels = np.asarray(f["labels"][:])
        idx = np.where(labels == class_id)[0]

        if idx.size == 0:
            raise ValueError(f"No tracks found with labels == {class_id}.")

        for name in params:
            base = base_param(name)
            transform = param_transform(name)
            key = PER_CELL_TRUTH_DATASET_MAP.get(base, base)

            if key not in f:
                raise KeyError(
                    f"Simulation file is missing per-cell truth dataset {key!r} "
                    f"for requested parameter {name!r}."
                )

            raw_vals = select_per_cell_state_values(f[key], idx, state_index)
            vals.append(
                float(np.mean(transform(np.asarray(raw_vals, dtype=np.float64))))
            )

        return np.asarray(vals, dtype=np.float64)


def available_class_ids(simulation_file):
    with h5py.File(simulation_file, "r") as f:
        if "gamma1_set" in f:
            return list(range(np.asarray(f["gamma1_set"]).shape[0]))
        if "labels" in f:
            return [int(x) for x in np.unique(np.asarray(f["labels"][:]))]
    raise KeyError(
        "Cannot infer class IDs: simulation file has neither gamma1_set nor labels."
    )


def parse_class_ids(spec, simulation_file):
    if spec is None:
        raise ValueError("Provide --class_id or --class_ids.")

    if isinstance(spec, int):
        return [int(spec)]

    spec = str(spec).strip()

    if spec == "all":
        return available_class_ids(simulation_file)

    out = []

    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue

        if ":" in part:
            pieces = part.split(":")
            if len(pieces) != 2:
                raise ValueError(f"Invalid class-id range {part!r}.")
            lo = int(pieces[0])
            hi = int(pieces[1])
            out.extend(range(lo, hi))
        else:
            out.append(int(part))

    if not out:
        raise ValueError("No class IDs resolved.")

    return sorted(dict.fromkeys(out))


def resolve_posterior_path(single, template, class_id, sampler_name):
    if template:
        return template.format(
            class_id=int(class_id), cls=int(class_id), classid=int(class_id)
        )
    if single:
        return single
    raise ValueError(
        f"Provide --posterior_{sampler_name} or --posterior_{sampler_name}_template."
    )


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_dicts_csv(path, rows):
    ensure_parent(path)

    if not rows:
        with open(path, "w", newline="") as f:
            f.write("")
        return

    fieldnames = sorted({k for r in rows for k in r.keys()})

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def save_json(path, obj):
    ensure_parent(path)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def central_interval(samples, level):
    alpha = 1.0 - float(level)
    lo = np.quantile(samples, alpha / 2.0, axis=0)
    hi = np.quantile(samples, 1.0 - alpha / 2.0, axis=0)
    return lo, hi


def posterior_summary_rows(entries, params, levels):
    rows = []
    interval_rows = []

    for e in entries:
        x = e["samples"]
        truth = e["truth"]

        mean = np.mean(x, axis=0)
        sd = np.std(x, axis=0, ddof=0)
        median = np.median(x, axis=0)
        ranks = np.mean(x <= truth[None, :], axis=0)

        for j, p in enumerate(params):
            z = (mean[j] - truth[j]) / sd[j] if sd[j] > 0 else np.nan

            rows.append(
                {
                    "class_id": e["class_id"],
                    "sampler": e["sampler"],
                    "path": e["path"],
                    "param": p,
                    "truth": truth[j],
                    "mean": mean[j],
                    "median": median[j],
                    "sd": sd[j],
                    "rank": ranks[j],
                    "z_bias": z,
                    "abs_z_bias": abs(z) if np.isfinite(z) else np.nan,
                }
            )

            for level in levels:
                lo, hi = central_interval(x[:, [j]], level)
                lo = float(lo[0])
                hi = float(hi[0])
                interval_rows.append(
                    {
                        "class_id": e["class_id"],
                        "sampler": e["sampler"],
                        "path": e["path"],
                        "param": p,
                        "level": float(level),
                        "truth": truth[j],
                        "mean": mean[j],
                        "median": median[j],
                        "sd": sd[j],
                        "lo": lo,
                        "hi": hi,
                        "covered": int(lo <= truth[j] <= hi),
                    }
                )

    return rows, interval_rows


def coverage_ladder_rows(interval_rows):
    groups = {}

    for r in interval_rows:
        key = (r["sampler"], float(r["level"]))
        groups.setdefault(key, []).append(int(r["covered"]))

    rows = []
    for (sampler, level), vals in sorted(groups.items()):
        vals = np.asarray(vals, dtype=np.float64)
        rows.append(
            {
                "sampler": sampler,
                "level": level,
                "covered": int(vals.sum()),
                "total": int(vals.size),
                "empirical_coverage": float(vals.mean()) if vals.size else np.nan,
            }
        )

    return rows


def scaled_dist(a, b, scale):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    return np.sqrt(np.sum(((a - b) / scale) ** 2, axis=-1))


def tarp_for_sampler(entries, params, n_refs, reference_mode, seed):
    rng = np.random.default_rng(seed)
    entries = list(entries)

    if len(entries) < 2:
        return None

    truths = np.stack([e["truth"] for e in entries], axis=0)
    pooled = np.concatenate([e["samples"] for e in entries], axis=0)
    scale = np.std(np.vstack([truths, pooled]), axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.0)

    u_values = []

    for _ in range(int(n_refs)):
        if reference_mode == "truth_shuffle":
            perm = rng.permutation(len(entries))
            if len(entries) > 1:
                bad = perm == np.arange(len(entries))
                if np.any(bad):
                    perm = np.roll(perm, 1)
            refs = truths[perm]
        elif reference_mode == "posterior_pool":
            idx = rng.choice(pooled.shape[0], size=len(entries), replace=True)
            refs = pooled[idx]
        else:
            raise ValueError(f"Unknown TARP reference mode {reference_mode!r}.")

        for i, e in enumerate(entries):
            d_true = scaled_dist(e["truth"], refs[i], scale)
            d_samp = scaled_dist(e["samples"], refs[i], scale)
            u_values.append(float(np.mean(d_samp <= d_true)))

    u = np.asarray(u_values, dtype=np.float64)
    alphas = np.linspace(0.0, 1.0, 101)
    empirical = np.asarray([np.mean(u <= a) for a in alphas])
    ks = float(np.max(np.abs(empirical - alphas)))

    return {
        "u": u,
        "alphas": alphas,
        "empirical": empirical,
        "ks": ks,
        "n_values": int(u.size),
    }


def run_tarp(entries, params, n_refs, reference_mode, seed):
    out = {}
    for sampler in sorted({e["sampler"] for e in entries}):
        sampler_entries = [e for e in entries if e["sampler"] == sampler]
        res = tarp_for_sampler(sampler_entries, params, n_refs, reference_mode, seed)
        if res is not None:
            out[sampler] = res
    return out


def save_tarp_outputs(out_prefix, tarp):
    summary_rows = []
    curve_rows = []

    for sampler, res in tarp.items():
        summary_rows.append(
            {
                "sampler": sampler,
                "ks": res["ks"],
                "n_values": res["n_values"],
            }
        )
        for a, y in zip(res["alphas"], res["empirical"]):
            curve_rows.append(
                {
                    "sampler": sampler,
                    "alpha": float(a),
                    "empirical": float(y),
                }
            )

    write_dicts_csv(f"{out_prefix}_tarp_summary.csv", summary_rows)
    write_dicts_csv(f"{out_prefix}_tarp_curve.csv", curve_rows)

    if not tarp:
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, label="ideal")

    for sampler, res in tarp.items():
        ax.plot(
            res["alphas"],
            res["empirical"],
            linewidth=1.5,
            label=f"{sampler} (KS={res['ks']:.3f})",
        )

    ax.set_xlabel("nominal")
    ax.set_ylabel("empirical")
    ax.set_title("TARP rank-CDF diagnostic")
    ax.legend(frameon=False)
    fig.savefig(f"{out_prefix}_tarp.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_intervals(out_prefix, interval_rows, params, levels):
    if not interval_rows:
        return

    max_level = max(float(x) for x in levels)

    for p in params:
        rows = [
            r
            for r in interval_rows
            if r["param"] == p and abs(float(r["level"]) - max_level) < 1e-12
        ]
        if not rows:
            continue

        rows = sorted(rows, key=lambda r: (int(r["class_id"]), str(r["sampler"])))
        y = np.arange(len(rows))

        fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(rows) + 1.5)))

        for i, r in enumerate(rows):
            ax.hlines(y[i], r["lo"], r["hi"], linewidth=5, alpha=0.45)
            ax.plot(r["median"], y[i], marker="o", markersize=4)
            ax.plot(r["truth"], y[i], marker="x", markersize=7)

        ax.set_yticks(y)
        ax.set_yticklabels([f"class {r['class_id']} / {r['sampler']}" for r in rows])
        ax.set_xlabel(LABELS.get(p, p))
        ax.set_title(f"{int(max_level * 100)}% posterior intervals with truth")
        fig.savefig(
            f"{out_prefix}_posterior_intervals_{p}.png", dpi=200, bbox_inches="tight"
        )
        plt.close(fig)


def plot_rank_points(out_prefix, summary_rows, params):
    if not summary_rows:
        return

    rows = sorted(
        summary_rows,
        key=lambda r: (str(r["param"]), int(r["class_id"]), str(r["sampler"])),
    )
    x = np.arange(len(rows))
    y = [float(r["rank"]) for r in rows]

    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(rows)), 4))
    ax.scatter(x, y)
    ax.axhline(0.5, linestyle="--", linewidth=1.0)
    ax.axhline(0.05, linestyle=":", linewidth=1.0)
    ax.axhline(0.95, linestyle=":", linewidth=1.0)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r['param']}\n{r['sampler']}\ncls {r['class_id']}" for r in rows],
        rotation=90,
    )
    ax.set_ylabel("posterior rank of truth")
    ax.set_title("Truth rank / PIT points")
    fig.savefig(f"{out_prefix}_rank_points.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_z_bias(out_prefix, summary_rows):
    if not summary_rows:
        return

    rows = sorted(
        summary_rows,
        key=lambda r: (int(r["class_id"]), str(r["sampler"]), str(r["param"])),
    )
    y = np.arange(len(rows))
    z = [float(r["z_bias"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.32 * len(rows) + 1.5)))
    ax.scatter(z, y)
    for v, ls in [(0, "-"), (-1, "--"), (1, "--"), (-2, ":"), (2, ":")]:
        ax.axvline(v, linestyle=ls, linewidth=1.0)

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"class {r['class_id']} / {r['sampler']} / {r['param']}" for r in rows]
    )
    ax.set_xlabel("(posterior mean - truth) / posterior sd")
    ax.set_title("Standardized posterior bias")
    fig.savefig(f"{out_prefix}_z_bias.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_coverage_ladder(out_prefix, coverage_rows):
    if not coverage_rows:
        return

    samplers = sorted({r["sampler"] for r in coverage_rows})

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, label="ideal")

    for sampler in samplers:
        rows = sorted(
            [r for r in coverage_rows if r["sampler"] == sampler],
            key=lambda r: float(r["level"]),
        )
        x = [float(r["level"]) for r in rows]
        y = [float(r["empirical_coverage"]) for r in rows]
        ax.plot(x, y, marker="o", label=sampler)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("central posterior interval level")
    ax.set_ylabel("empirical containment")
    ax.set_title("Coverage ladder")
    ax.legend(frameon=False)
    fig.savefig(f"{out_prefix}_coverage_ladder.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_corners(out_prefix, entries, params, max_plots, n_samples, seed):
    try:
        import corner
    except ImportError:
        return []

    rng = np.random.default_rng(seed)
    made = []
    labels = [LABELS.get(p, p) for p in params]

    for e in entries[: int(max_plots)]:
        x = e["samples"]
        if x.shape[0] > int(n_samples):
            idx = rng.choice(x.shape[0], size=int(n_samples), replace=False)
            x = x[idx]

        fig = corner.corner(
            x,
            labels=labels,
            truths=e["truth"],
            plot_datapoints=False,
            fill_contours=False,
            show_titles=True,
            hist_kwargs={"density": True, "histtype": "step", "linewidth": 1.5},
            contour_kwargs={"linewidths": 1.5},
        )

        path = f"{out_prefix}_corner_{e['sampler']}_class_{e['class_id']}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        made.append(path)

    return made


def pairwise_sq_dists(x, y):
    x2 = np.sum(x * x, axis=1)[:, None]
    y2 = np.sum(y * y, axis=1)[None, :]
    d = x2 + y2 - 2.0 * x @ y.T
    return np.maximum(d, 0.0)


def median_bandwidth(x, y):
    z = np.vstack([x, y])
    n = min(z.shape[0], 1000)
    rng = np.random.default_rng(0)
    if z.shape[0] > n:
        z = z[rng.choice(z.shape[0], n, replace=False)]
    d = pairwise_sq_dists(z, z)
    vals = d[np.triu_indices_from(d, k=1)]
    vals = vals[vals > 0]
    if vals.size == 0:
        return 1.0
    return float(np.sqrt(0.5 * np.median(vals)))


def mmd2_rbf(x, y, bandwidth):
    gamma = 1.0 / (2.0 * bandwidth * bandwidth)
    kxx = np.exp(-gamma * pairwise_sq_dists(x, x))
    kyy = np.exp(-gamma * pairwise_sq_dists(y, y))
    kxy = np.exp(-gamma * pairwise_sq_dists(x, y))

    n = x.shape[0]
    m = y.shape[0]

    if n > 1:
        xx = (kxx.sum() - np.trace(kxx)) / (n * (n - 1))
    else:
        xx = 0.0

    if m > 1:
        yy = (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
    else:
        yy = 0.0

    return float(xx + yy - 2.0 * kxy.mean())


def energy_distance_stat(x, y):
    dxy = np.sqrt(pairwise_sq_dists(x, y)).mean()
    dxx = np.sqrt(pairwise_sq_dists(x, x)).mean()
    dyy = np.sqrt(pairwise_sq_dists(y, y)).mean()
    return float(2.0 * dxy - dxx - dyy)


def permutation_pvalue(x, y, stat_fn, n_perm, seed):
    rng = np.random.default_rng(seed)
    obs = stat_fn(x, y)
    z = np.vstack([x, y])
    n = x.shape[0]
    count = 1

    for _ in range(int(n_perm)):
        perm = rng.permutation(z.shape[0])
        xp = z[perm[:n]]
        yp = z[perm[n:]]
        if stat_fn(xp, yp) >= obs:
            count += 1

    return obs, count / (int(n_perm) + 1.0)


def sliced_wasserstein(x, y, n_proj=128, seed=0):
    rng = np.random.default_rng(seed)
    d = x.shape[1]
    vals = []

    n = min(x.shape[0], y.shape[0])
    if x.shape[0] != n:
        x = x[rng.choice(x.shape[0], n, replace=False)]
    if y.shape[0] != n:
        y = y[rng.choice(y.shape[0], n, replace=False)]

    for _ in range(int(n_proj)):
        v = rng.normal(size=d)
        v = v / np.linalg.norm(v)
        px = np.sort(x @ v)
        py = np.sort(y @ v)
        vals.append(np.mean(np.abs(px - py)))

    return float(np.mean(vals))


def c2st_accuracy(x, y, seed):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return np.nan

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.vstack([x, y])
    labels = np.concatenate([np.zeros(x.shape[0]), np.ones(y.shape[0])])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(seed))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    scores = cross_val_score(clf, z, labels, cv=cv, scoring="accuracy")
    return float(scores.mean())


def pair_diagnostics(entries, max_samples, n_perm, seed):
    by_key = {}
    for e in entries:
        by_key.setdefault((int(e["class_id"]), str(e["sampler"])), e)

    class_ids = sorted({int(e["class_id"]) for e in entries})
    samplers = sorted({str(e["sampler"]) for e in entries})

    if len(samplers) < 2:
        return []

    a_name, b_name = samplers[:2]
    rows = []
    rng = np.random.default_rng(seed)

    for cid in class_ids:
        if (cid, a_name) not in by_key or (cid, b_name) not in by_key:
            continue

        a = by_key[(cid, a_name)]["samples"]
        b = by_key[(cid, b_name)]["samples"]

        n = min(a.shape[0], b.shape[0], int(max_samples))
        a = a[rng.choice(a.shape[0], n, replace=False if a.shape[0] >= n else True)]
        b = b[rng.choice(b.shape[0], n, replace=False if b.shape[0] >= n else True)]

        pooled_sd = np.sqrt(0.5 * (np.var(a, axis=0) + np.var(b, axis=0)))
        pooled_sd = np.where(pooled_sd > 1e-12, pooled_sd, 1.0)
        mean_diff_z = (np.mean(a, axis=0) - np.mean(b, axis=0)) / pooled_sd

        bw = median_bandwidth(a, b)

        mmd_obs, mmd_p = permutation_pvalue(
            a,
            b,
            lambda u, v: mmd2_rbf(u, v, bw),
            n_perm=n_perm,
            seed=seed + cid + 101,
        )

        energy_obs, energy_p = permutation_pvalue(
            a,
            b,
            energy_distance_stat,
            n_perm=n_perm,
            seed=seed + cid + 202,
        )

        rows.append(
            {
                "class_id": cid,
                "sampler_a": a_name,
                "sampler_b": b_name,
                "n": n,
                "c2st_accuracy": c2st_accuracy(a, b, seed + cid),
                "mmd2_rbf": mmd_obs,
                "mmd2_rbf_pvalue": mmd_p,
                "energy": energy_obs,
                "energy_pvalue": energy_p,
                "sliced_wasserstein": sliced_wasserstein(a, b, seed=seed + cid),
                "max_abs_standardized_mean_difference": float(
                    np.max(np.abs(mean_diff_z))
                ),
                "mean_abs_standardized_mean_difference": float(
                    np.mean(np.abs(mean_diff_z))
                ),
            }
        )

    return rows


def build_entries(args, class_ids, params):
    entries = []

    for cid in class_ids:
        truth = truth_from_simulation_file(
            args.simulation_file,
            cid,
            params,
            state_index=args.state_index,
        )

        pa = resolve_posterior_path(
            args.posterior_a, args.posterior_a_template, cid, "a"
        )
        xa = load_posterior_samples(
            pa,
            params,
            n=args.posterior_samples,
            seed=args.seed + int(cid),
            prefer_raw=args.prefer_raw,
        )
        entries.append(
            {
                "class_id": int(cid),
                "sampler": args.posterior_a_name,
                "path": pa,
                "truth": truth,
                "samples": xa,
            }
        )

        if args.posterior_b or args.posterior_b_template:
            pb = resolve_posterior_path(
                args.posterior_b, args.posterior_b_template, cid, "b"
            )
            xb = load_posterior_samples(
                pb,
                params,
                n=args.posterior_samples,
                seed=args.seed + int(cid) + 100000,
                prefer_raw=args.prefer_raw,
            )
            entries.append(
                {
                    "class_id": int(cid),
                    "sampler": args.posterior_b_name,
                    "path": pb,
                    "truth": truth,
                    "samples": xb,
                }
            )

    return entries


def write_internal_rows(out_prefix, entries, params):
    rows = []

    for e in entries:
        r = {
            "class_id": e["class_id"],
            "sampler": e["sampler"],
            "path": e["path"],
            "n_samples": int(e["samples"].shape[0]),
        }
        for j, p in enumerate(params):
            r[f"truth_{p}"] = float(e["truth"][j])
        rows.append(r)

    write_dicts_csv(f"{out_prefix}_internal_rows.csv", rows)


def parse_args():
    p = argparse.ArgumentParser(
        description="TARP and local posterior diagnostics from simulation truth and posterior HDF5 files."
    )

    p.add_argument(
        "--simulation_file",
        "--data_h5",
        "--data-h5",
        dest="simulation_file",
        required=True,
    )
    p.add_argument("--class_id", "--class-id", dest="class_id", type=int, default=None)
    p.add_argument("--class_ids", "--class-ids", dest="class_ids", default=None)
    p.add_argument("--state_index", type=int, default=-1)

    p.add_argument("--posterior_a", "--a", dest="posterior_a", default=None)
    p.add_argument("--posterior_a_template", dest="posterior_a_template", default=None)
    p.add_argument(
        "--posterior_a_name",
        "--label-a",
        "--label_a",
        dest="posterior_a_name",
        default="A",
    )

    p.add_argument("--posterior_b", "--b", dest="posterior_b", default=None)
    p.add_argument("--posterior_b_template", dest="posterior_b_template", default=None)
    p.add_argument(
        "--posterior_b_name",
        "--label-b",
        "--label_b",
        dest="posterior_b_name",
        default="B",
    )

    p.add_argument("--params", "--vars", nargs="*", default=None)
    p.add_argument(
        "--posterior_samples",
        "--max-samples",
        dest="posterior_samples",
        type=int,
        default=2000,
    )
    p.add_argument("--prefer_raw", action="store_true", default=False)

    p.add_argument("--tarp_refs", type=int, default=20)
    p.add_argument(
        "--tarp_reference",
        choices=["truth_shuffle", "posterior_pool"],
        default="truth_shuffle",
    )

    p.add_argument("--interval_levels", default="0.5,0.8,0.95")
    p.add_argument("--max_corner_plots", type=int, default=12)
    p.add_argument("--corner_samples", type=int, default=2000)
    p.add_argument("--no_extra_plots", action="store_true", default=False)

    p.add_argument("--pair_max_samples", type=int, default=1000)
    p.add_argument("--pair_permutations", type=int, default=199)

    p.add_argument("--out_prefix", required=True)
    p.add_argument("--seed", type=int, default=0)

    return p.parse_args()


def main():
    args = parse_args()

    params = parse_param_list(args.params)
    levels = [
        float(x) for x in str(args.interval_levels).replace(",", " ").split() if x
    ]

    if args.class_ids is not None:
        class_spec = args.class_ids
    elif args.class_id is not None:
        class_spec = str(args.class_id)
    else:
        raise ValueError("Provide --class_id or --class_ids.")

    class_ids = parse_class_ids(class_spec, args.simulation_file)

    if (
        len(class_ids) > 1
        and args.posterior_a is not None
        and args.posterior_a_template is None
    ):
        raise ValueError(
            "For multiple class IDs, use --posterior_a_template rather than a single --posterior_a."
        )

    if (
        len(class_ids) > 1
        and args.posterior_b is not None
        and args.posterior_b_template is None
    ):
        raise ValueError(
            "For multiple class IDs, use --posterior_b_template rather than a single --posterior_b."
        )

    entries = build_entries(args, class_ids, params)

    write_internal_rows(args.out_prefix, entries, params)

    summary_rows, interval_rows = posterior_summary_rows(entries, params, levels)
    coverage_rows = coverage_ladder_rows(interval_rows)

    write_dicts_csv(f"{args.out_prefix}_truth_diagnostics.csv", summary_rows)
    write_dicts_csv(f"{args.out_prefix}_posterior_intervals.csv", interval_rows)
    write_dicts_csv(f"{args.out_prefix}_coverage_ladder.csv", coverage_rows)

    tarp = run_tarp(
        entries,
        params,
        n_refs=args.tarp_refs,
        reference_mode=args.tarp_reference,
        seed=args.seed,
    )
    save_tarp_outputs(args.out_prefix, tarp)

    pair_rows = pair_diagnostics(
        entries,
        max_samples=args.pair_max_samples,
        n_perm=args.pair_permutations,
        seed=args.seed,
    )
    write_dicts_csv(f"{args.out_prefix}_local_pair_diagnostics.csv", pair_rows)

    corner_paths = []

    if not args.no_extra_plots:
        plot_intervals(args.out_prefix, interval_rows, params, levels)
        plot_rank_points(args.out_prefix, summary_rows, params)
        plot_z_bias(args.out_prefix, summary_rows)
        plot_coverage_ladder(args.out_prefix, coverage_rows)
        corner_paths = plot_corners(
            args.out_prefix,
            entries,
            params,
            max_plots=args.max_corner_plots,
            n_samples=args.corner_samples,
            seed=args.seed,
        )

    save_json(
        f"{args.out_prefix}_summary.json",
        {
            "simulation_file": args.simulation_file,
            "class_ids": class_ids,
            "params": params,
            "state_index": args.state_index,
            "n_entries": len(entries),
            "tarp_samplers": list(tarp.keys()),
            "corner_plots": corner_paths,
            "outputs": {
                "internal_rows": f"{args.out_prefix}_internal_rows.csv",
                "truth_diagnostics": f"{args.out_prefix}_truth_diagnostics.csv",
                "posterior_intervals": f"{args.out_prefix}_posterior_intervals.csv",
                "coverage_ladder": f"{args.out_prefix}_coverage_ladder.csv",
                "pair_diagnostics": f"{args.out_prefix}_local_pair_diagnostics.csv",
                "tarp_summary": f"{args.out_prefix}_tarp_summary.csv",
                "tarp_curve": f"{args.out_prefix}_tarp_curve.csv",
            },
        },
    )


if __name__ == "__main__":
    main()
