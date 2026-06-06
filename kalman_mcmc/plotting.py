from . import config as _config

import matplotlib.pyplot as plt
import numpy as np


def plot_class_parameter_posteriors(samples, label, true_params=None, out_dir="plots"):
    param_specs = [
        ("gamma0", "gamma0", r"Posterior of $\gamma_0$", "gamma0"),
        ("gamma1", "gamma1", r"Posterior of $\gamma_1$", "gamma1"),
        ("sd0", "sigma_p", r"Posterior of $\sigma_0$", "sigma_p"),
        ("sd1", "sigma_r", r"Posterior of $\sigma_1$", "sigma_r"),
    ]

    true_params = {} if true_params is None else true_params

    for sample_name, true_name, title_base, file_base in param_specs:
        vals = np.asarray(samples[sample_name])

        true_val = true_params.get(true_name, None)
        true_arr = None if true_val is None else np.asarray(true_val)

        if vals.ndim == 1:
            vals_1d = vals.reshape(-1)
            mu = np.mean(vals_1d)
            sd = np.std(vals_1d, ddof=1)
            vmin = np.min(vals_1d)
            vmax = np.max(vals_1d)

            if np.isfinite(sd) and sd > 0:
                xlo = max(vmin, mu - 3.0 * sd)
                xhi = min(vmax, mu + 3.0 * sd)
                if not np.isfinite(xlo) or not np.isfinite(xhi) or xlo >= xhi:
                    xlo, xhi = vmin, vmax
            else:
                xlo, xhi = vmin, vmax

            plt.figure()
            plt.hist(vals_1d, bins=30, range=(xlo, xhi))
            if true_arr is not None:
                plt.axvline(float(true_arr), linestyle="--", color="k")
            plt.xlim(xlo, xhi)
            plt.xlabel(sample_name)
            plt.ylabel("Count")
            plt.title(title_base if label is None else f"Class {label}: {title_base}")
            plt.savefig(
                f"{out_dir}/{file_base}"
                + ("" if label is None else f"_{label}")
                + ".pdf"
            )
            plt.clf()
            plt.close()
            continue

        K = vals.shape[1]

        if true_arr is not None and true_arr.ndim == 0:
            true_arr = np.repeat(float(true_arr), K)

        for k in range(K):
            vals_k = vals[:, k].reshape(-1)
            mu = np.mean(vals_k)
            sd = np.std(vals_k, ddof=1)
            vmin = np.min(vals_k)
            vmax = np.max(vals_k)

            if np.isfinite(sd) and sd > 0:
                xlo = max(vmin, mu - 3.0 * sd)
                xhi = min(vmax, mu + 3.0 * sd)
                if not np.isfinite(xlo) or not np.isfinite(xhi) or xlo >= xhi:
                    xlo, xhi = vmin, vmax
            else:
                xlo, xhi = vmin, vmax

            plt.figure()
            plt.hist(vals_k, bins=30, range=(xlo, xhi))
            if true_arr is not None and k < len(true_arr):
                plt.axvline(true_arr[k], linestyle="--", color="k")
            plt.xlim(xlo, xhi)
            plt.xlabel(f"{sample_name}[{k}]")
            plt.ylabel("Count")
            plt.title(f"Class {k}: {title_base}")
            plt.savefig(f"{out_dir}/{file_base}_{k}.pdf")
            plt.clf()
            plt.close()

def plot_hyper_parameter_posteriors(samples, label=None, out_dir="plots"):
    hyper_names = [
        "log_tau_loc",
        "log_tau_scale",
        "b_loc",
        "b_scale",
        # "sd0_loc",
        # "sd0_scale",
        "sd1_loc",
        "sd1_scale",
    ]

    optional_names = [
        # "m0_loc_pol",
        # "m0_scale_pol",
        "P0_scale_pol",
        "P0_scale_pos",
    ]

    suffix = "" if label is None else f"_{label}"

    for name in hyper_names:
        if name not in samples:
            continue
        _plot_posterior_hist(
            samples=samples,
            name=name,
            out_path=f"{out_dir}/{name}{suffix}.pdf",
            xlabel=name,
            title=f"Posterior of {name}",
        )

    for name in optional_names:
        if name not in samples:
            continue

        vals = np.asarray(samples[name])

        if vals.ndim == 1:
            _plot_posterior_hist(
                samples=samples,
                name=name,
                out_path=f"{out_dir}/{name}{suffix}.pdf",
                xlabel=name,
                title=f"Posterior of {name}",
            )
        else:
            flat_last_dim = vals.shape[-1]
            for j in range(flat_last_dim):
                plt.figure()
                plt.hist(vals[..., j].reshape(-1), bins=30)
                plt.xlabel(f"{name}[{j}]")
                plt.ylabel("Count")
                plt.title(f"Posterior of {name}[{j}]")
                plt.savefig(f"{out_dir}/{name}_{j}{suffix}.pdf")
                plt.clf()
                plt.close()

def _plot_posterior_hist(
    samples, name, out_path, true_value=None, xlabel=None, title=None, bins=30
):
    vals = np.asarray(samples[name]).reshape(-1)

    mu = np.mean(vals)
    sd = np.std(vals, ddof=1)

    vmin = np.min(vals)
    vmax = np.max(vals)

    if np.isfinite(sd) and sd > 0:
        xlo = max(vmin, mu - 3.0 * sd)
        xhi = min(vmax, mu + 3.0 * sd)
        if not np.isfinite(xlo) or not np.isfinite(xhi) or xlo >= xhi:
            xlo, xhi = vmin, vmax
    else:
        xlo, xhi = vmin, vmax

    print(vmin, vmax)
    print(xlo, xhi)

    plt.figure()
    plt.hist(vals, bins=bins, range=(xlo, xhi))

    if true_value is not None:
        plt.axvline(true_value, linestyle="--", color="k")

    plt.xlim(xlo, xhi)
    plt.xlabel(name if xlabel is None else xlabel)
    plt.ylabel("Count")
    if title is not None:
        plt.title(title)
    plt.savefig(out_path)
    plt.clf()
    plt.close()
