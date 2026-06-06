import argparse
import importlib
import os

import corner
import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from kaljax.filter import forward
from kaljax.smooth import backward_sample
from tqdm import tqdm


def import_mcmc_module(module_name: str):
    return importlib.import_module(module_name)


def h5_attr(hf, name, default=None):
    if name not in hf.attrs:
        return default
    val = hf.attrs[name]
    if isinstance(val, bytes):
        return val.decode("utf-8")
    return val


def split_attr_paths(x):
    if x is None:
        return None
    if isinstance(x, bytes):
        x = x.decode("utf-8")
    x = str(x)
    if not x:
        return None
    return x.split("|")


def posterior_defaults(posterior_h5):
    out = {}
    with h5py.File(posterior_h5, "r") as hf:
        out["dt"] = h5_attr(hf, "dt", None)
        out["sigma_obs"] = h5_attr(hf, "sigma_obs", None)
        out["ramp_width"] = h5_attr(hf, "ramp_width", None)
        out["lambda_pre_mult"] = h5_attr(hf, "lambda_pre_mult", None)
        out["pre_coupling"] = h5_attr(hf, "pre_coupling", None)
        out["class_id"] = h5_attr(hf, "class_id", None)
        out["track_idx"] = h5_attr(hf, "track_idx", None)
        out["real_data_defaults"] = bool(h5_attr(hf, "real_data_defaults", False))
        out["h5_group"] = h5_attr(hf, "h5_group", None)
        out["channel"] = h5_attr(hf, "channel", None)
        out["data_files"] = split_attr_paths(h5_attr(hf, "data_files", None))
    if out["h5_group"] == "":
        out["h5_group"] = None
    return out


def require_value(name, value):
    if value is None:
        raise ValueError(
            f"{name} was not supplied and could not be read from the posterior HDF5 attrs."
        )
    return value


def read_posterior_samples(posterior_h5, class_id=None, n_param_samples=1000, seed=0):
    with h5py.File(posterior_h5, "r") as hf:
        group = hf["samples"] if "samples" in hf else hf

        def read_one(*names, default=None):
            for name in names:
                if name in group:
                    return np.asarray(group[name][:])
            if default is not None:
                return default
            raise KeyError(f"None of {names} found in posterior samples group.")

        gamma0 = read_one("gamma0")
        gamma1 = read_one("gamma1")
        sd1 = read_one("sd1", "sigma_r", "sd1_from_raw")
        if "sd0" in group:
            sd0 = np.asarray(group["sd0"][:])
        else:
            sd0 = np.ones_like(sd1)

    if gamma0.ndim == 2:
        if class_id is None or int(class_id) < 0:
            raise ValueError(
                "Posterior samples are class-indexed; pass --class_id to select a class."
            )
        c = int(class_id)
        gamma0 = gamma0[:, c]
        gamma1 = gamma1[:, c]
        sd1 = sd1[:, c]
        sd0 = sd0[:, c] if np.asarray(sd0).ndim == 2 else sd0

    gamma0 = np.asarray(gamma0).reshape(-1)
    gamma1 = np.asarray(gamma1).reshape(-1)
    sd1 = np.asarray(sd1).reshape(-1)
    sd0 = np.asarray(sd0).reshape(-1)
    if sd0.size == 1 and sd1.size > 1:
        sd0 = np.full_like(sd1, float(sd0[0]))

    n = gamma0.shape[0]
    k = min(int(n_param_samples), n)
    idx = np.random.default_rng(seed).choice(n, size=k, replace=False)
    return {
        "gamma0": gamma0[idx],
        "gamma1": gamma1[idx],
        "sd0": sd0[idx],
        "sd1": sd1[idx],
        "draw_idx": idx,
    }


def select_track(tracks, pad_mask, labels, onset_idx, *, class_id=0, track_id=0):
    tracks_np = np.asarray(tracks)
    mask_np = np.asarray(pad_mask).astype(bool)
    onset_np = np.asarray(onset_idx).astype(int)
    labels_np = None if labels is None else np.asarray(labels).astype(int)

    if labels_np is not None and class_id is not None and int(class_id) >= 0:
        keep = labels_np == int(class_id)
        selected_global = np.where(keep)[0]
        if selected_global.size == 0:
            raise ValueError(f"No tracks found for class_id={class_id}.")
        if track_id < 0 or track_id >= selected_global.size:
            raise IndexError(
                f"track_id={track_id} is out of range for class_id={class_id}; "
                f"there are {selected_global.size} selected tracks."
            )
        global_idx = int(selected_global[int(track_id)])
    else:
        if track_id < 0 or track_id >= tracks_np.shape[0]:
            raise IndexError(
                f"track_id={track_id} is out of range; there are {tracks_np.shape[0]} tracks."
            )
        global_idx = int(track_id)

    valid = ~mask_np[global_idx]
    return {
        "global_track_id": global_idx,
        "track_id_within_class": int(track_id),
        "y_full": tracks_np[global_idx],
        "mask_full": mask_np[global_idx],
        "onset_idx": int(onset_np[global_idx]),
        "valid": valid,
        "y_valid": tracks_np[global_idx][valid],
        "n_tracks_total": int(tracks_np.shape[0]),
        "n_tracks_class": int(np.sum(labels_np == int(class_id)))
        if labels_np is not None and class_id is not None and int(class_id) >= 0
        else int(tracks_np.shape[0]),
    }


def read_truth_if_available(data_paths, class_id, state_index=-1):
    if data_paths is None or len(data_paths) != 1:
        return None
    path = data_paths[0]
    keys = {"gamma0": "gamma1_set", "gamma1": "gamma2_set", "sd1": "sigma_r_set"}
    truth = {}
    try:
        with h5py.File(path, "r") as f:
            for canonical, h5_name in keys.items():
                if h5_name not in f:
                    return None
                arr = np.asarray(f[h5_name][:])
                if arr.ndim == 0:
                    val = arr
                elif arr.ndim == 1:
                    val = arr[int(class_id)]
                else:
                    val = arr[int(class_id), int(state_index)]
                truth[canonical] = float(val)
    except Exception:
        return None
    return truth


def smooth_or_simulate_one(
    *,
    mcmc_mod,
    y_full,
    pad_mask_full,
    onset_idx,
    gamma0,
    gamma1,
    sd0,
    sd1,
    sigma_obs,
    dt,
    ramp_width,
    lambda_pre_mult,
    pre_coupling,
    key,
    mode="smooth",
):
    dtype = jnp.asarray(y_full).dtype
    T = y_full.shape[0]
    F_seq, Q_seq = mcmc_mod.build_FQ_timevarying_for_track(
        jnp.asarray(gamma0, dtype=dtype),
        jnp.asarray(gamma1, dtype=dtype),
        jnp.asarray(sd0, dtype=dtype),
        jnp.asarray(sd1, dtype=dtype),
        jnp.asarray(onset_idx),
        T,
        dt,
        ramp_width,
        lambda_pre_mult,
        pre_coupling,
    )
    H = jnp.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=dtype)
    R = (jnp.asarray(sigma_obs, dtype=dtype) ** 2) * jnp.eye(2, dtype=dtype)
    m0 = jnp.zeros((4,), dtype=dtype)
    P0 = jnp.eye(4, dtype=dtype) * jnp.asarray(100.0, dtype=dtype)

    k1, k2 = jax.random.split(key)

    if mode == "smooth":
        res = forward(
            jnp.asarray(y_full, dtype=dtype),
            F_seq,
            H,
            Q_seq,
            R,
            m0,
            P0,
            mask=~jnp.asarray(pad_mask_full, dtype=bool),
        )
        x = backward_sample(res.m, res.P, F_seq, Q_seq, k1)
    elif mode == "replicate":
        x = simulate_latent_forward(F_seq, Q_seq, m0, P0, k1)
    else:
        raise ValueError("mode must be either 'smooth' or 'replicate'.")

    eps = jax.random.normal(k2, (T, 2), dtype=dtype)
    L_y = jnp.linalg.cholesky(R + 1e-9 * jnp.eye(2, dtype=dtype))
    y_rep = (H @ x.T).T + eps @ L_y.T
    return np.asarray(x), np.asarray(y_rep)


def simulate_latent_forward(F_seq, Q_seq, m0, P0, key):
    T = F_seq.shape[0]
    dtype = F_seq.dtype
    keys = jax.random.split(key, T + 1)
    L0 = jnp.linalg.cholesky(P0 + 1e-9 * jnp.eye(P0.shape[0], dtype=dtype))
    x0 = m0 + L0 @ jax.random.normal(keys[0], m0.shape, dtype=dtype)

    def step(x_prev, inputs):
        F, Q, k = inputs
        L = jnp.linalg.cholesky(Q + 1e-9 * jnp.eye(Q.shape[0], dtype=dtype))
        x_next = F @ x_prev + L @ jax.random.normal(k, x_prev.shape, dtype=dtype)
        return x_next, x_next

    _, xs = jax.lax.scan(step, x0, (F_seq, Q_seq, keys[1:]))
    return xs


def posterior_predictive(
    *,
    mcmc_mod,
    data_paths,
    posterior_h5,
    class_id,
    track_id,
    dt,
    sigma_obs,
    ramp_width,
    lambda_pre_mult,
    pre_coupling,
    h5_group=None,
    channel="all",
    split_channel_labels=False,
    real_data_defaults=False,
    n_param_samples=1000,
    seed=0,
    mode="smooth",
):
    tracks, pad_mask, labels, onset_idx = mcmc_mod.load_dataset(
        path=data_paths[0],
        data_paths=data_paths,
        h5_group=h5_group,
        channel=channel,
        split_channel_labels=split_channel_labels,
        real_data_defaults=real_data_defaults,
    )
    selected = select_track(
        tracks,
        pad_mask,
        labels,
        onset_idx,
        class_id=class_id,
        track_id=track_id,
    )
    posterior = read_posterior_samples(
        posterior_h5,
        class_id=class_id,
        n_param_samples=n_param_samples,
        seed=seed,
    )

    latent_samples = []
    obs_samples = []
    keys = jax.random.split(jax.random.PRNGKey(seed), len(posterior["gamma0"]))
    for i in tqdm(range(len(posterior["gamma0"])), desc="posterior predictive"):
        x, y = smooth_or_simulate_one(
            mcmc_mod=mcmc_mod,
            y_full=selected["y_full"],
            pad_mask_full=selected["mask_full"],
            onset_idx=selected["onset_idx"],
            gamma0=posterior["gamma0"][i],
            gamma1=posterior["gamma1"][i],
            sd0=posterior["sd0"][i],
            sd1=posterior["sd1"][i],
            sigma_obs=sigma_obs,
            dt=dt,
            ramp_width=ramp_width,
            lambda_pre_mult=lambda_pre_mult,
            pre_coupling=pre_coupling,
            key=keys[i],
            mode=mode,
        )
        valid = selected["valid"]
        latent_samples.append(x[valid])
        obs_samples.append(y[valid])

    return {
        "latent_samples": np.stack(latent_samples, axis=0),
        "obs_samples": np.stack(obs_samples, axis=0),
        "y_meas": selected["y_valid"],
        "selected": selected,
        "posterior": posterior,
    }


def bands(samples):
    return {
        "lo": np.quantile(samples, 0.16, axis=0),
        "med": np.quantile(samples, 0.50, axis=0),
        "hi": np.quantile(samples, 0.84, axis=0),
    }


def save_h5(result, out_h5, attrs=None):
    os.makedirs(os.path.dirname(out_h5) or ".", exist_ok=True)
    with h5py.File(out_h5, "w") as f:
        f.create_dataset(
            "latent_samples", data=result["latent_samples"], compression="gzip"
        )
        f.create_dataset("obs_samples", data=result["obs_samples"], compression="gzip")
        f.create_dataset("y_meas", data=result["y_meas"], compression="gzip")
        f.create_dataset(
            "global_track_id",
            data=np.asarray(result["selected"]["global_track_id"], dtype=np.int32),
        )
        f.create_dataset(
            "onset_idx",
            data=np.asarray(result["selected"]["onset_idx"], dtype=np.int32),
        )
        g = f.create_group("posterior_draws")
        for name in ["gamma0", "gamma1", "sd0", "sd1", "draw_idx"]:
            g.create_dataset(
                name, data=np.asarray(result["posterior"][name]), compression="gzip"
            )
        if attrs:
            for k, v in attrs.items():
                if v is not None:
                    f.attrs[k] = v


def plot_time_xy(result, dt, out_prefix):
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    obs_b = bands(result["obs_samples"])
    lat_b = bands(result["latent_samples"])
    y = result["y_meas"]
    t = np.arange(y.shape[0]) * dt

    fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for j, lab in enumerate(["x", "y"]):
        ax[j].plot(t, obs_b["med"][:, j], lw=2.5, label="posterior predictive median")
        ax[j].fill_between(
            t, obs_b["lo"][:, j], obs_b["hi"][:, j], alpha=0.2, label="68% band"
        )
        ax[j].scatter(t, y[:, j], marker="x", s=35, color="k", label="observed")
        ax[j].set_ylabel(lab)
    ax[-1].set_xlabel("t")
    ax[0].legend(loc="best")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_time.pdf")
    plt.savefig(f"{out_prefix}_time.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 7))
    n_show = min(100, result["obs_samples"].shape[0])
    show_idx = np.linspace(0, result["obs_samples"].shape[0] - 1, n_show).astype(int)
    for i in show_idx:
        ax.plot(
            result["obs_samples"][i, :, 0],
            result["obs_samples"][i, :, 1],
            alpha=0.08,
            lw=1.0,
        )
    ax.plot(
        obs_b["med"][:, 0],
        obs_b["med"][:, 1],
        lw=3.0,
        label="posterior predictive median",
    )
    ax.scatter(y[:, 0], y[:, 1], marker="x", s=45, color="k", label="observed")
    ax.scatter([y[0, 0]], [y[0, 1]], s=80, marker="o", label="start")
    ax.scatter([y[-1, 0]], [y[-1, 1]], s=80, marker="s", label="end")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_xy.pdf")
    plt.savefig(f"{out_prefix}_xy.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for j, lab in enumerate([r"$p_x$", r"$p_y$"]):
        ax[j].plot(t, lat_b["med"][:, j], lw=2.5, label="latent median")
        ax[j].fill_between(
            t, lat_b["lo"][:, j], lat_b["hi"][:, j], alpha=0.2, label="68% band"
        )
        ax[j].set_ylabel(lab)
    ax[-1].set_xlabel("t")
    ax[0].legend(loc="best")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_polarity.pdf")
    plt.savefig(f"{out_prefix}_polarity.png", dpi=200)
    plt.close(fig)


def corner_ranges(x, truths=None, rel_pad=0.05, abs_pad=1e-3):
    ranges = []
    for j in range(x.shape[1]):
        col = x[:, j]
        col = col[np.isfinite(col)]
        lo, hi = (
            (float(np.min(col)), float(np.max(col)))
            if col.size
            else (-abs_pad, abs_pad)
        )
        if truths is not None and truths[j] is not None and np.isfinite(truths[j]):
            lo = min(lo, float(truths[j]))
            hi = max(hi, float(truths[j]))
        width = hi - lo
        if not np.isfinite(width) or width <= 0:
            centre = 0.5 * (lo + hi)
            pad = max(abs_pad, rel_pad * max(abs(centre), 1.0))
        else:
            pad = max(abs_pad, rel_pad * width)
            centre = None
        ranges.append(
            (centre - pad, centre + pad) if centre is not None else (lo - pad, hi + pad)
        )
    return ranges


def plot_corner(
    posterior, data_paths, class_id, out_prefix, state_index=-1, log_sd=False
):
    samples = np.stack(
        [posterior["gamma0"], posterior["gamma1"], posterior["sd1"]], axis=1
    )
    labels = [r"$\gamma_0$", r"$\gamma_1$", r"$\sigma_r$ (sd1)"]
    truth = read_truth_if_available(data_paths, class_id, state_index=state_index)
    truths = None
    if log_sd:
        samples = samples.copy()
        samples[:, 2] = np.log(np.clip(samples[:, 2], 1e-12, None))
        labels[2] = r"$\log \sigma_r$"
    if truth is not None:
        truths = [truth["gamma0"], truth["gamma1"], truth["sd1"]]
        if log_sd:
            truths[2] = float(np.log(max(truths[2], 1e-12)))
    fig = corner.corner(
        samples,
        labels=labels,
        truths=truths,
        range=corner_ranges(samples, truths),
        show_titles=True,
        title_fmt=".3f",
        quantiles=[0.16, 0.5, 0.84],
        smooth=1.0,
    )
    suffix = "_corner_logsd" if log_sd else "_corner"
    fig.savefig(f"{out_prefix}{suffix}.pdf")
    fig.savefig(f"{out_prefix}{suffix}.png", dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description="Posterior predictive for KF/MCMC posterior files using the MCMC fit module utilities."
    )
    p.add_argument(
        "--data",
        nargs="+",
        default=None,
        help="Real-data HDF5 file(s). Implies real-data defaults unless overridden.",
    )
    p.add_argument(
        "--data_h5", default=None, help="Single simulated or exported HDF5 file."
    )
    p.add_argument("--posterior_h5", required=True)
    p.add_argument("--mcmc_module", default="ou_hmc_kj_reparam_masked_onset")
    p.add_argument("--h5_group", default=None)
    p.add_argument("--channel", default=None, choices=[None, "all", "nk", "tumour"])
    p.add_argument("--split_channel_labels", action="store_true")
    p.add_argument("--real_data_defaults", action="store_true")
    p.add_argument("--class_id", type=int, default=None)
    p.add_argument(
        "--track_id",
        type=int,
        default=0,
        help="Track index within the selected class/filter.",
    )
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--sigma_obs", type=float, default=None)
    p.add_argument("--ramp_width", type=float, default=None)
    p.add_argument("--lambda_pre_mult", type=float, default=None)
    p.add_argument("--pre_coupling", type=float, default=None)
    p.add_argument("--n_param_samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", choices=["smooth", "replicate"], default="smooth")
    p.add_argument("--out_prefix", default="posterior_predictive")
    p.add_argument("--out_h5", default=None)
    p.add_argument("--no_corner", action="store_true")
    p.add_argument("--corner_log_sd", action="store_true")
    p.add_argument("--state_index", type=int, default=-1)
    args = p.parse_args()

    defaults = posterior_defaults(args.posterior_h5)
    data_paths = args.data if args.data is not None else None
    if data_paths is None:
        if args.data_h5 is not None:
            data_paths = [args.data_h5]
        elif defaults.get("data_files") is not None:
            data_paths = defaults["data_files"]
        else:
            raise ValueError(
                "Pass --data, --data_h5, or use a posterior_h5 with data_files metadata."
            )

    real_data_defaults = bool(
        args.real_data_defaults
        or args.data is not None
        or defaults.get("real_data_defaults", False)
    )
    h5_group = args.h5_group if args.h5_group is not None else defaults.get("h5_group")
    channel = args.channel if args.channel is not None else defaults.get("channel")
    if channel is None or channel == "":
        channel = "nk" if real_data_defaults else "all"
    class_id = args.class_id
    if class_id is None:
        posterior_class = defaults.get("class_id")
        class_id = (
            0
            if posterior_class is None or int(posterior_class) < 0
            else int(posterior_class)
        )

    dt = float(
        require_value("dt", args.dt if args.dt is not None else defaults.get("dt"))
    )
    sigma_obs = float(
        require_value(
            "sigma_obs",
            args.sigma_obs if args.sigma_obs is not None else defaults.get("sigma_obs"),
        )
    )
    ramp_width = float(
        require_value(
            "ramp_width",
            args.ramp_width
            if args.ramp_width is not None
            else defaults.get("ramp_width"),
        )
    )
    lambda_pre_mult = float(
        require_value(
            "lambda_pre_mult",
            args.lambda_pre_mult
            if args.lambda_pre_mult is not None
            else defaults.get("lambda_pre_mult"),
        )
    )
    pre_coupling = float(
        args.pre_coupling
        if args.pre_coupling is not None
        else defaults.get("pre_coupling", 0.0)
    )

    mcmc_mod = import_mcmc_module(args.mcmc_module)
    result = posterior_predictive(
        mcmc_mod=mcmc_mod,
        data_paths=data_paths,
        posterior_h5=args.posterior_h5,
        class_id=class_id,
        track_id=args.track_id,
        dt=dt,
        sigma_obs=sigma_obs,
        ramp_width=ramp_width,
        lambda_pre_mult=lambda_pre_mult,
        pre_coupling=pre_coupling,
        h5_group=h5_group,
        channel=channel,
        split_channel_labels=args.split_channel_labels,
        real_data_defaults=real_data_defaults,
        n_param_samples=args.n_param_samples,
        seed=args.seed,
        mode=args.mode,
    )

    attrs = {
        "posterior_h5": args.posterior_h5,
        "data_files": "|".join(map(str, data_paths)),
        "class_id": int(class_id),
        "track_id": int(args.track_id),
        "global_track_id": int(result["selected"]["global_track_id"]),
        "mode": args.mode,
        "dt": dt,
        "sigma_obs": sigma_obs,
        "ramp_width": ramp_width,
        "lambda_pre_mult": lambda_pre_mult,
        "pre_coupling": pre_coupling,
        "h5_group": "" if h5_group is None else str(h5_group),
        "channel": str(channel),
        "real_data_defaults": bool(real_data_defaults),
        "n_tracks_class": int(result["selected"]["n_tracks_class"]),
    }
    if args.out_h5 is not None:
        save_h5(result, args.out_h5, attrs=attrs)
    plot_time_xy(result, dt=dt, out_prefix=args.out_prefix)
    if not args.no_corner:
        plot_corner(
            result["posterior"],
            data_paths,
            class_id,
            args.out_prefix,
            state_index=args.state_index,
            log_sd=args.corner_log_sd,
        )

    print(
        "Posterior predictive complete: "
        f"class_id={class_id}, track_id={args.track_id}, "
        f"global_track_id={result['selected']['global_track_id']}, "
        f"n_tracks_class={result['selected']['n_tracks_class']}."
    )


if __name__ == "__main__":
    main()
