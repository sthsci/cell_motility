from . import config as _config

import argparse
import h5py
import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import HMC, MCMC, NUTS
from numpyro.infer.initialization import init_to_value

from .data import load_dataset, select_state_param
from .mcmc_model import map_warmstart, ou_pol_model
from .utils import descale_samples


def fit_ou_pol(
    path="coupled_ou_tracks.h5",
    dt=1 / 3,
    sigma_obs=0.5,
    num_warmup=5000,
    num_samples=1000,
    rng_seed=0,
    label=23,
    track_idx=None,
    mask_on=False,
    out_name="mcmc_samples.h5",
    ramp_width=2.0,
    lambda_pre_mult=4.0,
    pre_coupling=0.0,
    kernel="nuts",
    prior_log_rate_loc=-0.5,
    prior_log_rate_scale=0.75,
    prior_b_loc=0.0,
    prior_b_scale=0.75,
    prior_log_sd1_loc=0.0,
    prior_log_sd1_scale=0.4,
    h5_group=None,
    channel="all",
    split_channel_labels=False,
    data_paths=None,
    real_data_defaults=False,
):

    if real_data_defaults:
        if h5_group is None:
            h5_group = "onset_motifs/all"
        if channel == "all":
            channel = "nk"

    tracks, pad_mask, labels, onset_idx = load_dataset(
        path,
        h5_group=h5_group,
        channel=channel,
        split_channel_labels=split_channel_labels,
        data_paths=data_paths,
        real_data_defaults=real_data_defaults,
    )

    if not mask_on:
        pad_mask = np.zeros_like(pad_mask)

    print("All tracks", tracks.shape)

    true_params = None
    true_gamma1 = true_gamma2 = true_sigma_p = true_sigma_r = None

    if labels is None:
        N = tracks.shape[0]
        labels = jnp.zeros(N, dtype=jnp.int32)
        K = 1
        print("No labels provided; using whole dataset as a single class")
        print("Tracks", tracks.shape)
        print("K", K)
    else:
        truth_available = False
        true_gamma1_all = true_gamma2_all = true_sigma_p_all = true_sigma_r_all = None
        if data_paths is None:
            try:
                with h5py.File(path, "r") as f:
                    required = ["gamma1_set", "gamma2_set", "sigma_p_set", "sigma_r_set"]
                    truth_available = all(k in f for k in required)
                    if truth_available:
                        true_gamma1_all = select_state_param(f["gamma1_set"][:], state_index=-1)
                        true_gamma2_all = select_state_param(f["gamma2_set"][:], state_index=-1)
                        true_sigma_p_all = select_state_param(f["sigma_p_set"][:], state_index=-1)
                        true_sigma_r_all = select_state_param(f["sigma_r_set"][:], state_index=-1)
            except OSError:
                truth_available = False

        if label is None:
            K = int(jnp.max(labels)) + 1
            if truth_available:
                true_gamma1 = true_gamma1_all
                true_gamma2 = true_gamma2_all
                true_sigma_p = true_sigma_p_all
                true_sigma_r = true_sigma_r_all
            print("Using all labels")
            print("Tracks", tracks.shape)
            print("K", K)
        else:
            keep = labels == label
            tracks = tracks[keep]
            pad_mask = pad_mask[keep]
            labels = jnp.zeros(tracks.shape[0], dtype=jnp.int32)
            onset_idx = onset_idx[keep]

            K = 1
            print("Using label", label)
            print("Label tracks", tracks.shape)
            print("K", K)

            if tracks.shape[0] == 0:
                raise ValueError(f"No tracks remain for class_id/label={label}.")

            if truth_available:
                true_gamma1 = true_gamma1_all[label]
                true_gamma2 = true_gamma2_all[label]
                true_sigma_p = true_sigma_p_all[label]
                true_sigma_r = true_sigma_r_all[label]
                true_params = {
                    "gamma0": np.asarray(true_gamma1),
                    "gamma1": np.asarray(true_gamma2),
                    "sd0": np.asarray(true_sigma_p),
                    "sd1": np.asarray(true_sigma_r),
                }

        if track_idx is not None:
            if track_idx < 0 or track_idx >= tracks.shape[0]:
                raise IndexError(
                    f"track_idx={track_idx} out of range for {tracks.shape[0]} retained tracks"
                )

            tracks = tracks[track_idx : track_idx + 1]
            pad_mask = pad_mask[track_idx : track_idx + 1]
            onset_idx = onset_idx[track_idx : track_idx + 1]
            labels = jnp.zeros(1, dtype=jnp.int32)

            if true_params is not None:
                true_params = {
                    "gamma0": np.asarray(true_params["gamma0"]).reshape(-1)[0]
                    if np.asarray(true_params["gamma0"]).ndim == 0
                    else np.asarray(true_params["gamma0"])[track_idx],
                    "gamma1": np.asarray(true_params["gamma1"]).reshape(-1)[0]
                    if np.asarray(true_params["gamma1"]).ndim == 0
                    else np.asarray(true_params["gamma1"])[track_idx],
                    "sd0": np.asarray(true_params["sd0"]).reshape(-1)[0]
                    if np.asarray(true_params["sd0"]).ndim == 0
                    else np.asarray(true_params["sd0"])[track_idx],
                    "sd1": np.asarray(true_params["sd1"]).reshape(-1)[0]
                    if np.asarray(true_params["sd1"]).ndim == 0
                    else np.asarray(true_params["sd1"])[track_idx],
                }

            K = 1
            print("Using single retained track", track_idx)
            print("Single-track shape", tracks.shape)
            print("K", K)

    dr = tracks[:, 1:] - tracks[:, :-1]  # (N,T-1,2)
    s = jnp.median(jnp.sqrt(jnp.sum(dr**2, axis=-1)))  # scalar
    s = jnp.maximum(s, 1e-6)

    # P0_pol_scale = 5.0
    P0_pol_scale = 0.1  # 0.5?
    P0_pos_scale = sigma_obs

    tracks = tracks / s
    sigma_obs = sigma_obs / s

    P0_pol_scale = P0_pol_scale / s
    P0_pos_scale = P0_pos_scale / s

    map_vals = map_warmstart(
        jax.random.PRNGKey(0),
        tracks,
        pad_mask,
        labels,
        onset_idx,
        K,
        sigma_obs,
        dt,
        scale=s,
        P0_pol_scale=P0_pol_scale,
        P0_pos_scale=P0_pos_scale,
        ramp_width=ramp_width,
        lambda_pre_mult=lambda_pre_mult,
        pre_coupling=pre_coupling,
        prior_log_rate_loc=prior_log_rate_loc,
        prior_log_rate_scale=prior_log_rate_scale,
        prior_b_loc=prior_b_loc,
        prior_b_scale=prior_b_scale,
        prior_log_sd1_loc=prior_log_sd1_loc,
        prior_log_sd1_scale=prior_log_sd1_scale,
    )

    if kernel == "nuts":
        mcmc_kernel = NUTS(ou_pol_model, init_strategy=init_to_value(values=map_vals))
    elif kernel == "hmc":
        mcmc_kernel = HMC(ou_pol_model, init_strategy=init_to_value(values=map_vals))
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    # nuts_kernel = NUTS(
    #     ou_pol_model,
    #     init_strategy=init_to_median(num_samples=100),
    # )

    mcmc = MCMC(
        mcmc_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=1,
        progress_bar=True,
    )

    mcmc.run(
        jax.random.PRNGKey(rng_seed),
        tracks,
        pad_mask,
        labels,
        onset_idx,
        K,
        sigma_obs,
        dt,
        scale=s,
        P0_pol_scale=P0_pol_scale,
        P0_pos_scale=P0_pos_scale,
        ramp_width=ramp_width,
        lambda_pre_mult=lambda_pre_mult,
        pre_coupling=pre_coupling,
        prior_log_rate_loc=prior_log_rate_loc,
        prior_log_rate_scale=prior_log_rate_scale,
        prior_b_loc=prior_b_loc,
        prior_b_scale=prior_b_scale,
        prior_log_sd1_loc=prior_log_sd1_loc,
        prior_log_sd1_scale=prior_log_sd1_scale,
    )

    mcmc.print_summary()

    samples = mcmc.get_samples()

    # if label is None:
    #     true_params = None
    # else:
    #     true_params = {
    #         "gamma0": true_gamma1,
    #         "gamma1": true_gamma2,
    #         "sigma_p": true_sigma_p,
    #         "sigma_r": true_sigma_r,
    #     }

    samples_physical = descale_samples(samples, s)

    # plot_class_parameter_posteriors(
    #     samples_physical,
    #     label=label if labels is not None else None,
    #     true_params=true_params,
    #     out_dir="plots",
    # )

    # plot_hyper_parameter_posteriors(
    #     samples,
    #     label=label,
    #     out_dir="plots",
    # )

    # with h5py.File("mcmc_samples.h5", "w") as hf:
    #     for param_name, param_array in samples.items():
    #         # ensure it’s a NumPy array (JAX DeviceArray → np.ndarray)
    #         data = np.array(param_array)
    #         hf.create_dataset(
    #             name=param_name,
    #             data=data,
    #             compression="gzip",  # reduce file size
    #             chunks=True,  # enable chunking for large arrays
    #         )

    with h5py.File(out_name, "w") as hf:
        hf.attrs["posterior_schema"] = "samples_named_physical_v1"
        hf.attrs["method"] = "kalman_mcmc"
        hf.attrs["kernel"] = kernel
        hf.attrs["scale_s"] = float(s)
        hf.attrs["dt"] = float(dt)
        hf.attrs["sigma_obs"] = float(sigma_obs * s)
        hf.attrs["sigma_obs_scaled"] = float(sigma_obs)
        hf.attrs["ramp_width"] = float(ramp_width)
        hf.attrs["lambda_pre_mult"] = float(lambda_pre_mult)
        hf.attrs["pre_coupling"] = float(pre_coupling)
        hf.attrs["class_id"] = -1 if label is None else int(label)
        hf.attrs["track_idx"] = -1 if track_idx is None else int(track_idx)
        hf.attrs["mask_on"] = bool(mask_on)
        hf.attrs["real_data_defaults"] = bool(real_data_defaults)
        hf.attrs["h5_group"] = "" if h5_group is None else str(h5_group)
        hf.attrs["channel"] = str(channel)
        if data_paths is not None:
            hf.attrs["data_files"] = "|".join(map(str, data_paths))
        hf.attrs["prior_log_rate_loc"] = float(prior_log_rate_loc)
        hf.attrs["prior_log_rate_scale"] = float(prior_log_rate_scale)
        hf.attrs["prior_b_loc"] = float(prior_b_loc)
        hf.attrs["prior_b_scale"] = float(prior_b_scale)
        hf.attrs["prior_log_sd1_loc"] = float(prior_log_sd1_loc)
        hf.attrs["prior_log_sd1_scale"] = float(prior_log_sd1_scale)
        hf.attrs["sd0_fixed"] = 1.0
        hf.attrs["diffusion_shared_pre_post"] = True
        hf.attrs["pre_drift_model"] = "derived_decay_zero_coupling"

        samples_group = hf.create_group("samples")
        for param_name, param_array in samples_physical.items():
            samples_group.create_dataset(
                name=param_name,
                data=np.asarray(param_array),
                compression="gzip",
                chunks=True,
            )

        if true_params is not None:
            truths_group = hf.create_group("truth")
            for param_name, param_array in true_params.items():
                arr = np.asarray(param_array)
                if arr.ndim == 0:
                    truths_group.create_dataset(name=param_name, data=arr)
                else:
                    truths_group.create_dataset(
                        name=param_name,
                        data=arr,
                        compression="gzip",
                    )

    return mcmc


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_h5", "--path", dest="path", type=str, default="coupled_ou_tracks_onset.h5")
    parser.add_argument(
        "--data",
        nargs="+",
        default=None,
        help=(
            "One or more exported real-data HDF5 files. This is a high-level alias for real-data mode: "
            "by default it reads onset_motifs/all, keeps nk tracks, uses the HDF5 masks, "
            "and assigns one class per input file. With one file, all retained tracks are class 0."
        ),
    )
    parser.add_argument(
        "--real_data",
        action="store_true",
        default=False,
        help="Use exported-real-data defaults even when passing --data_h5/--path rather than --data.",
    )
    parser.add_argument(
        "--h5_group",
        type=str,
        default=None,
        help="HDF5 group containing tracks/mask/onset_idx. For exported real data, use e.g. onset_motifs/all, onset_motifs/nk, onset_motifs/tumour, nk, or tumour.",
    )
    parser.add_argument(
        "--channel",
        choices=["all", "nk", "tumour"],
        default="all",
        help="Optional channel filter when the selected group contains a channel dataset.",
    )
    parser.add_argument(
        "--split_channel_labels",
        action="store_true",
        default=False,
        help="If the selected real-data group contains mixed channels, assign channel labels rather than treating all retained tracks as one class.",
    )
    parser.add_argument("--dt", type=float, default=1 / 3)
    parser.add_argument("--sigma_obs", type=float, default=2.0)
    parser.add_argument("--num_warmup", type=int, default=3000)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--rng_seed", type=int, default=0)
    parser.add_argument("--class_id", "--label", dest="label", type=int, default=None, help="Class/file label to fit. Use -1 or omit to fit all labels jointly.")
    parser.add_argument("--track_idx", type=int, default=None)
    parser.add_argument("--mask_on", action="store_true", default=True)
    parser.add_argument("--mask_off", action="store_false", dest="mask_on")
    parser.add_argument("--out_h5", "--out_name", dest="out_name", type=str, default="mcmc_samples_masked.h5")
    parser.add_argument("--ramp_width", type=float, default=2.0)
    parser.add_argument("--lambda_pre_mult", type=float, default=4.0)
    parser.add_argument("--pre_coupling", type=float, default=0.0)
    parser.add_argument("--kernel", type=str, default="nuts", choices=["nuts", "hmc"])
    parser.add_argument("--prior_log_rate_loc", type=float, default=-0.5)
    parser.add_argument("--prior_log_rate_scale", type=float, default=0.75)
    parser.add_argument("--prior_b_loc", type=float, default=0.0)
    parser.add_argument("--prior_b_scale", type=float, default=0.75)
    parser.add_argument("--prior_log_sd1_loc", type=float, default=0.0)
    parser.add_argument("--prior_log_sd1_scale", type=float, default=0.4)

    args = parser.parse_args()

    real_data_defaults = bool(args.real_data or args.data is not None)
    if real_data_defaults:
        if args.h5_group is None:
            args.h5_group = "onset_motifs/all"
        if args.channel == "all":
            args.channel = "nk"

    label = args.label
    if label is not None and label < 0:
        label = None

    if args.data is not None:
        print("Real-data mode via --data")
        print("Data files:", args.data)
        print("Defaults: h5_group={!r}, channel={!r}, classes=file indices".format(args.h5_group, args.channel))
    elif real_data_defaults:
        print("Real-data mode via --real_data")
        print("Defaults: h5_group={!r}, channel={!r}".format(args.h5_group, args.channel))

    fit_ou_pol(
        path=args.path,
        dt=args.dt,
        sigma_obs=args.sigma_obs,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        rng_seed=args.rng_seed,
        label=label,
        track_idx=args.track_idx,
        mask_on=args.mask_on,
        out_name=args.out_name,
        ramp_width=args.ramp_width,
        lambda_pre_mult=args.lambda_pre_mult,
        pre_coupling=args.pre_coupling,
        kernel=args.kernel,
        prior_log_rate_loc=args.prior_log_rate_loc,
        prior_log_rate_scale=args.prior_log_rate_scale,
        prior_b_loc=args.prior_b_loc,
        prior_b_scale=args.prior_b_scale,
        prior_log_sd1_loc=args.prior_log_sd1_loc,
        prior_log_sd1_scale=args.prior_log_sd1_scale,
        h5_group=args.h5_group,
        channel=args.channel,
        split_channel_labels=args.split_channel_labels,
        data_paths=args.data,
        real_data_defaults=real_data_defaults,
    )


if __name__ == "__main__":
    main()
