from . import config as _config

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from kaljax.model import ParametricSSM
from kaljax.smooth import backward_sample, rts_batched
from tqdm import tqdm

from .kalman_analytical import (
    _build_matrices_pol,
    make_shock_driver,
    ou_pol_F,
    ou_pol_G,
    ou_pol_Q,
)


def test_kalman_forward_pol_scan(
    h5_path, dt, track_class=0, track_id=0, sigma_obs=0.01, shock=False
):
    """
    Plot observed positions vs. Kalman posterior mean ±1 s.d.
    for one trajectory, using *true* parameters.
    """

    with h5py.File(h5_path, "r") as f:
        tracks = f["tracks"][:]
        tracks_true = f["tracks_true"][:]
        polarity = f["polarity"][:]
        pad_mask = f["mask"][:]
        labels = f["labels"][:]

        a_set = f["gamma1_set"][:]
        b_set = f["gamma2_set"][:]
        sd0_set = f["sigma_p_set"][:]
        sd1_set = f["sigma_r_set"][:]

    y_meas_full = tracks[labels == track_class][track_id]
    y_true_full = tracks_true[labels == track_class][track_id]
    p_meas_full = polarity[labels == track_class][track_id]
    pad_mask = pad_mask[labels == track_class][track_id].astype(bool)
    clazz = labels[labels == track_class][track_id]

    valid = ~np.asarray(pad_mask)
    y_meas = y_meas_full[valid]
    y_true = y_true_full[valid]
    p_meas = p_meas_full[valid]

    ll = []

    print("True", a_set[clazz])

    for c in np.linspace(0.001, 5.0, 20):
        a_param = c * a_set[clazz]
        b_param = b_set[clazz]
        sd0 = sd0_set[clazz]
        sd1 = sd1_set[clazz]

        # print(a_param, b_param)

        # Pull out just one F and Q
        F = ou_pol_F(jnp.array([a_param]), jnp.array([b_param]), dt)[0]
        Q = ou_pol_Q(
            jnp.array([a_param]),
            jnp.array([b_param]),
            jnp.array([sd0]),
            jnp.array([sd1]),
            dt,
        )[0]

        dtype = y_meas.dtype
        H = jnp.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=dtype)
        R = (jnp.array(sigma_obs, dtype=dtype) ** 2) * jnp.eye(2, dtype=dtype)
        m0 = jnp.zeros(4, dtype=dtype)
        P0 = jnp.eye(4, dtype=dtype) * jnp.array(100.0, dtype=dtype)

        theta = jnp.array([a_param, b_param, sd0, sd1, sigma_obs], dtype=dtype)
        ssm = ParametricSSM(
            theta=theta, build_matrices=lambda th: _build_matrices_pol(th, dt)
        )

        # By the time the data gets here, it should already be masked, so we don't need it on ssm, etc

        if shock:
            G = ou_pol_G(dt)
            u = make_shock_driver(
                num_cells=1, T=y_meas.shape[0], dt=dt, period=30, jump=10
            ).squeeze()
            res = ssm(y_meas, m0, P0, G=G, u=u)
        else:
            res = ssm(y_meas, m0, P0)

        m_filt, P_filt, ll_seq = res.m, res.P, res.ll

        ll.append(-ll_seq.sum())

    # print(c, -ll_seq.sum())
    # print('NLL', -ll_seq.sum())

    plt.plot(np.linspace(0.001, 3.0, 20) * a_set[clazz], ll)
    plt.ylabel("NLL")
    plt.xlabel("pol")
    plt.savefig("ll.png")
    plt.clf()

def test_kalman_forward(
    h5_path, dt, track_class=0, track_id=0, sigma_obs=0.01, shock=False
):
    """
    Plot observed positions vs. Kalman posterior mean ±1 s.d.
    for one trajectory, using *true* parameters.
    """

    with h5py.File(h5_path, "r") as f:
        tracks = f["tracks"][:]
        tracks_true = f["tracks_true"][:]
        polarity = f["polarity"][:]
        pad_mask = f["mask"][:]
        labels = f["labels"][:]

        a_set = f["gamma1_set"][:]
        b_set = f["gamma2_set"][:]
        sd0_set = f["sigma_p_set"][:]
        sd1_set = f["sigma_r_set"][:]

    y_meas_full = tracks[labels == track_class][track_id]
    y_true_full = tracks_true[labels == track_class][track_id]
    p_meas_full = polarity[labels == track_class][track_id]
    pad_mask = pad_mask[labels == track_class][track_id].astype(bool)
    clazz = labels[labels == track_class][track_id]

    valid = ~np.asarray(pad_mask)
    y_meas = y_meas_full[valid]
    y_true = y_true_full[valid]
    p_meas = p_meas_full[valid]

    # y_meas = tracks[track_id]#[::10]
    # y_true = tracks_true[track_id]#[::10]
    # p_meas = polarity[track_id]#[::10]
    # m_meas = mask[track_id].astype(bool)#[::10]
    # clazz  = labels[track_id]

    a_param = a_set[clazz]
    b_param = b_set[clazz]
    sd0 = sd0_set[clazz]
    sd1 = sd1_set[clazz]

    # print(a_param, b_param)

    # Pull out just one F and Q
    F = ou_pol_F(jnp.array([a_param]), jnp.array([b_param]), dt)[0]
    Q = ou_pol_Q(
        jnp.array([a_param]),
        jnp.array([b_param]),
        jnp.array([sd0]),
        jnp.array([sd1]),
        dt,
    )[0]

    dtype = y_meas.dtype
    H = jnp.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=dtype)
    R = (jnp.array(sigma_obs, dtype=dtype) ** 2) * jnp.eye(2, dtype=dtype)
    m0 = jnp.array([3.0, 3.0, 0.0, 0.0], dtype=dtype)
    P0 = jnp.eye(4, dtype=dtype) * jnp.array(100.0, dtype=dtype)

    theta = jnp.array([a_param, b_param, sd0, sd1, sigma_obs], dtype=dtype)
    ssm = ParametricSSM(
        theta=theta, build_matrices=lambda th: _build_matrices_pol(th, dt)
    )

    if shock:
        G = ou_pol_G(dt)
        u = make_shock_driver(
            num_cells=1, T=y_meas.shape[0], dt=dt, period=100, jump=5
        ).squeeze()
        res = ssm(y_meas, m0, P0, G=G, u=u)
    else:
        res = ssm(y_meas, m0, P0)

    m_filt, P_filt, ll_seq = res.m, res.P, res.ll

    F, H, Q, R = ssm.build_matrices(theta)
    m_filt_b = m_filt[None, ...]
    P_filt_b = P_filt[None, ...]

    if shock:
        m_smooth_b, P_smooth_b = rts_batched(
            m_filt_b,
            P_filt_b,
            F,
            Q,
            G=G,
            u=u[None, ...],
        )
    else:
        m_smooth_b, P_smooth_b = rts_batched(
            m_filt_b,
            P_filt_b,
            F,
            Q,
        )

    m_filt = m_smooth_b[0]
    P_filt = P_smooth_b[0]

    sd_filt = jnp.sqrt(jnp.stack((P_filt[:, 2, 2], P_filt[:, 3, 3]), axis=-1))

    sd_pol_x = jnp.sqrt(P_filt[:, 0, 0])
    sd_pol_y = jnp.sqrt(P_filt[:, 1, 1])
    sd_x = jnp.sqrt(P_filt[:, 2, 2])
    sd_y = jnp.sqrt(P_filt[:, 3, 3])
    sd_all = jnp.stack((sd_pol_x, sd_pol_y, sd_x, sd_y), axis=1)

    t_axis = jnp.arange(y_meas.shape[0]) * dt

    fig, ax = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    coord_lbl = ["polarity x", "polarity y", "x‑coordinate", "y‑coordinate"]

    for d in range(4):
        if d < 2:
            # latent
            ax[d].plot(t_axis, p_meas[:, d], "-", label="True polarity")
            ax[d].plot(t_axis, m_filt[:, d], "-", label="Kalman mean")
        else:
            # observed
            ax[d].plot(t_axis, y_true[:, d - 2], "-", label="True position")
            ax[d].plot(t_axis, y_meas[:, d - 2], "o", label="observed")
            ax[d].plot(t_axis, m_filt[:, d], "-", label="Kalman mean")
        ax[d].fill_between(
            t_axis,
            m_filt[:, d] - sd_all[:, d],
            m_filt[:, d] + sd_all[:, d],
            alpha=0.2,
            label="±1 s.d.",
        )
        ax[d].set_ylabel(coord_lbl[d])
        ax[d].legend(loc="best")

    ax[-1].set_xlabel("time")

    fig.suptitle(
        f"$\gamma_0 = ${a_param:.2f}, $\gamma_1 = ${b_param:.2f}, $\sigma_0 = ${sd0:.2f}, $\sigma_1 = ${sd1:.2f}"
    )
    plt.tight_layout()
    plt.savefig("kalman_full_kj.png")
    plt.clf()

    plt.plot(m_filt[:, 2], m_filt[:, 3], lw=3.0)
    plt.plot(y_true[:, 0], y_true[:, 1], lw=3.0)
    plt.scatter(
        y_meas[:, 0],
        y_meas[:, 1],
        marker="x",
        s=300,
        color="k",
        linewidth=3.0,
        alpha=0.25,
    )
    plt.axis("off")
    plt.savefig("y.pdf", transparent=True)
    plt.clf()

def _sample_posterior_params(posterior_h5_path, clazz=None, n_param_samples=1000, seed=0):
    with h5py.File(posterior_h5_path, "r") as hf:
        group = hf["samples"] if "samples" in hf else hf
        g0 = group["gamma0"][:]
        g1 = group["gamma1"][:]
        s0 = group["sd0"][:]
        s1 = group["sd1"][:]

    if g0.ndim == 2 and clazz is not None:
        g0 = g0[:, clazz]
        g1 = g1[:, clazz]
        s0 = s0[:, clazz]
        s1 = s1[:, clazz]

    n = g0.shape[0]
    k = min(n_param_samples, n)
    idx = np.random.default_rng(seed).choice(n, size=k, replace=False)

    return g0[idx], g1[idx], s0[idx], s1[idx]

def inference_kalman_forward(
    h5_path,
    dt,
    track_id=0,
    class_id=0,
    sigma_obs=0.01,
    shock=False,
    posterior_h5="mcmc_samples.h5",
    n_param_samples=1000,
    seed=0,
    out_prefix="kalman",
):
    with h5py.File(h5_path, "r") as f:
        tracks = f["tracks"][:]
        tracks_true = f["tracks_true"][:]
        polarity = f["polarity"][:]
        pad_mask = f["mask"][:]
        labels = f["labels"][:]

    y_meas_full = tracks[labels == class_id][track_id]
    y_true_full = tracks_true[labels == class_id][track_id]
    p_meas_full = polarity[labels == class_id][track_id]
    pad_mask = pad_mask[labels == class_id][track_id].astype(bool)

    valid = ~np.asarray(pad_mask)
    y_meas = y_meas_full[valid]
    y_true = y_true_full[valid]
    p_meas = p_meas_full[valid]

    g0_s, g1_s, sd0_s, sd1_s = _sample_posterior_params(
        posterior_h5,
        clazz=None,
        n_param_samples=n_param_samples,
        seed=seed,
    )

    dtype = y_meas.dtype
    H = jnp.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=dtype)
    R = (jnp.array(sigma_obs, dtype=dtype) ** 2) * jnp.eye(2, dtype=dtype)
    m0 = jnp.array([5.0, 5.0, 0.0, 0.0], dtype=dtype)
    P0 = jnp.eye(4, dtype=dtype) * jnp.array(100.0, dtype=dtype)

    if shock:
        G = ou_pol_G(dt)
        u = make_shock_driver(
            num_cells=1, T=y_meas.shape[0], dt=dt, period=30, jump=10
        ).squeeze()
    else:
        G = None
        u = None

    T = y_meas.shape[0]
    lat_samples = []  # (S, T, 2) for [p_x, p_y]
    obs_samples = []  # (S, T, 2) for [x, y]

    rng = np.random.default_rng(seed)

    for a_param, b_param, sd0, sd1 in tqdm(zip(g0_s, g1_s, sd0_s, sd1_s)):
        F = ou_pol_F(jnp.array([a_param]), jnp.array([b_param]), dt)[0]
        Q = ou_pol_Q(
            jnp.array([a_param]),
            jnp.array([b_param]),
            jnp.array([sd0]),
            jnp.array([sd1]),
            dt,
        )[0]

        theta = jnp.array([a_param, b_param, sd0, sd1, sigma_obs], dtype=dtype)
        ssm = ParametricSSM(
            theta=theta, build_matrices=lambda th: _build_matrices_pol(th, dt)
        )

        if G is not None:
            res = ssm(y_meas, m0, P0, G=G, u=u)
        else:
            res = ssm(y_meas, m0, P0)

        m_filt, P_filt = res.m, res.P
        F, H, Q, R = ssm.build_matrices(theta)

        key_lat, key_obs = jax.random.split(jax.random.PRNGKey(rng.integers(2**31 - 1)))

        if G is not None:
            x_samp = backward_sample(m_filt, P_filt, F, Q, key_lat, G=G, u=u)
        else:
            x_samp = backward_sample(m_filt, P_filt, F, Q, key_lat)

        lat = np.asarray(x_samp[:, :2])

        eps_y = jax.random.normal(key_obs, (T, 2), dtype=x_samp.dtype)
        L_y = jnp.linalg.cholesky(R + 1e-9 * jnp.eye(2, dtype=x_samp.dtype))
        obs = np.asarray((H @ x_samp.T).T + eps_y @ L_y.T)

        lat_samples.append(lat)
        obs_samples.append(obs)

    lat_samples = np.stack(lat_samples, axis=0)  # (S,T,2)
    obs_samples = np.stack(obs_samples, axis=0)  # (S,T,2)

    lat_med = np.quantile(lat_samples, 0.5, axis=0)
    # lat_lo  = np.quantile(lat_samples, 0.025, axis=0)
    # lat_hi  = np.quantile(lat_samples, 0.975, axis=0)
    lat_lo = np.quantile(lat_samples, 1 - 0.68, axis=0)
    lat_hi = np.quantile(lat_samples, 0.68, axis=0)

    obs_med = np.quantile(obs_samples, 0.5, axis=0)
    # obs_lo  = np.quantile(obs_samples, 0.025, axis=0)
    # obs_hi  = np.quantile(obs_samples, 0.975, axis=0)
    obs_lo = np.quantile(obs_samples, 1 - 0.68, axis=0)
    obs_hi = np.quantile(obs_samples, 0.68, axis=0)

    t_axis = np.arange(T) * dt
    fig, ax = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    labels_ = ["polarity x", "polarity y", "x-coordinate", "y-coordinate"]

    ax[0].plot(t_axis, p_meas[:, 0], "-", label="True polarity")
    ax[0].plot(t_axis, lat_med[:, 0], "-", label="Median")
    ax[0].fill_between(t_axis, lat_lo[:, 0], lat_hi[:, 0], alpha=0.2, label="68% CI")
    ax[0].set_ylabel(labels_[0])
    ax[0].set_xlim(0, 10)
    ax[0].legend(loc="best")

    ax[1].plot(t_axis, p_meas[:, 1], "-", label="True polarity")
    ax[1].plot(t_axis, lat_med[:, 1], "-", label="Median")
    ax[1].fill_between(t_axis, lat_lo[:, 1], lat_hi[:, 1], alpha=0.2, label="68% CI")
    ax[1].set_ylabel(labels_[1])
    ax[1].set_xlim(0, 10)
    ax[1].legend(loc="best")

    ax[2].plot(t_axis, y_true[:, 0], "-", label="True position")
    ax[2].plot(t_axis, y_meas[:, 0], "o", label="Observed")
    ax[2].plot(t_axis, obs_med[:, 0], "-", label="Median")
    ax[2].fill_between(t_axis, obs_lo[:, 0], obs_hi[:, 0], alpha=0.2, label="68% CI")
    ax[2].set_ylabel(labels_[2])
    ax[2].set_xlim(0, 10)
    ax[2].legend(loc="best")

    ax[3].plot(t_axis, y_true[:, 1], "-", label="True position")
    ax[3].plot(t_axis, y_meas[:, 1], "o", label="Observed")
    ax[3].plot(t_axis, obs_med[:, 1], "-", label="Median")
    ax[3].fill_between(t_axis, obs_lo[:, 1], obs_hi[:, 1], alpha=0.2, label="68% CI")
    ax[3].set_ylabel(labels_[3])
    ax[3].set_xlim(0, 10)
    ax[3].legend(loc="best")

    ax[-1].set_xlabel("time")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_full.png")
    plt.clf()

    fig, ax = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    labels_ = ["polarity x", "polarity y", "x-coordinate", "y-coordinate"]

    ax[0].plot(t_axis, lat_med[:, 0], "-", label="Estimate", lw=5.0)
    ax[0].plot(t_axis, p_meas[:, 0], "-", label="True trajectory", lw=5.0)
    ax[0].scatter(
        [-100], [-100], marker="x", s=300, color="k", linewidth=3.0, label="Observed"
    )
    ax[0].fill_between(
        t_axis, lat_lo[:, 0], lat_hi[:, 0], alpha=0.2
    )  # , label="68% CI")
    ax[0].set_ylabel(r"$\theta_x$", fontsize=21)
    ax[0].tick_params(axis="both", which="major", labelsize=18)
    ax[0].set_xlim(0, 10)
    ax[0].set_ylim(-2, 4)
    ax[0].legend(loc="best", fontsize=21)

    ax[1].plot(t_axis, obs_med[:, 0], "-", label="Median", lw=5.0)
    ax[1].plot(t_axis, y_true[:, 0], "-", label="True position", lw=5.0)
    ax[1].scatter(
        t_axis,
        y_meas[:, 0],
        marker="x",
        s=300,
        color="k",
        linewidth=3.0,
        label="Observed",
    )
    ax[1].fill_between(
        t_axis, obs_lo[:, 0], obs_hi[:, 0], alpha=0.2
    )  # , label="68% CI")
    ax[1].set_ylabel(r"$r_x$", fontsize=21)
    ax[1].tick_params(axis="both", which="major", labelsize=18)
    ax[1].set_xlim(0, 10)
    # ax[2].legend(loc="best")

    ax[-1].set_xlabel("t", fontsize=21)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_poster.png")
    plt.savefig(f"{out_prefix}_poster.pdf")
    plt.clf()

    plt.plot(obs_med[:, 0], obs_med[:, 1])
    plt.plot(y_meas[:, 0], y_meas[:, 1], ".", markersize=10)
    plt.savefig(f"{out_prefix}_y.png")
    plt.clf()
