import json
import math
from typing import Dict, Tuple

import corner
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from models import TrajectoryTransformer
from ou_sim import simulate_coupled_pr_switching


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def valid_onset_theta(theta: torch.Tensor, hp: dict) -> torch.Tensor:
    theta = torch.as_tensor(theta)
    gamma0 = theta[:, 0]
    gamma1 = theta[:, 1]
    log_sd1 = theta[:, 2]

    t = lambda k, default: torch.as_tensor(
        hp.get(k, default), device=theta.device, dtype=torch.float32
    )

    min_rate = t("min_rate", 1e-4)
    max_rate = t("max_rate", 20.0)
    min_coupling = t("min_coupling", 1e-6)
    max_coupling = t("max_coupling", 20.0)
    min_log_sd1 = t("min_log_sd1", math.log(1e-3))
    max_log_sd1 = t("max_log_sd1", math.log(50.0))

    rate = -gamma0

    return (
        torch.isfinite(theta).all(dim=1)
        & (rate > min_rate)
        & (rate < max_rate)
        & (gamma1 > min_coupling)
        & (gamma1 < max_coupling)
        & (log_sd1 > min_log_sd1)
        & (log_sd1 < max_log_sd1)
    )


def sample_gaussian_prior(n: int, hp: dict, device) -> torch.Tensor:
    out = []
    remaining = int(n)

    while remaining > 0:
        m = max(remaining * 2, 1024)
        t = lambda k: torch.as_tensor(hp[k], device=device, dtype=torch.float32)

        gamma0 = -torch.distributions.LogNormal(
            t("log_rate_loc"), t("log_rate_scale")
        ).sample((m,))

        gamma1 = torch.distributions.LogNormal(t("b_loc"), t("b_scale")).sample((m,))
        log_sd1 = torch.distributions.Normal(t("sd1_loc"), t("sd1_scale")).sample((m,))

        theta = torch.stack([gamma0, gamma1, log_sd1], dim=1)
        theta = theta[valid_onset_theta(theta, hp)]

        if theta.shape[0] == 0:
            continue

        take = min(remaining, theta.shape[0])
        out.append(theta[:take])
        remaining -= take

    return torch.cat(out, dim=0)


def log_prior_gaussian(theta: torch.Tensor, hp: dict) -> torch.Tensor:
    gamma0, gamma1, log_sd1 = theta[:, 0], theta[:, 1], theta[:, 2]
    two_pi = math.log(2.0 * math.pi)
    t = lambda k: torch.as_tensor(hp[k], device=theta.device, dtype=torch.float32)

    def lognorm_logpdf(x, mu, sd):
        return torch.where(
            x > 0,
            -((torch.log(x) - mu) ** 2) / (2 * sd * sd)
            - torch.log(x)
            - torch.log(sd)
            - 0.5 * two_pi,
            torch.full_like(x, -float("inf")),
        )

    lp_g0 = torch.where(
        gamma0 < 0,
        lognorm_logpdf(-gamma0, t("log_rate_loc"), t("log_rate_scale")),
        torch.full_like(gamma0, -float("inf")),
    )

    lp_g1 = lognorm_logpdf(gamma1, t("b_loc"), t("b_scale"))

    log_sd1 = theta[:, 2]
    lp_sd1 = (
        -0.5 * ((log_sd1 - t("sd1_loc")) / t("sd1_scale")) ** 2
        - torch.log(t("sd1_scale"))
        - 0.5 * two_pi
    )

    lp = lp_g0 + lp_g1 + lp_sd1
    valid = valid_onset_theta(theta, hp)
    return torch.where(valid, lp, torch.full_like(lp, -float("inf")))


def apply_standard_scaler(
    tracks: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    return (tracks - mean.view(1, 1, -1)) / (std.view(1, 1, -1) + eps)


def build_model_input(
    tracks: torch.Tensor,
    onset_idx: torch.Tensor | None,
    mean: torch.Tensor,
    std: torch.Tensor,
    input_dim: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError(
            f"Expected raw tracks with shape [B, T, 2], got {tuple(tracks.shape)}"
        )

    B, T, _ = tracks.shape
    if input_dim == 2:
        if mean.numel() == 2 and std.numel() == 2:
            return apply_standard_scaler(tracks, mean, std, eps=eps)
        if mean.numel() == 3 and std.numel() == 3:
            return apply_standard_scaler(tracks, mean[:2], std[:2], eps=eps)
        raise ValueError(
            f"Unsupported scaler shape for input_dim=2: mean={tuple(mean.shape)}, std={tuple(std.shape)}"
        )

    if input_dim != 3:
        raise ValueError(f"Unsupported model input_dim={input_dim}; expected 2 or 3.")

    if onset_idx is None:
        raise ValueError("Model expects input_dim=3, but onset_idx was not provided.")

    onset_idx = torch.as_tensor(onset_idx, device=tracks.device, dtype=torch.long).view(
        B
    )
    time_idx = torch.arange(T, device=tracks.device).view(1, T)
    is_post = (time_idx >= onset_idx[:, None]).to(tracks.dtype).unsqueeze(-1)

    if mean.numel() == 3 and std.numel() == 3:
        x = torch.cat([tracks, is_post], dim=-1)
        return apply_standard_scaler(x, mean, std, eps=eps)

    if mean.numel() == 2 and std.numel() == 2:
        tracks_scaled = apply_standard_scaler(tracks, mean, std, eps=eps)
        return torch.cat([tracks_scaled, is_post], dim=-1)

    raise ValueError(
        f"Unsupported scaler shape for input_dim=3: mean={tuple(mean.shape)}, std={tuple(std.shape)}"
    )


def _infer_num_layers(state: dict) -> int:
    layers = set()
    prefix = "encoder.layers."
    for k in state.keys():
        if k.startswith(prefix):
            rest = k[len(prefix) :]
            idx = rest.split(".", 1)[0]
            if idx.isdigit():
                layers.add(int(idx))
    if not layers:
        raise KeyError("Could not infer num_layers from checkpoint state_dict.")
    return max(layers) + 1


def load_embedding_model(
    ckpt_path: str, *, t_max: int, device: str | torch.device = "cpu"
):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"]

    if "coord_proj.weight" not in state:
        raise KeyError("Checkpoint is missing coord_proj.weight.")
    input_dim = int(state["coord_proj.weight"].shape[1])
    d_model = int(state["coord_proj.weight"].shape[0])
    embed_dim = int(state["proj.3.weight"].shape[0])
    num_classes = int(state["classifier.weight"].shape[0])
    dim_feedforward = int(state["encoder.layers.0.linear1.weight"].shape[0])
    num_layers = int(ckpt.get("num_layers", _infer_num_layers(state)))
    nhead = int(ckpt.get("n_head", ckpt.get("nhead", 2)))
    dropout = float(ckpt.get("dropout", 0.1))

    model = TrajectoryTransformer(
        t_max=t_max,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        embed_dim=embed_dim,
        num_classes=num_classes,
        input_dim=input_dim,
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    reg_head = None
    if "reg_head_state_dict" in ckpt:
        out_dim = int(ckpt["reg_head_state_dict"]["weight"].shape[0])
        reg_head = nn.Linear(embed_dim, out_dim).to(device)
        reg_head.load_state_dict(ckpt["reg_head_state_dict"], strict=True)
        reg_head.eval()

    mean = torch.as_tensor(ckpt["scaler_mean"], device=device, dtype=torch.float32)
    std = torch.as_tensor(ckpt["scaler_std"], device=device, dtype=torch.float32)
    theta_mean = torch.as_tensor(
        ckpt.get("theta_mean", torch.zeros(1, 3)), device=device, dtype=torch.float32
    )
    theta_std = torch.as_tensor(
        ckpt.get("theta_std", torch.ones(1, 3)), device=device, dtype=torch.float32
    )
    return model, reg_head, mean, std, theta_mean, theta_std


@torch.no_grad()
def embed_tracks(
    model,
    tracks: torch.Tensor,
    mask: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    onset_idx: torch.Tensor | None = None,
    chunk_size: int = 512,
) -> torch.Tensor:
    outs = []
    n = tracks.shape[0]
    input_dim = int(getattr(model, "input_dim", tracks.shape[-1]))
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        onset_chunk = None if onset_idx is None else onset_idx[s:e]
        x = build_model_input(tracks[s:e], onset_chunk, mean, std, input_dim)
        z, _ = model(x, mask=mask[s:e])
        outs.append(F.normalize(z, dim=1))
    return torch.cat(outs, dim=0)


def simulate_coupled_pr_for_theta_fixed_onset(
    theta,
    onset_idx: torch.Tensor,
    t_max: int,
    dt: float,
    *,
    reps: int = 1,
    p0: torch.Tensor | None = None,
    measurement_sd: float = 2.0,
    spikes: bool = False,
    device: str | torch.device = "cpu",
    ramp_width: float = 2.0,
    lambda_pre_mult: float = 4.0,
    sigma_p_fixed: float = 1.0,
    pre_coupling: float = 1e-3,
) -> torch.Tensor:
    theta = torch.as_tensor(theta, device=device, dtype=torch.float32).view(-1, 3)
    onset_idx = torch.as_tensor(onset_idx, device=device, dtype=torch.long).view(-1)

    if theta.shape[0] != onset_idx.shape[0]:
        raise ValueError("theta and onset_idx must have the same batch size.")

    b = theta.shape[0]
    n = b * reps

    gamma0_post = theta[:, 0].repeat_interleave(reps)
    gamma1_post = theta[:, 1].repeat_interleave(reps)
    sd1_post = torch.exp(theta[:, 2]).repeat_interleave(reps)

    gamma0_pre = lambda_pre_mult * gamma0_post
    gamma1_pre = torch.full_like(gamma1_post, float(pre_coupling))

    gamma0_per_state = torch.stack([gamma0_pre, gamma0_post], dim=1)
    gamma1_per_state = torch.stack([gamma1_pre, gamma1_post], dim=1)

    sigma_p = torch.full((n,), float(sigma_p_fixed), device=device, dtype=torch.float32)
    sigma_p_per_state = torch.stack([sigma_p, sigma_p], dim=1)
    sigma_r_per_state = torch.stack([sd1_post, sd1_post], dim=1)

    onset_rep = onset_idx.repeat_interleave(reps).view(n, 1)

    if p0 is None:
        p0 = torch.zeros((n, 2), device=device, dtype=torch.float32)
    else:
        p0 = torch.as_tensor(p0, device=device, dtype=torch.float32)
        if p0.ndim == 1:
            p0 = p0.view(1, 2).repeat(n, 1)
        elif p0.shape[0] != n:
            k = math.ceil(n / p0.shape[0])
            p0 = p0.repeat(k, 1)[:n]

    _, tracks, _ = simulate_coupled_pr_switching(
        n_cells=n,
        t_max=t_max,
        dt=dt,
        p0=p0,
        gamma1_per_state=gamma0_per_state,
        gamma2_per_state=gamma1_per_state,
        sigma_p_per_state=sigma_p_per_state,
        sigma_r_per_state=sigma_r_per_state,
        switch_times=onset_rep,
        measurement_sd=measurement_sd,
        spikes=spikes,
        device=device,
        ramp_width=ramp_width,
    )

    return tracks


@torch.no_grad()
def summarize_embeddings(
    z: torch.Tensor,
    reg_head,
    theta_mean: torch.Tensor,
    theta_std: torch.Tensor,
    summary_source: str,
) -> torch.Tensor:
    if summary_source == "latent":
        return z
    if reg_head is None:
        raise ValueError(
            "Checkpoint has no reg_head_state_dict, so summary_source='theta' is unavailable."
        )
    pred = reg_head(z)
    return pred


@torch.no_grad()
def observed_summaries(
    model,
    reg_head,
    tracks: torch.Tensor,
    mask: torch.Tensor,
    onset_idx: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    theta_mean: torch.Tensor,
    theta_std: torch.Tensor,
    *,
    summary_source: str,
    embed_chunk_size: int,
) -> torch.Tensor:
    z = embed_tracks(
        model, tracks, mask, mean, std, onset_idx=onset_idx, chunk_size=embed_chunk_size
    )
    return summarize_embeddings(z, reg_head, theta_mean, theta_std, summary_source)


@torch.no_grad()
def simulated_summaries_for_theta_batch(
    *,
    model,
    reg_head,
    thetas: torch.Tensor,
    obs_mask: torch.Tensor,
    obs_onset: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    theta_mean: torch.Tensor,
    theta_std: torch.Tensor,
    t_max: int,
    dt: float,
    reps: int,
    measurement_sd: float,
    spikes: bool,
    device: str | torch.device,
    ramp_width: float,
    lambda_pre_mult: float,
    sigma_p_fixed: float,
    pre_coupling: float,
    summary_source: str,
    embed_chunk_size: int,
    prior: dict | None = None,
) -> torch.Tensor:
    b = thetas.shape[0]
    n_obs = obs_mask.shape[0]
    if prior is not None:
        valid = valid_onset_theta(thetas, prior)
    else:
        valid = torch.isfinite(thetas).all(dim=1)

    if summary_source == "theta":
        summary_dim = int(theta_mean.numel())
    else:
        summary_dim = int(model.classifier.in_features)

    if not valid.all():
        out = torch.full((b, n_obs, summary_dim), float("inf"), device=device)
        if valid.any():
            valid_idx = torch.where(valid)[0]
            sub = simulated_summaries_for_theta_batch(
                model=model,
                reg_head=reg_head,
                thetas=thetas[valid],
                obs_mask=obs_mask,
                obs_onset=obs_onset,
                mean=mean,
                std=std,
                theta_mean=theta_mean,
                theta_std=theta_std,
                t_max=t_max,
                dt=dt,
                reps=reps,
                measurement_sd=measurement_sd,
                spikes=spikes,
                device=device,
                ramp_width=ramp_width,
                lambda_pre_mult=lambda_pre_mult,
                sigma_p_fixed=sigma_p_fixed,
                pre_coupling=pre_coupling,
                summary_source=summary_source,
                embed_chunk_size=embed_chunk_size,
                prior=None,
            )
            out[valid_idx] = sub
        return out

    out_chunks = []
    obs_chunk = max(1, embed_chunk_size // max(1, reps * b))

    for s in range(0, n_obs, obs_chunk):
        e = min(s + obs_chunk, n_obs)
        c = e - s
        mask_chunk = obs_mask[s:e].to(device)
        onset_chunk = obs_onset[s:e].to(device)

        theta_exp = thetas[:, None, :].expand(b, c, 3).reshape(b * c, 3)
        onset_exp = onset_chunk[None, :].expand(b, c).reshape(b * c)
        try:
            tracks = simulate_coupled_pr_for_theta_fixed_onset(
                theta_exp,
                onset_exp,
                t_max,
                dt,
                reps=reps,
                measurement_sd=measurement_sd,
                spikes=spikes,
                device=device,
                ramp_width=ramp_width,
                lambda_pre_mult=lambda_pre_mult,
                sigma_p_fixed=sigma_p_fixed,
                pre_coupling=pre_coupling,
            )
        except RuntimeError as exc:
            msg = str(exc).lower()
            if (
                "cholesky" not in msg
                and "positive-definite" not in msg
                and "linalg" not in msg
            ):
                raise
            return torch.full((b, n_obs, summary_dim), float("inf"), device=device)
        mask_rep = mask_chunk[None, :, :].expand(b, c, t_max).reshape(b * c, t_max)
        mask_rep = mask_rep.repeat_interleave(reps, dim=0)
        z = embed_tracks(
            model,
            tracks,
            mask_rep,
            mean,
            std,
            onset_idx=onset_exp.repeat_interleave(reps),
            chunk_size=embed_chunk_size,
        )
        z = z.view(b * c, reps, -1).mean(dim=1)
        s_chunk = summarize_embeddings(
            z, reg_head, theta_mean, theta_std, summary_source
        ).view(b, c, -1)
        out_chunks.append(s_chunk)

    return torch.cat(out_chunks, dim=1)


def aggregate_track_summaries(s: torch.Tensor, mode: str = "none") -> torch.Tensor:
    if mode == "none":
        return s

    if s.ndim != 2:
        raise ValueError("Expected track summaries with shape [n_tracks, n_features].")

    n = s.shape[0]
    root_n = math.sqrt(n)
    mean = s.mean(dim=0)

    if mode == "mean":
        return mean

    if mode == "mean_scaled":
        return mean * root_n

    std = s.std(dim=0, unbiased=False)

    if mode == "mean_std":
        return torch.cat([mean, std], dim=0)

    if mode == "mean_std_scaled":
        return torch.cat([mean * root_n, std], dim=0)

    se = std / root_n

    if mode == "mean_se":
        return torch.cat([mean, se], dim=0)

    if mode == "mean_std_se":
        return torch.cat([mean, std, se], dim=0)

    raise ValueError(f"Unknown class summary mode: {mode}")


def aggregate_simulated_summaries(s: torch.Tensor, mode: str = "none") -> torch.Tensor:
    if mode == "none":
        return s

    if s.ndim != 3:
        raise ValueError("Expected simulated summaries with shape [n_particles, n_tracks, n_features].")

    n = s.shape[1]
    root_n = math.sqrt(n)
    mean = s.mean(dim=1)

    if mode == "mean":
        return mean

    if mode == "mean_scaled":
        return mean * root_n

    std = s.std(dim=1, unbiased=False)

    if mode == "mean_std":
        return torch.cat([mean, std], dim=1)

    if mode == "mean_std_scaled":
        return torch.cat([mean * root_n, std], dim=1)

    se = std / root_n

    if mode == "mean_se":
        return torch.cat([mean, se], dim=1)

    if mode == "mean_std_se":
        return torch.cat([mean, std, se], dim=1)

    raise ValueError(f"Unknown class summary mode: {mode}")


def abc_distance(
    s_sim: torch.Tensor, s_obs: torch.Tensor, mode: str = "rms_scaled"
) -> torch.Tensor:
    if s_sim.ndim == 2 and s_obs.ndim == 1:
        return torch.norm(s_sim - s_obs.unsqueeze(0), dim=1)

    per_track = torch.norm(s_sim - s_obs.unsqueeze(0), dim=-1)

    if mode == "mean":
        return per_track.mean(dim=1)

    if mode == "rms":
        return torch.sqrt(torch.mean(per_track**2, dim=1))

    if mode == "rms_scaled":
        return torch.sqrt(torch.mean(per_track**2, dim=1)) * math.sqrt(
            per_track.shape[1]
        )

    if mode == "sum":
        return per_track.sum(dim=1)

    raise ValueError(f"Unknown ABC distance mode: {mode}")


def make_pd_cov(cov, min_jitter=1e-6, max_tries=12):
    cov = np.asarray(cov, dtype=np.float64)
    cov = 0.5 * (cov + cov.T)
    eye = np.eye(cov.shape[0], dtype=np.float64)
    jitter = float(min_jitter)
    for _ in range(max_tries):
        cov_try = cov + jitter * eye
        try:
            np.linalg.cholesky(cov_try)
            return cov_try
        except np.linalg.LinAlgError:
            jitter *= 10.0
    evals, evecs = np.linalg.eigh(cov)
    floor = max(min_jitter, 1e-8 - evals.min())
    evals = np.clip(evals, floor, None)
    cov_pd = (evecs * evals[None, :]) @ evecs.T
    cov_pd = 0.5 * (cov_pd + cov_pd.T)
    np.linalg.cholesky(cov_pd)
    return cov_pd


def torch_cholesky_from_numpy_cov(cov, device):
    cov_t = torch.as_tensor(cov, device=device, dtype=torch.float32)
    return torch.linalg.cholesky(cov_t)


@torch.no_grad()
def abc_latent_smc(
    *,
    model,
    reg_head,
    mean: torch.Tensor,
    std: torch.Tensor,
    theta_mean: torch.Tensor,
    theta_std: torch.Tensor,
    s_obs: torch.Tensor,
    obs_mask: torch.Tensor,
    obs_onset: torch.Tensor,
    t_max: int,
    dt: float,
    prior: dict,
    n_keep: int = 1000,
    batch_size: int = 256,
    reps: int = 8,
    measurement_sd: float = 2.0,
    spikes: bool = False,
    T: int = 5,
    alpha: float = 0.5,
    pilot_mult: int = 4,
    device: str | torch.device = "cpu",
    ramp_width: float = 2.0,
    lambda_pre_mult: float = 4.0,
    sigma_p_fixed: float = 1.0,
    pre_coupling: float = 1e-3,
    summary_source: str = "latent",
    embed_chunk_size: int = 512,
    distance_mode: str = "rms_scaled",
    class_summary_mode: str = "mean_se",
):
    s_obs_abc = aggregate_track_summaries(s_obs, mode=class_summary_mode)
    pilot_n = pilot_mult * batch_size
    pilot_theta = sample_gaussian_prior(pilot_n, prior, device)
    s_p = simulated_summaries_for_theta_batch(
        model=model,
        reg_head=reg_head,
        thetas=pilot_theta,
        obs_mask=obs_mask,
        obs_onset=obs_onset,
        mean=mean,
        std=std,
        theta_mean=theta_mean,
        theta_std=theta_std,
        t_max=t_max,
        dt=dt,
        reps=reps,
        measurement_sd=measurement_sd,
        spikes=spikes,
        device=device,
        ramp_width=ramp_width,
        lambda_pre_mult=lambda_pre_mult,
        sigma_p_fixed=sigma_p_fixed,
        pre_coupling=pre_coupling,
        summary_source=summary_source,
        embed_chunk_size=embed_chunk_size,
        prior=prior,
    )
    s_p_abc = aggregate_simulated_summaries(s_p, mode=class_summary_mode)
    d_pilot = abc_distance(s_p_abc, s_obs_abc, mode=distance_mode)
    if not torch.isfinite(d_pilot).all():
        finite = torch.isfinite(d_pilot)
        if not finite.any():
            raise RuntimeError(
                "All pilot ABC simulations failed; tighten the prior or increase stability_margin."
            )
        fill = torch.max(d_pilot[finite]) * 10.0
        d_pilot = torch.where(finite, d_pilot, fill)
    eps = float(torch.quantile(d_pilot, alpha).item())

    theta_prev = None
    w_prev = None
    dist_prev = None
    cov = None

    def mvn_logpdf(x, mean, inv_cov, logdet_cov):
        d = x.shape[1]
        dx = x - mean
        q = np.einsum("ni,ij,nj->n", dx, inv_cov, dx)
        return -0.5 * (q + d * np.log(2.0 * np.pi) + logdet_cov)

    for t in range(T):
        kept_theta, kept_dist = [], []
        n_acc = 0
        pbar = tqdm(
            total=n_keep, desc=f"ABC-SMC pop {t + 1}/{T} (eps={eps:.3e})", leave=True
        )

        while n_acc < n_keep:
            if t == 0:
                thetas = sample_gaussian_prior(batch_size, prior, device)
            else:
                idx = np.random.choice(
                    theta_prev.shape[0], size=batch_size, replace=True, p=w_prev
                )
                base = theta_prev[idx]
                noise = torch.randn(batch_size, theta_prev.shape[1], device=device)
                L = torch_cholesky_from_numpy_cov(cov, device)
                thetas = base.to(device) + noise @ L.T
                lp = log_prior_gaussian(thetas, prior)
                m_support = torch.isfinite(lp)
                if not m_support.any():
                    continue
                thetas = thetas[m_support]
                if thetas.shape[0] == 0:
                    continue

            s_sim = simulated_summaries_for_theta_batch(
                model=model,
                reg_head=reg_head,
                thetas=thetas,
                obs_mask=obs_mask,
                obs_onset=obs_onset,
                mean=mean,
                std=std,
                theta_mean=theta_mean,
                theta_std=theta_std,
                t_max=t_max,
                dt=dt,
                reps=reps,
                measurement_sd=measurement_sd,
                spikes=spikes,
                device=device,
                ramp_width=ramp_width,
                lambda_pre_mult=lambda_pre_mult,
                sigma_p_fixed=sigma_p_fixed,
                pre_coupling=pre_coupling,
                summary_source=summary_source,
                embed_chunk_size=embed_chunk_size,
                prior=prior,
            )
            s_sim_abc = aggregate_simulated_summaries(s_sim, mode=class_summary_mode)
            d = abc_distance(s_sim_abc, s_obs_abc, mode=distance_mode)
            m = torch.isfinite(d) & (d <= eps)
            if m.any():
                acc_theta = thetas[m].detach().cpu()
                acc_dist = d[m].detach().cpu()
                kept_theta.append(acc_theta)
                kept_dist.append(acc_dist)
                n_new = acc_theta.shape[0]
                n_acc += n_new
                pbar.update(n_new)

        theta_t = torch.cat(kept_theta, dim=0)[:n_keep]
        dist_t = torch.cat(kept_dist, dim=0)[:n_keep]
        pbar.close()

        theta_np = theta_t.numpy()
        dist_np = dist_t.numpy()

        if t == 0:
            w_t = np.full(n_keep, 1.0 / n_keep, dtype=np.float64)
        else:
            lp = (
                log_prior_gaussian(theta_t, prior)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            inv_cov = np.linalg.inv(cov)
            _, logdet = np.linalg.slogdet(cov)
            log_wprev = np.log(w_prev + 1e-300)
            log_q = np.empty(n_keep, dtype=np.float64)
            for i in range(n_keep):
                x = theta_np[i : i + 1]
                comp = mvn_logpdf(theta_prev, x[0], inv_cov, logdet) + log_wprev
                mmax = comp.max()
                log_q[i] = mmax + np.log(np.exp(comp - mmax).sum())
            log_w = lp - log_q
            mmax = log_w.max()
            w_unn = np.exp(log_w - mmax)
            w_t = w_unn / w_unn.sum()

        mu = (theta_np * w_t[:, None]).sum(axis=0)
        xc = theta_np - mu[None, :]
        cov_raw = 2.0 * ((xc * w_t[:, None]).T @ xc)
        cov = make_pd_cov(cov_raw, min_jitter=1e-6, max_tries=12)

        theta_prev = theta_t.clone()
        w_prev = w_t.copy()
        dist_prev = dist_t.clone()
        eps = float(np.quantile(dist_np, alpha))

    return (
        theta_prev,
        dist_prev,
        torch.as_tensor(w_prev, dtype=torch.float32),
        float(eps),
    )


def save_abc_posterior_h5(
    path,
    theta,
    dist,
    weights=None,
    attrs=None,
    *,
    resample_weighted: bool = True,
    seed: int = 0,
):
    theta_np = (
        theta.detach().cpu().numpy() if torch.is_tensor(theta) else np.asarray(theta)
    )
    dist_np = dist.detach().cpu().numpy() if torch.is_tensor(dist) else np.asarray(dist)

    weights_np = None
    if weights is not None:
        weights_np = (
            weights.detach().cpu().numpy()
            if torch.is_tensor(weights)
            else np.asarray(weights)
        ).astype(np.float64)
        weights_sum = weights_np.sum()
        if not np.isfinite(weights_sum) or weights_sum <= 0:
            raise ValueError("ABC weights are not normalisable.")
        weights_np = weights_np / weights_sum

    theta_smc_raw = theta_np.copy()
    dist_smc_raw = dist_np.copy()
    weights_out = weights_np

    if resample_weighted and weights_np is not None:
        rng = np.random.default_rng(seed)
        idx = rng.choice(
            theta_np.shape[0], size=theta_np.shape[0], replace=True, p=weights_np
        )
        theta_np = theta_np[idx]
        dist_np = dist_np[idx]
        weights_out = np.full(
            theta_np.shape[0], 1.0 / theta_np.shape[0], dtype=np.float64
        )

    gamma0 = theta_np[:, 0]
    gamma1 = theta_np[:, 1]
    sd1 = np.exp(theta_np[:, 2])
    sd0 = np.ones_like(sd1)

    with h5py.File(path, "w") as f:
        f.create_dataset("theta_raw", data=theta_np)
        f.create_dataset("dist", data=dist_np)
        f.create_dataset("theta_smc_raw", data=theta_smc_raw)
        f.create_dataset("dist_smc_raw", data=dist_smc_raw)

        g = f.create_group("samples")
        g.create_dataset("gamma0", data=gamma0)
        g.create_dataset("gamma1", data=gamma1)
        g.create_dataset("sd0", data=sd0)
        g.create_dataset("sd1", data=sd1)

        if weights_out is not None:
            f.create_dataset("weights", data=weights_out)
        if weights_np is not None:
            f.create_dataset("weights_smc_raw", data=weights_np)

        if attrs:
            for k, v in attrs.items():
                f.attrs[k] = v

        f.attrs["posterior_schema"] = "samples_named_physical_v1"
        f.attrs["theta_raw_columns"] = "gamma0,gamma1,log_sd1"
        f.attrs["sd0_fixed"] = 1.0
        f.attrs["diffusion_shared_pre_post"] = True
        f.attrs["pre_drift_model"] = "derived_decay_fixed_coupling"
        f.attrs["resample_weighted"] = bool(resample_weighted)


def _load_dataset(
    data_file: str,
    *,
    onset_field: str | None = None,
    onset_value: int | None = None,
) -> dict:
    """
    Load tracks and any optional ancillary fields from an HDF5 file.

    Required fields:
        tracks  : [N, T, 2]
        mask    : [N, T]   (boolean; True == padded/invalid)

    Optional fields (any subset may be absent for real data):
        labels      : [N]              class labels for class-based selection
        onset_idx   : [N]              per-track onset (long)
        gamma1, gamma2, sigma_r        per-track simulator truth (rare)
        gamma1_set, gamma2_set,        per-class simulator truth
        sigma_r_set

    Parameters
    ----------
    onset_field : str | None
        Name of an alternative HDF5 dataset to use as the onset index. If
        absent and 'onset_idx' is also missing, the value of `onset_value`
        is used (broadcast to all tracks).
    onset_value : int | None
        Fallback constant onset to use for every track when no onset
        information is in the file.
    """
    out: dict = {}
    with h5py.File(data_file, "r") as f:
        if "tracks" not in f:
            raise KeyError(f"'tracks' dataset missing from {data_file}")
        if "mask" not in f:
            raise KeyError(f"'mask' dataset missing from {data_file}")
        out["tracks"] = f["tracks"][:]
        out["mask"] = f["mask"][:]
        N = out["tracks"].shape[0]
        T = out["tracks"].shape[1]

        # Labels — optional
        out["labels"] = f["labels"][:] if "labels" in f else None

        # Onset index — required, but with several fallbacks
        if onset_field is not None and onset_field in f:
            out["onset_idx"] = np.asarray(f[onset_field][:]).astype(np.int64)
        elif "onset_idx" in f:
            out["onset_idx"] = np.asarray(f["onset_idx"][:]).astype(np.int64)
        elif onset_value is not None:
            out["onset_idx"] = np.full((N,), int(onset_value), dtype=np.int64)
        else:
            raise KeyError(
                f"Onset information not found. Provide a dataset named "
                f"'onset_idx' (or pass --onset_field) in {data_file}, or "
                f"specify --onset_value to use a constant onset for all tracks."
            )

        # Per-track truth (rarely present)
        out["gamma1_per_track"] = f["gamma1"][:] if "gamma1" in f else None
        out["gamma2_per_track"] = f["gamma2"][:] if "gamma2" in f else None
        out["sigma_r_per_track"] = f["sigma_r"][:] if "sigma_r" in f else None

        # Per-class truth (more common when labels are present)
        out["gamma1_per_class"] = f["gamma1_set"][:] if "gamma1_set" in f else None
        out["gamma2_per_class"] = f["gamma2_set"][:] if "gamma2_set" in f else None
        out["sigma_r_per_class"] = f["sigma_r_set"][:] if "sigma_r_set" in f else None

    return out


def _select_state(x, state_index=-1):
    x = np.asarray(x)
    if x.ndim >= 2:
        return x[..., state_index]
    return x


def _resolve_truth(
    ds: dict,
    sel_idx: np.ndarray,
    track_id: int | None,
    track_class: int | None,
) -> list | None:
    """
    Build a per-track truth array [gamma0, gamma1, sd1] for the selected
    tracks, or None if the dataset has no truth information at all.

    Returns None silently when truth fields are missing (real data case).
    """
    g0_pt = ds["gamma1_per_track"]
    g1_pt = ds["gamma2_per_track"]
    sd1_pt = ds["sigma_r_per_track"]
    g0_pc = ds["gamma1_per_class"]
    g1_pc = ds["gamma2_per_class"]
    sd1_pc = ds["sigma_r_per_class"]
    labels = ds["labels"]

    have_per_track = g0_pt is not None and g1_pt is not None and sd1_pt is not None
    have_per_class = (
        g0_pc is not None
        and g1_pc is not None
        and sd1_pc is not None
        and labels is not None
    )

    if not (have_per_track or have_per_class):
        return None

    # Collect per-selected-track truth values
    if have_per_track:
        g0 = np.asarray(g0_pt)[sel_idx]
        g1 = np.asarray(g1_pt)[sel_idx]
        sd1 = np.asarray(sd1_pt)[sel_idx]
    else:
        # Per-class truth: index by labels of the selected tracks
        sel_labels = np.asarray(labels)[sel_idx]
        g0 = np.asarray(g0_pc)[sel_labels]
        g1 = np.asarray(g1_pc)[sel_labels]
        sd1 = np.asarray(sd1_pc)[sel_labels]

    g0 = _select_state(g0, state_index=-1)
    g1 = _select_state(g1, state_index=-1)
    sd1 = _select_state(sd1, state_index=-1)

    if track_id is None:
        # Use up to 128 selected tracks (matches original behaviour)
        n_use = min(128, len(sel_idx))
        return [
            np.float32(g0[:n_use]),
            np.float32(g1[:n_use]),
            np.float32(sd1[:n_use]),
        ]
    else:
        return [
            np.float32(g0[track_id]),
            np.float32(g1[track_id]),
            np.float32(sd1[track_id]),
        ]


def main():
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "ABC-SMC inference of OU+coupling parameters from observed "
            "tracks. Works on simulated data (with ground-truth labels "
            "and parameter values) or on real data (no labels, no truth)."
        )
    )
    p.add_argument(
        "--data_file",
        required=True,
        help="HDF5 file containing 'tracks' and 'mask'. "
        "Optionally contains 'labels', 'onset_idx', and "
        "ground-truth parameter fields.",
    )
    p.add_argument("--model_ckpt", required=True)

    # Selection — both optional now
    p.add_argument(
        "--track_class",
        type=int,
        default=None,
        help="If the data file contains a 'labels' dataset, restrict "
        "inference to tracks of this class. Ignored when no labels "
        "are present.",
    )
    p.add_argument(
        "--track_id",
        type=int,
        default=None,
        help="If set, run inference on a single track within the "
        "selected subset. Otherwise use a population of up to "
        "--max_tracks tracks.",
    )
    p.add_argument(
        "--max_tracks",
        type=int,
        default=129,
        help="Maximum number of tracks to use when --track_id is not "
        "given. Population mode averages embeddings across these "
        "tracks to form a single observation summary.",
    )

    p.add_argument(
        "--subset_mode",
        choices=["first", "random"],
        default="first",
        help="How to choose the population subset when --track_id is not given. "
        "'random' uses --subset_seed and takes a nested deterministic subset.",
    )
    p.add_argument(
        "--subset_seed",
        type=int,
        default=0,
        help="Seed used when --subset_mode=random.",
    )

    # Onset handling for real data
    p.add_argument(
        "--onset_field",
        type=str,
        default=None,
        help="Name of an alternative HDF5 dataset to use as the per-track "
        "onset (overrides the default 'onset_idx').",
    )
    p.add_argument(
        "--onset_value",
        type=int,
        default=None,
        help="Constant onset to use for every track when no onset field "
        "is present in the data file. Mutually exclusive with onset "
        "datasets.",
    )

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dt", type=float, default=1 / 3)
    p.add_argument("--measurement_sd", type=float, default=2.0)
    p.add_argument("--spikes", action="store_true")
    p.add_argument("--reps", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--n_keep", type=int, default=1000)
    p.add_argument("--sampler", choices=["smc"], default="smc")
    p.add_argument("--smc_T", type=int, default=5)
    p.add_argument("--smc_alpha", type=float, default=0.5)
    p.add_argument("--smc_pilot_mult", type=int, default=4)
    p.add_argument("--min_post_onset_length", type=int, default=50)
    p.add_argument("--ramp_width", type=float, default=2.0)
    p.add_argument("--lambda_pre_mult", type=float, default=4.0)
    p.add_argument("--embed_chunk_size", type=int, default=512)
    p.add_argument("--summary_source", choices=["latent", "theta"], default="latent")
    p.add_argument(
        "--distance_mode",
        choices=["mean", "rms", "rms_scaled", "sum"],
        default="rms_scaled",
    )
    p.add_argument(
        "--class_summary_mode",
        choices=[
            "none",
            "mean",
            "mean_scaled",
            "mean_std",
            "mean_std_scaled",
            "mean_se",
            "mean_std_se",
        ],
        default="mean_se",
    )
    p.add_argument("--sigma_p_fixed", type=float, default=1.0)
    p.add_argument("--pre_coupling", type=float, default=1e-3)
    p.add_argument("--prior_log_rate_loc", type=float, default=-0.5)
    p.add_argument("--prior_log_rate_scale", type=float, default=0.75)
    p.add_argument("--prior_b_loc", type=float, default=0.0)
    p.add_argument("--prior_b_scale", type=float, default=0.75)
    p.add_argument("--prior_log_sd1_loc", type=float, default=0.0)
    p.add_argument("--prior_log_sd1_scale", type=float, default=0.4)
    p.add_argument("--prior_min_rate", type=float, default=1e-4)
    p.add_argument("--prior_max_rate", type=float, default=20.0)
    p.add_argument("--prior_min_coupling", type=float, default=1e-6)
    p.add_argument("--prior_max_coupling", type=float, default=20.0)
    p.add_argument("--prior_min_sd1", type=float, default=1e-3)
    p.add_argument("--prior_max_sd1", type=float, default=50.0)
    p.add_argument(
        "--out_prefix",
        default="abc_posterior",
        help="Prefix for output files. Final filenames are "
        "<prefix>_<class>_<track_id>.h5 / corner_<...>.pdf",
    )
    p.add_argument("--out_h5", type=str, default=None)
    p.add_argument("--out_corner", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    ds = _load_dataset(
        args.data_file,
        onset_field=args.onset_field,
        onset_value=args.onset_value,
    )
    tracks = ds["tracks"]
    mask = ds["mask"]
    onset_idx = ds["onset_idx"]
    labels = ds["labels"]
    N_total = tracks.shape[0]

    # ------------------------------------------------------------------
    # Resolve track selection
    # ------------------------------------------------------------------
    if labels is None:
        if args.track_class is not None:
            print(
                f"Warning: --track_class={args.track_class} was given but "
                f"the data file has no 'labels' dataset. Ignoring."
            )
        sel_idx = np.arange(N_total)
        class_label = "all"
    else:
        if args.track_class is None:
            # No class specified but labels exist — use all tracks
            sel_idx = np.arange(N_total)
            class_label = "all"
        else:
            mask_class = labels == args.track_class
            sel_idx = np.where(mask_class)[0]
            if sel_idx.size == 0:
                raise ValueError(f"No tracks found with label == {args.track_class}.")
            class_label = str(args.track_class)

    y_meas = tracks[sel_idx]
    mask_meas = mask[sel_idx]
    onset_obs = onset_idx[sel_idx]

    # ------------------------------------------------------------------
    # Resolve ground truth (optional — silent when absent)
    # ------------------------------------------------------------------
    theta_true = _resolve_truth(ds, sel_idx, args.track_id, args.track_class)
    if theta_true is None:
        print(
            "No ground-truth fields detected in data file — running "
            "inference without truth comparison."
        )
    else:
        print(f"Ground-truth values resolved: {theta_true}")

    # ------------------------------------------------------------------
    # Build observation tensors
    # ------------------------------------------------------------------
    X = torch.as_tensor(y_meas, device=device, dtype=torch.float32)
    M = torch.as_tensor(mask_meas, device=device, dtype=torch.bool)
    O = torch.as_tensor(onset_obs, device=device, dtype=torch.long)
    t_max = X.shape[1]

    # ------------------------------------------------------------------
    # Load embedding model
    # ------------------------------------------------------------------
    model, reg_head, mean, std, theta_mean, theta_std = load_embedding_model(
        args.model_ckpt, t_max=t_max, device=device
    )
    if args.summary_source == "theta" and reg_head is None:
        raise ValueError(
            "summary_source='theta' requested but checkpoint lacks reg_head_state_dict."
        )

    # ------------------------------------------------------------------
    # Single-track vs population observation
    # ------------------------------------------------------------------
    if args.track_id is not None:
        if args.track_id >= X.shape[0]:
            raise IndexError(
                f"track_id={args.track_id} out of range ({X.shape[0]} tracks selected)."
            )
        Xo = X[args.track_id : args.track_id + 1]
        Mo = M[args.track_id : args.track_id + 1]
        Oo = O[args.track_id : args.track_id + 1]
        track_id_name = str(args.track_id)
    else:
        n_use = min(args.max_tracks, X.shape[0])
        if args.subset_mode == "first":
            use_idx = torch.arange(X.shape[0], device=device)[:n_use]
        elif args.subset_mode == "random":
            rng = np.random.default_rng(args.subset_seed)
            perm_np = rng.permutation(X.shape[0])[:n_use]
            use_idx = torch.as_tensor(perm_np, device=device, dtype=torch.long)
        else:
            raise ValueError(f"Unknown subset_mode: {args.subset_mode}")
        Xo = X[use_idx]
        Mo = M[use_idx]
        Oo = O[use_idx]
        track_id_name = f"AllTracks_n{n_use}"

    print(
        f"ABC using {Xo.shape[0]} tracks out of {X.shape[0]} selected tracks "
        f"(subset_mode={args.subset_mode}, subset_seed={args.subset_seed})."
    )

    s_obs = observed_summaries(
        model,
        reg_head,
        Xo,
        Mo,
        Oo,
        mean,
        std,
        theta_mean,
        theta_std,
        summary_source=args.summary_source,
        embed_chunk_size=args.embed_chunk_size,
    )

    onset_lo = 0
    onset_hi = t_max - args.min_post_onset_length
    if onset_hi < onset_lo:
        raise ValueError("Need t_max >= min_post_onset_length.")

    prior = dict(
        log_rate_loc=args.prior_log_rate_loc,
        log_rate_scale=args.prior_log_rate_scale,
        b_loc=args.prior_b_loc,
        b_scale=args.prior_b_scale,
        sd1_loc=args.prior_log_sd1_loc,
        sd1_scale=args.prior_log_sd1_scale,
        min_rate=args.prior_min_rate,
        max_rate=args.prior_max_rate,
        min_coupling=args.prior_min_coupling,
        max_coupling=args.prior_max_coupling,
        min_log_sd1=math.log(args.prior_min_sd1),
        max_log_sd1=math.log(args.prior_max_sd1),
    )

    # ------------------------------------------------------------------
    # Run ABC-SMC
    # ------------------------------------------------------------------
    theta_post, dist_post, w_post, eps_used = abc_latent_smc(
        model=model,
        reg_head=reg_head,
        mean=mean,
        std=std,
        theta_mean=theta_mean,
        theta_std=theta_std,
        s_obs=s_obs,
        obs_mask=Mo,
        obs_onset=Oo,
        t_max=t_max,
        dt=args.dt,
        prior=prior,
        n_keep=args.n_keep,
        batch_size=args.batch_size,
        reps=args.reps,
        measurement_sd=args.measurement_sd,
        spikes=args.spikes,
        T=args.smc_T,
        alpha=args.smc_alpha,
        pilot_mult=args.smc_pilot_mult,
        device=device,
        ramp_width=args.ramp_width,
        lambda_pre_mult=args.lambda_pre_mult,
        sigma_p_fixed=args.sigma_p_fixed,
        pre_coupling=args.pre_coupling,
        summary_source=args.summary_source,
        embed_chunk_size=args.embed_chunk_size,
        distance_mode=args.distance_mode,
        class_summary_mode=args.class_summary_mode,
    )

    # ------------------------------------------------------------------
    # Save posterior
    # ------------------------------------------------------------------
    attrs = {
        "reps": args.reps,
        "batch_size": args.batch_size,
        "dt": args.dt,
        "measurement_sd": args.measurement_sd,
        "summary_source": args.summary_source,
        "distance_mode": args.distance_mode,
        "class_summary_mode": args.class_summary_mode,
        "sigma_p_fixed": args.sigma_p_fixed,
        "lambda_pre_mult": args.lambda_pre_mult,
        "pre_coupling": args.pre_coupling,
        "track_class": -1 if args.track_class is None else args.track_class,
        "track_id": -1 if args.track_id is None else args.track_id,
        "data_file": args.data_file,
        "n_obs_tracks": int(Xo.shape[0]),
        "max_tracks": int(args.max_tracks),
        "subset_mode": args.subset_mode,
        "subset_seed": int(args.subset_seed),
        "seed": args.seed,
        "prior_log_rate_loc": args.prior_log_rate_loc,
        "prior_log_rate_scale": args.prior_log_rate_scale,
        "prior_b_loc": args.prior_b_loc,
        "prior_b_scale": args.prior_b_scale,
        "prior_log_sd1_loc": args.prior_log_sd1_loc,
        "prior_log_sd1_scale": args.prior_log_sd1_scale,
        "prior_min_rate": args.prior_min_rate,
        "prior_max_rate": args.prior_max_rate,
        "prior_min_coupling": args.prior_min_coupling,
        "prior_max_coupling": args.prior_max_coupling,
        "prior_min_sd1": args.prior_min_sd1,
        "prior_max_sd1": args.prior_max_sd1,
    }

    out_h5 = args.out_h5 or f"{args.out_prefix}_{class_label}_{track_id_name}.h5"
    save_abc_posterior_h5(
        out_h5,
        theta_post,
        dist_post,
        weights=w_post,
        attrs=attrs,
        resample_weighted=True,
        seed=args.seed,
    )
    print(f"Saved posterior to {out_h5}")

    # ------------------------------------------------------------------
    # Diagnostic corner plot — truths overlaid only when available
    # ------------------------------------------------------------------
    theta_np = theta_post.detach().cpu().numpy()
    theta_np[:, 2] = np.exp(theta_np[:, 2])  # log_sd1 -> sd1

    if theta_true is None:
        truths_plot = None
    else:
        truths_plot = [
            float(np.nanmean(np.asarray(theta_true[0], dtype=np.float64))),
            float(np.nanmean(np.asarray(theta_true[1], dtype=np.float64))),
            float(np.nanmean(np.asarray(theta_true[2], dtype=np.float64))),
        ]

    fig = corner.corner(
        theta_np,
        labels=[r"$\gamma_0$", r"$\gamma_1$", r"$\mathrm{sd1}$"],
        truths=truths_plot,
        show_titles=True,
    )
    out_pdf = args.out_corner or f"corner_{class_label}_{track_id_name}.pdf"
    fig.savefig(out_pdf, dpi=150)
    print(f"Saved corner plot to {out_pdf}")

    # ------------------------------------------------------------------
    # Posterior summary
    # ------------------------------------------------------------------
    theta_report = theta_post.detach().cpu().clone()
    theta_report[:, 2] = torch.exp(theta_report[:, 2])

    summary = {
        "n_kept": int(theta_post.shape[0]),
        "n_obs_tracks": int(Xo.shape[0]),
        "dist_min": float(dist_post.min().item()),
        "dist_median": float(dist_post.median().item()),
        "theta_mean": [float(x) for x in theta_report.mean(0)],
        "theta_std": [float(x) for x in theta_report.std(0, unbiased=False)],
        "epsilon": float(eps_used),
    }
    if truths_plot is not None:
        summary["theta_true"] = [float(x) for x in truths_plot]

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
