"""
ou_sim.py
=========
Unified simulator for the coupled polarisation-position OU model with
N-state switching dynamics. Supports both class-based and per-cell
sampling modes, full padding/masking, and provides per-theta ABC
wrappers.

This file replaces both `ou_sim_onset.py` and `ou_sim_switching.py`:

  - The exact closed-form (F, Q) construction from ou_sim_onset.py is
    retained (it matches the Kalman filter, has no Euler-Maruyama bias,
    and handles arbitrary dt without stability worries).

  - The Euler-Maruyama path from ou_sim_switching.py is dropped — for
    this linear-Gaussian SDE under piecewise-constant parameters the
    exact discrete-time recursion is strictly better and gives bit-for-bit
    consistency with the inference Kalman filter.

  - The legacy single-onset simulator is dropped — it is now the special
    case n_states=2 of the switching simulator with a single switch
    playing the role of the onset.

  - The simple OU baseline (without polarity) is dropped — nothing in
    the switching pipeline uses it.

The N-state convention matches the rest of the project:
  - state 0 = pre-onset uncoupled regime (small gamma2, fast decay)
  - states 1..N-1 = post-onset coupled regimes
  - log_tau is sorted ascending so state 0 has the smallest tau
  - the first switch time plays the role of the offset/onset

Per-cell physical parameters per state:
  - gamma0 (negative): polarity OU decay rate; tau = -1/gamma0
  - gamma1 (positive): position-from-polarity coupling
  - sigma_p (positive): polarity diffusion sd; usually fixed across states
  - sigma_r (positive): position diffusion sd

Two top-level entry points cover the usage:

  - `simulate_dataset(...)` is the training/evaluation generator. It
    runs class-based (Sobol parameter grid, suitable for the contrastive
    loss + classifier) or per-cell (independent per-track parameters,
    suitable for diagnostics or NPE training) modes.

  - `simulate_for_theta(...)` is the per-theta wrapper used by ABC and
    other simulation-in-the-loop frameworks. It accepts a flat theta
    vector and produces `reps` simulated tracks.
"""

import argparse
import math
import os
from typing import Dict, Optional, Tuple

import h5py
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import rcParams
from torch.quasirandom import SobolEngine
from tqdm import tqdm

rcParams["axes.facecolor"]   = "FFFFFF"
rcParams["savefig.facecolor"] = "FFFFFF"
rcParams["xtick.direction"]  = "in"
rcParams["ytick.direction"]  = "in"
rcParams.update({"figure.autolayout": True})


# ===========================================================================
# Closed-form F, Q construction (exact discrete-time recursion)
# ===========================================================================

def _torch_expm1_over_x(x: torch.Tensor, thresh: float = 1e-6) -> torch.Tensor:
    """Stable evaluation of (e^x - 1) / x, with a Taylor series near 0."""
    series = 1.0 + x / 2.0 + (x**2) / 6.0 + (x**3) / 24.0
    return torch.where(torch.abs(x) < thresh, series, torch.expm1(x) / x)


def _torch_w_alpha(alpha: torch.Tensor, thresh: float = 1e-6) -> torch.Tensor:
    """Stable evaluation of (1 - 2 r1 + r2) / alpha^2 with a series near 0."""
    r1 = _torch_expm1_over_x(alpha, thresh)
    r2 = _torch_expm1_over_x(2.0 * alpha, thresh)
    series = (1.0 / 3.0) + alpha / 4.0 + (7.0 / 60.0) * (alpha**2)
    exact = (1.0 - 2.0 * r1 + r2) / (alpha**2)
    return torch.where(torch.abs(alpha) < thresh, series, exact)


def _build_exact_FQ_coupled_pr(
    gamma1: torch.Tensor,
    gamma2: torch.Tensor,
    sigma_p: torch.Tensor,
    sigma_r: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Closed-form discrete-time (F, Q) for the coupled polarisation-position
    OU model under piecewise-constant parameters over dt.

    Inputs are vectors of length n (batch of cells); outputs are
    [n, 4, 4] tensors. State order: [p_x, p_y, x, y].
    """
    alpha = gamma1 * dt
    phi   = torch.exp(alpha)

    r1 = _torch_expm1_over_x(alpha)
    r2 = _torch_expm1_over_x(2.0 * alpha)

    beta = gamma2 * dt * r1
    q00  = (sigma_p**2) * dt * r2
    q01  = 0.5 * gamma2 * (sigma_p**2) * (dt**2) * (r1**2)
    q11  = (sigma_r**2) * dt + (gamma2**2) * (sigma_p**2) * (dt**3) * _torch_w_alpha(alpha)

    n = gamma1.shape[0]
    device, dtype = gamma1.device, gamma1.dtype
    F = torch.zeros((n, 4, 4), device=device, dtype=dtype)
    Q = torch.zeros((n, 4, 4), device=device, dtype=dtype)

    F[:, 0, 0] = phi
    F[:, 1, 1] = phi
    F[:, 2, 0] = beta
    F[:, 3, 1] = beta
    F[:, 2, 2] = 1.0
    F[:, 3, 3] = 1.0

    Q[:, 0, 0] = q00
    Q[:, 1, 1] = q00
    Q[:, 0, 2] = q01
    Q[:, 2, 0] = q01
    Q[:, 1, 3] = q01
    Q[:, 3, 1] = q01
    Q[:, 2, 2] = q11
    Q[:, 3, 3] = q11

    return F, Q


def _cholesky_with_jitter(Q: torch.Tensor, max_tries: int = 8) -> torch.Tensor:
    """Cholesky of Q with progressively larger jitter; eigh fallback."""
    Q = 0.5 * (Q + Q.transpose(-1, -2))
    eye = torch.eye(Q.shape[-1], device=Q.device, dtype=Q.dtype).expand_as(Q)
    jitter = 1e-6
    for _ in range(max_tries):
        try:
            return torch.linalg.cholesky(Q + jitter * eye)
        except torch._C._LinAlgError:
            jitter *= 10.0
    evals, evecs = torch.linalg.eigh(Q)
    evals = torch.clamp(evals, min=1e-8)
    Qp = evecs @ torch.diag_embed(evals) @ evecs.transpose(-1, -2)
    Qp = 0.5 * (Qp + Qp.transpose(-1, -2))
    return torch.linalg.cholesky(Qp)


# ===========================================================================
# State weights from switch times
# ===========================================================================

def compute_state_weights_from_switches(
    switch_times: torch.Tensor,    # [B, n_states - 1]  long or float
    t_max: int,
    n_states: int,
    ramp_width: float = 2.0,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build [B, t_max, n_states] sigmoid-blended state weights.

    The weights sum to 1 across the state axis at every timestep. The
    construction matches `build_explicit_switch_FQ_for_track` in the
    KF+MCMC code so the simulator and the filter produce identical
    per-timestep dynamics for the same parameters.
    """
    device = torch.device(device)

    if n_states == 1:
        B = switch_times.shape[0] if switch_times.ndim >= 1 else 1
        return torch.ones(B, t_max, 1, device=device, dtype=dtype)

    sw, _ = torch.sort(
        switch_times.to(device=device, dtype=dtype), dim=1
    )                                                          # [B, K]
    B = sw.shape[0]
    times = (torch.arange(t_max, device=device, dtype=dtype) + 1.0).view(1, t_max, 1)
    sw = sw.view(B, 1, n_states - 1)
    gates = torch.sigmoid((times - sw) / ramp_width)            # [B, T, K]

    weights = [1.0 - gates[:, :, 0:1]]
    for k in range(1, n_states - 1):
        weights.append(gates[:, :, k - 1 : k] - gates[:, :, k : k + 1])
    weights.append(gates[:, :, n_states - 2 : n_states - 1])
    return torch.cat(weights, dim=-1)                            # [B, T, N]


# ===========================================================================
# Onset / switch / mask samplers
# ===========================================================================

def sample_switch_times(
    n_cells: int,
    n_states: int,
    t_max: int,
    *,
    onset_lo: int = 10,
    onset_hi: int = 30,
    biological_lo: int = 35,
    biological_hi: int = 80,
    min_post_last_switch: int = 10,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Sample per-cell switch times: first is the onset (uniform on
    [onset_lo, onset_hi)), subsequent are biological (uniform on
    [biological_lo, biological_hi)). All sorted within each cell.
    """
    n_switches = n_states - 1
    if n_switches <= 0:
        return torch.zeros(n_cells, 0, dtype=torch.long, device=device)

    sw0 = torch.randint(
        low=onset_lo, high=onset_hi,
        size=(n_cells, 1), generator=generator, device=device,
    )
    if n_switches == 1:
        return sw0.long()

    bio_hi = min(biological_hi, t_max - min_post_last_switch)
    if bio_hi <= biological_lo:
        raise ValueError(
            f"biological_hi must exceed biological_lo + min_post_last_switch; "
            f"got biological_hi={biological_hi}, t_max={t_max}, "
            f"min_post_last_switch={min_post_last_switch}."
        )
    sw_bio = torch.randint(
        low=biological_lo, high=bio_hi,
        size=(n_cells, n_switches - 1), generator=generator, device=device,
    )

    out, _ = torch.sort(torch.cat([sw0, sw_bio], dim=1).long(), dim=1)
    return out


def _sample_discretized_normal_mask_lengths(
    n_cells: int,
    t_max: int,
    *,
    mask_mean: float,
    mask_sd: float,
    min_valid_length: int = 5,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Per-cell suffix mask length: truncated discretised normal capped so
    each cell retains at least `min_valid_length` valid timesteps.
    """
    mask_lengths = torch.normal(
        mean=torch.full((n_cells,), float(mask_mean), device=device),
        std=torch.full((n_cells,), float(mask_sd), device=device),
    )
    mask_lengths = torch.round(mask_lengths).to(torch.long)
    max_mask = t_max - min_valid_length
    return torch.clamp(mask_lengths, min=0, max=max_mask)


# ===========================================================================
# Parameter samplers (class-based and per-cell)
# ===========================================================================

def sample_per_class_params(
    n_param_sets: int,
    n_states: int,
    *,
    gamma1_range: Tuple[float, float] = (-1.0, -0.01),
    gamma2_range: Tuple[float, float] = (0.5, 5.0),
    sigma_r_range: Tuple[float, float] = (0.25, 1.2),
    sigma_p_value: float = 1.0,
    state0_uncoupled: bool = True,
    pre_decay_mult: float = 4.0,
    pre_coupling: float = 0.0,
    seed: int | None = None,
    device: str | torch.device = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Quasi-random per-class samples of (gamma1, gamma2, sigma_p, sigma_r)
    for each state. Returns dict with shape-[n_param_sets, n_states]
    tensors.

    With `state0_uncoupled=True`, state 0 is pinned to a fast-decaying
    uncoupled regime (gamma2 ~ 0, gamma1 = pre_decay_mult * state-1
    gamma1). Post-onset states are sorted by |gamma1| ascending so state
    1 has the slowest decay.

    With `state0_uncoupled=False`, every state is independently sampled
    from the supplied ranges and sorted by |gamma1| ascending.
    """
    n_post = n_states - 1 if state0_uncoupled else n_states
    if n_post < 1:
        raise ValueError("Need at least one post-onset state.")

    se = SobolEngine(dimension=3 * n_post, scramble=True, seed=seed)
    u = se.draw(n_param_sets).to(device=device)

    lows = torch.tensor(
        [gamma1_range[0], gamma2_range[0], sigma_r_range[0]] * n_post,
        device=device,
    )
    highs = torch.tensor(
        [gamma1_range[1], gamma2_range[1], sigma_r_range[1]] * n_post,
        device=device,
    )
    params = (lows + u * (highs - lows)).view(n_param_sets, n_post, 3)

    g1_post = params[:, :, 0]
    g2_post = params[:, :, 1]
    sr_post = params[:, :, 2]

    # Sort post-onset states by gamma1 descending: gamma1 itself is
    # negative, so "descending" puts the value closest to zero first,
    # i.e. smallest |gamma1| / largest tau first.
    order   = torch.argsort(g1_post, dim=1, descending=True)
    g1_post = torch.gather(g1_post, 1, order)
    g2_post = torch.gather(g2_post, 1, order)
    sr_post = torch.gather(sr_post, 1, order)

    if state0_uncoupled:
        gamma1_state0  = pre_decay_mult * g1_post[:, 0:1]
        gamma2_state0  = torch.full_like(g2_post[:, 0:1], float(pre_coupling))
        sigma_r_state0 = sr_post[:, 0:1]
        gamma1_set     = torch.cat([gamma1_state0,  g1_post], dim=1)
        gamma2_set     = torch.cat([gamma2_state0,  g2_post], dim=1)
        sigma_r_set    = torch.cat([sigma_r_state0, sr_post], dim=1)
    else:
        gamma1_set, gamma2_set, sigma_r_set = g1_post, g2_post, sr_post

    sigma_p_set = torch.full_like(gamma1_set, sigma_p_value)

    return {
        "gamma1_set":  gamma1_set.to(dtype=torch.float32),
        "gamma2_set":  gamma2_set.to(dtype=torch.float32),
        "sigma_p_set": sigma_p_set.to(dtype=torch.float32),
        "sigma_r_set": sigma_r_set.to(dtype=torch.float32),
    }


def sample_per_cell_params(
    n_cells: int,
    n_states: int,
    *,
    log_tau_loc_per_state: Optional[list[float]] = None,
    log_tau_scale_per_state: Optional[list[float]] = None,
    log_b_loc_per_state: Optional[list[float]] = None,
    log_b_scale_per_state: Optional[list[float]] = None,
    log_sd1_loc_per_state: Optional[list[float]] = None,
    log_sd1_scale_per_state: Optional[list[float]] = None,
    sigma_p_value: float = 1.0,
    enforce_log_tau_ordering: bool = True,
    seed: int | None = None,
    device: str | torch.device = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Per-cell log-normal sampling of (gamma1, gamma2, sigma_r) for each
    state, with optional ordering by sorted log_tau.

    Default per-state priors give state 0 a fast decorrelation
    (uncoupled-like, tau ~ 0.25) and subsequent states slower
    decorrelations (coupled-OU-like, tau ~ 1.0).
    """
    if log_tau_loc_per_state is None:
        log_tau_loc_per_state = [-1.39] + [0.0] * (n_states - 1)
    if log_tau_scale_per_state is None:
        log_tau_scale_per_state = [0.3] + [0.5] * (n_states - 1)
    if log_b_loc_per_state is None:
        log_b_loc_per_state = [-3.0] + [0.4] * (n_states - 1)
    if log_b_scale_per_state is None:
        log_b_scale_per_state = [0.3] + [0.5] * (n_states - 1)
    if log_sd1_loc_per_state is None:
        log_sd1_loc_per_state = [0.0] * n_states
    if log_sd1_scale_per_state is None:
        log_sd1_scale_per_state = [0.4] * n_states

    rng = np.random.default_rng(seed)

    log_tau = np.stack([
        rng.normal(log_tau_loc_per_state[k],
                   log_tau_scale_per_state[k],
                   size=n_cells)
        for k in range(n_states)
    ], axis=1)
    if enforce_log_tau_ordering and n_states > 1:
        log_tau = np.sort(log_tau, axis=1)
    gamma1 = -1.0 / np.exp(log_tau)                            # negative

    log_b = np.stack([
        rng.normal(log_b_loc_per_state[k],
                   log_b_scale_per_state[k],
                   size=n_cells)
        for k in range(n_states)
    ], axis=1)
    gamma2 = np.exp(log_b)                                     # positive

    log_sd1 = np.stack([
        rng.normal(log_sd1_loc_per_state[k],
                   log_sd1_scale_per_state[k],
                   size=n_cells)
        for k in range(n_states)
    ], axis=1)
    sigma_r = np.exp(log_sd1)

    sigma_p = np.full_like(gamma1, sigma_p_value, dtype=np.float32)

    t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    return {
        "gamma1_per_cell":  t(gamma1),
        "gamma2_per_cell":  t(gamma2),
        "sigma_p_per_cell": t(sigma_p),
        "sigma_r_per_cell": t(sigma_r),
    }


# ===========================================================================
# Core simulator (exact discrete-time recursion, N-state switching)
# ===========================================================================

def simulate_coupled_pr_switching(
    n_cells: int,
    t_max: int,
    dt: float,
    p0: torch.Tensor,                                # [n_cells, 2] (informational; sim uses zero init)
    gamma1_per_state: torch.Tensor,                  # [n_cells, n_states]  (negative)
    gamma2_per_state: torch.Tensor,                  # [n_cells, n_states]  (non-negative)
    sigma_p_per_state: torch.Tensor,                 # [n_cells, n_states]
    sigma_r_per_state: torch.Tensor,                 # [n_cells, n_states]
    switch_times: torch.Tensor,                      # [n_cells, n_states - 1]
    *,
    measurement_sd: float = 0.1,
    spikes: bool = False,
    device: str | torch.device = "cpu",
    ramp_width: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Exact discrete-time recursion of the coupled polarisation-position
    OU model with N-state switching.

    The full latent state is [p_x, p_y, x, y]; the polarity initial value
    is taken as zero internally (matching the original onset-only
    simulator's convention). Spikes, when enabled, are a deterministic
    additive bump to polarity every 30 timesteps; they are gated by
    `1 - state_weights[..., 0]` so they only fire post-first-switch.

    Returns
    -------
    p_traj       : [n_cells, t_max, 2]   polarity trajectory
    r_traj       : [n_cells, t_max, 2]   observed positions (with measurement noise)
    r_traj_true  : [n_cells, t_max, 2]   true positions (no measurement noise)
    """
    device = torch.device(device)
    dtype  = p0.dtype
    n_states = gamma1_per_state.shape[1]

    g1 = gamma1_per_state.to(device=device, dtype=dtype)
    g2 = gamma2_per_state.to(device=device, dtype=dtype)
    sp = sigma_p_per_state.to(device=device, dtype=dtype)
    sr = sigma_r_per_state.to(device=device, dtype=dtype)

    state_weights = compute_state_weights_from_switches(
        switch_times=switch_times,
        t_max=t_max,
        n_states=n_states,
        ramp_width=ramp_width,
        device=device,
        dtype=dtype,
    )                                                # [n_cells, t_max, n_states]

    g1_t = (state_weights * g1.unsqueeze(1)).sum(dim=-1)     # [n_cells, t_max]
    g2_t = (state_weights * g2.unsqueeze(1)).sum(dim=-1)
    sp_t = (state_weights * sp.unsqueeze(1)).sum(dim=-1)
    sr_t = (state_weights * sr.unsqueeze(1)).sum(dim=-1)

    # Initial state: zero polarity (matches the original simulator).
    x0 = torch.zeros_like(p0, device=device)
    p_init = torch.zeros_like(p0, device=device)
    z = torch.cat([p_init, x0], dim=1)                       # [n_cells, 4]

    p_traj      = torch.empty(n_cells, t_max, 2, device=device, dtype=dtype)
    r_traj      = torch.empty(n_cells, t_max, 2, device=device, dtype=dtype)
    r_traj_true = torch.empty(n_cells, t_max, 2, device=device, dtype=dtype)

    # Spike gate: enabled post-first-switch (only meaningful for N>=2).
    if spikes and n_states >= 2:
        spike_gate_t = 1.0 - state_weights[:, :, 0]          # [n_cells, t_max]
    else:
        spike_gate_t = torch.zeros(n_cells, t_max, device=device, dtype=dtype)

    for t in range(t_max):
        p_traj[:, t]      = z[:, :2]
        r_traj_true[:, t] = z[:, 2:]
        r_traj[:, t]      = z[:, 2:] + torch.normal(
            mean=0.0, std=measurement_sd, size=z[:, 2:].shape, device=device,
        )

        if t == t_max - 1:
            break

        if spikes and ((t + 1) % 30 == 0):
            z[:, :2] = z[:, :2] + (10.0 * spike_gate_t[:, t : t + 1])

        F_t, Q_t = _build_exact_FQ_coupled_pr(
            gamma1=g1_t[:, t],
            gamma2=g2_t[:, t],
            sigma_p=sp_t[:, t],
            sigma_r=sr_t[:, t],
            dt=dt,
        )
        L_t = _cholesky_with_jitter(Q_t)
        eps = torch.randn(n_cells, 4, device=device, dtype=dtype)
        w_t = torch.einsum("nij,nj->ni", L_t, eps)
        z   = torch.einsum("nij,nj->ni", F_t, z) + w_t

    return p_traj, r_traj, r_traj_true


# ===========================================================================
# High-level dataset generator
# ===========================================================================

def simulate_dataset(
    n_cells: int,
    t_max: int,
    *,
    n_states: int = 2,
    mode: str = "class",                          # "class" | "per_cell"
    n_param_sets: int = 16,
    dt: float = 1.0,
    measurement_sd: float = 0.1,
    spikes: bool = False,
    device: str | torch.device = "cpu",
    p_params: Tuple[float, float] = (5.0, 0.5),
    # class-mode arguments
    gamma1_range: Tuple[float, float] = (-1.0, -0.01),
    gamma2_range: Tuple[float, float] = (0.5, 5.0),
    sigma_r_range: Tuple[float, float] = (0.25, 1.2),
    sigma_p_value: float = 1.0,
    state0_uncoupled: bool = True,
    pre_decay_mult: float = 4.0,
    pre_coupling: float = 0.0,
    # per-cell-mode arguments
    log_tau_loc_per_state: Optional[list[float]] = None,
    log_tau_scale_per_state: Optional[list[float]] = None,
    log_b_loc_per_state: Optional[list[float]] = None,
    log_b_scale_per_state: Optional[list[float]] = None,
    log_sd1_loc_per_state: Optional[list[float]] = None,
    log_sd1_scale_per_state: Optional[list[float]] = None,
    enforce_log_tau_ordering: bool = True,
    # switch-time sampler
    onset_lo: int = 10,
    onset_hi: int = 30,
    biological_lo: int = 35,
    biological_hi: int = 80,
    min_post_last_switch: int = 10,
    # masking
    mask_length_mean: float = 0.0,
    mask_length_sd: float = 0.0,
    min_valid_length: int = 5,
    missing_prob: float = 0.0,
    zero_masked_values: bool = False,
    # general
    seed: int | None = None,
    ramp_width: float = 2.0,
) -> dict:
    """
    Top-level dataset generator. Returns everything the training script,
    KF+MCMC fitter, and ABC inference need.

    Parameters
    ----------
    mode : {"class", "per_cell"}
        - "class": Sobol-designed parameter grid with n_param_sets
          unique parameter vectors, cells randomly assigned to a class.
          Produces class labels usable by the classifier head and the
          contrastive loss.
        - "per_cell": every cell has independently sampled parameters
          from log-normal priors. Suitable for diagnostics and density
          estimation; class labels are returned as zeros.

    Padding / masking
    -----------------
        mask_length_mean, mask_length_sd : suffix-pad-mask distribution
            (truncated-normal). Set both to 0 for no padding.
        missing_prob : per-timestep probability of missing data within
            the valid window (in addition to the suffix mask).
        zero_masked_values : if True, replace masked positions with 0.0
            in `tracks`. The unmasked observed positions are always
            available in `tracks_obs_full`.

    Returns
    -------
    dict with keys:
        tracks            : [n_cells, t_max, 2]  (possibly zero-masked)
        tracks_obs_full   : [n_cells, t_max, 2]  observed positions (no zeroing)
        tracks_true       : [n_cells, t_max, 2]  true positions (no measurement noise)
        polarity          : [n_cells, t_max, 2]  latent polarity trajectory
        mask              : [n_cells, t_max]     True == padded/invalid
        labels            : [n_cells]            class id (class mode) or zeros (per_cell)
        switch_times      : [n_cells, n_states - 1]
        onset_idx         : [n_cells]            alias for switch_times[:, 0]
        valid_lengths     : [n_cells]
        mask_lengths      : [n_cells]
        gamma1            : [n_cells, n_states]  per-cell physical params
        gamma2            : [n_cells, n_states]
        sigma_p           : [n_cells, n_states]
        sigma_r           : [n_cells, n_states]
        param_sets        : dict of [n_param_sets, n_states] (class mode)
                            or empty dict (per_cell mode)
        n_states          : int
        ramp_width        : float
        mode              : str
    """
    if n_states < 1:
        raise ValueError("n_states must be >= 1.")
    if mode == "class" and n_param_sets > n_cells:
        raise ValueError("n_param_sets must not exceed n_cells")
    if mode not in ("class", "per_cell"):
        raise ValueError(f"mode must be 'class' or 'per_cell'; got {mode!r}")

    device = torch.device(device)
    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)

    # -- per-cell physical parameters -------------------------------------
    if mode == "class":
        param_sets = sample_per_class_params(
            n_param_sets=n_param_sets,
            n_states=n_states,
            gamma1_range=gamma1_range,
            gamma2_range=gamma2_range,
            sigma_r_range=sigma_r_range,
            sigma_p_value=sigma_p_value,
            state0_uncoupled=state0_uncoupled,
            pre_decay_mult=pre_decay_mult,
            pre_coupling=pre_coupling,
            seed=seed,
            device=device,
        )
        labels = torch.randint(0, n_param_sets, (n_cells,),
                                generator=rng, device=device)
        gamma1_per_cell  = param_sets["gamma1_set"][labels]
        gamma2_per_cell  = param_sets["gamma2_set"][labels]
        sigma_p_per_cell = param_sets["sigma_p_set"][labels]
        sigma_r_per_cell = param_sets["sigma_r_set"][labels]
    else:
        param_sets = {}
        labels = torch.zeros(n_cells, dtype=torch.long, device=device)
        per_cell = sample_per_cell_params(
            n_cells=n_cells,
            n_states=n_states,
            log_tau_loc_per_state=log_tau_loc_per_state,
            log_tau_scale_per_state=log_tau_scale_per_state,
            log_b_loc_per_state=log_b_loc_per_state,
            log_b_scale_per_state=log_b_scale_per_state,
            log_sd1_loc_per_state=log_sd1_loc_per_state,
            log_sd1_scale_per_state=log_sd1_scale_per_state,
            sigma_p_value=sigma_p_value,
            enforce_log_tau_ordering=enforce_log_tau_ordering,
            seed=seed,
            device=device,
        )
        gamma1_per_cell  = per_cell["gamma1_per_cell"]
        gamma2_per_cell  = per_cell["gamma2_per_cell"]
        sigma_p_per_cell = per_cell["sigma_p_per_cell"]
        sigma_r_per_cell = per_cell["sigma_r_per_cell"]

    # -- per-cell switch times --------------------------------------------
    switch_times = sample_switch_times(
        n_cells=n_cells,
        n_states=n_states,
        t_max=t_max,
        onset_lo=onset_lo, onset_hi=onset_hi,
        biological_lo=biological_lo, biological_hi=biological_hi,
        min_post_last_switch=min_post_last_switch,
        device=device, generator=rng,
    )
    onset_idx = (switch_times[:, 0].clone() if n_states >= 2 else
                  torch.zeros(n_cells, dtype=torch.long, device=device))

    # -- initial polarity (informational; sim uses zero init) -------------
    shape = torch.ones((n_cells, 2), device=device)
    p_set = torch.normal(shape * p_params[0], shape * p_params[1])

    # -- simulate ---------------------------------------------------------
    p_traj, r_traj_obs_full, r_traj_true = simulate_coupled_pr_switching(
        n_cells=n_cells, t_max=t_max, dt=dt, p0=p_set,
        gamma1_per_state=gamma1_per_cell,
        gamma2_per_state=gamma2_per_cell,
        sigma_p_per_state=sigma_p_per_cell,
        sigma_r_per_state=sigma_r_per_cell,
        switch_times=switch_times,
        measurement_sd=measurement_sd, spikes=spikes,
        device=device, ramp_width=ramp_width,
    )

    # -- masking ----------------------------------------------------------
    mask_lengths = _sample_discretized_normal_mask_lengths(
        n_cells, t_max,
        mask_mean=mask_length_mean, mask_sd=mask_length_sd,
        min_valid_length=min_valid_length, device=device,
    )
    valid_lengths = t_max - mask_lengths
    time_index    = torch.arange(t_max, device=device)[None, :]
    suffix_pad_mask = time_index >= valid_lengths[:, None]

    if missing_prob > 0.0:
        random_missing = (
            torch.rand(n_cells, t_max, generator=rng, device=device) < missing_prob
        )
        random_missing = random_missing & (~suffix_pad_mask)
    else:
        random_missing = torch.zeros(n_cells, t_max, dtype=torch.bool, device=device)
    pad_mask = suffix_pad_mask | random_missing

    tracks = r_traj_obs_full.clone()
    if zero_masked_values:
        tracks[pad_mask] = 0.0

    return {
        "tracks":          tracks,
        "tracks_obs_full": r_traj_obs_full,
        "tracks_true":     r_traj_true,
        "polarity":        p_traj,
        "mask":            pad_mask,
        "labels":          labels,
        "switch_times":    switch_times,
        "onset_idx":       onset_idx,
        "valid_lengths":   valid_lengths,
        "mask_lengths":    mask_lengths,
        "gamma1":          gamma1_per_cell,    # [n_cells, n_states]
        "gamma2":          gamma2_per_cell,
        "sigma_p":         sigma_p_per_cell,
        "sigma_r":         sigma_r_per_cell,
        "param_sets":      param_sets,
        "n_states":        n_states,
        "ramp_width":      ramp_width,
        "mode":            mode,
    }


# ===========================================================================
# HDF5 I/O
# ===========================================================================

def save_dataset(out: dict, path: str) -> None:
    """
    Write a dataset produced by `simulate_dataset` to HDF5.

    Per-cell parameter arrays have shape [n_cells, n_states]; per-class
    arrays (class mode only) are stored under the `<param>_set` keys
    with shape [n_param_sets, n_states]. The legacy single-onset
    `onset_idx` is also written for backwards compatibility with
    inference scripts expecting a scalar onset per cell.
    """
    with h5py.File(path, "w") as f:
        for k in ("tracks", "tracks_obs_full", "tracks_true", "polarity",
                  "mask", "labels", "switch_times", "onset_idx",
                  "valid_lengths", "mask_lengths",
                  "gamma1", "gamma2", "sigma_p", "sigma_r"):
            f.create_dataset(k, data=out[k].cpu().numpy(), compression="gzip")
        for name, tensor in out["param_sets"].items():
            f.create_dataset(name, data=tensor.cpu().numpy(), compression="gzip")
        f.attrs["n_states"]   = int(out["n_states"])
        f.attrs["ramp_width"] = float(out["ramp_width"])
        f.attrs["mode"]       = str(out["mode"])


def read_dataset(path: str) -> dict:
    """Read a dataset produced by `save_dataset` into a plain dict."""
    out = {}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            out[k] = f[k][:]
        for k in f.attrs.keys():
            out[k] = f.attrs[k]
    return out


# ===========================================================================
# Per-theta wrapper for ABC and simulation-in-the-loop
# ===========================================================================

THETA_PER_STATE = 3   # (gamma1, gamma2, sigma_r) per state; sigma_p is fixed


def theta_dim_for(n_states: int) -> int:
    """Flat theta dimension: 3 params per state + (n_states - 1) switches."""
    return THETA_PER_STATE * n_states + max(0, n_states - 1)


def split_theta(theta: torch.Tensor, n_states: int) -> dict:
    """
    Split a flat per-cell theta vector into its per-state and switch-time
    blocks. The flat layout matches what the training and ABC scripts use:
        [gamma1_s0, gamma2_s0, sigma_r_s0,
         gamma1_s1, gamma2_s1, sigma_r_s1, ...,
         switch_0, switch_1, ...]
    """
    state_dim = THETA_PER_STATE * n_states
    state_block = theta[:, :state_dim].view(-1, n_states, THETA_PER_STATE)
    switches    = theta[:, state_dim:]
    return {
        "gamma1_per_state":  state_block[:, :, 0],
        "gamma2_per_state":  state_block[:, :, 1],
        "sigma_r_per_state": state_block[:, :, 2],
        "switch_times":      switches,
    }


def simulate_for_theta(
    theta: torch.Tensor,
    t_max: int,
    dt: float,
    *,
    n_states: int,
    reps: int = 1,
    p0: torch.Tensor | None = None,
    p_params: Tuple[float, float] = (5.0, 0.5),
    measurement_sd: float = 0.1,
    sigma_p_value: float = 1.0,
    spikes: bool = False,
    device: str | torch.device = "cpu",
    ramp_width: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-theta wrapper for ABC and other simulation-in-the-loop methods.

    Each row of `theta` (shape [B, theta_dim]) is replicated `reps`
    times and simulated, returning [B * reps, t_max, 2] tracks and an
    all-False pad_mask.

    The theta layout is the flat per-cell vector described in
    `split_theta`. `sigma_p` is fixed across all states at
    `sigma_p_value` (matches the training convention).
    """
    expected_dim = theta_dim_for(n_states)
    theta = torch.as_tensor(theta, device=device, dtype=torch.float32).view(-1, expected_dim)
    b = theta.shape[0]
    n = b * reps

    s = split_theta(theta, n_states)
    g1_b = s["gamma1_per_state"]
    g2_b = s["gamma2_per_state"]
    sr_b = s["sigma_r_per_state"]
    sp_b = torch.full_like(g1_b, sigma_p_value)
    sw_b = s["switch_times"]

    g1 = g1_b.repeat_interleave(reps, dim=0)
    g2 = g2_b.repeat_interleave(reps, dim=0)
    sp = sp_b.repeat_interleave(reps, dim=0)
    sr = sr_b.repeat_interleave(reps, dim=0)
    sw = sw_b.repeat_interleave(reps, dim=0)

    if p0 is None:
        mu, sd = p_params
        p0 = torch.normal(
            torch.full((n, 2), mu, device=device),
            torch.full((n, 2), sd, device=device),
        )
    else:
        p0 = torch.as_tensor(p0, device=device, dtype=torch.float32)
        if p0.ndim == 1:
            p0 = p0.view(1, 2).repeat(n, 1)
        elif p0.shape[0] != n:
            k = math.ceil(n / p0.shape[0])
            p0 = p0.repeat(k, 1)[:n]

    _, tracks, _ = simulate_coupled_pr_switching(
        n_cells=n, t_max=t_max, dt=dt, p0=p0,
        gamma1_per_state=g1, gamma2_per_state=g2,
        sigma_p_per_state=sp, sigma_r_per_state=sr,
        switch_times=sw,
        measurement_sd=measurement_sd, spikes=spikes,
        device=device, ramp_width=ramp_width,
    )
    pad_mask = torch.zeros(n, t_max, dtype=torch.bool, device=device)
    return tracks, pad_mask


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified coupled-PR switching simulator."
    )
    parser.add_argument("--mode", choices=["class", "per_cell"], default="class")
    parser.add_argument("--n_states", type=int, default=2,
                        help="Number of regimes (>= 1). 2 = onset-only, "
                             ">=3 adds biological switches after the offset.")
    parser.add_argument("--n_cells", type=int, default=32768)
    parser.add_argument("--n_param_sets", type=int, default=32,
                        help="Number of Sobol parameter classes (class mode only).")
    parser.add_argument("--t_max", type=int, default=100)
    parser.add_argument("--dt", type=float, default=1 / 3)
    parser.add_argument("--measurement_sd", type=float, default=2.0)
    parser.add_argument("--mask_length_mean", type=float, default=5.0)
    parser.add_argument("--mask_length_sd", type=float, default=20.0)
    parser.add_argument("--min_valid_length", type=int, default=50)
    parser.add_argument("--missing_prob", type=float, default=0.0)
    parser.add_argument("--ramp_width", type=float, default=2.0)
    parser.add_argument("--lambda_pre_mult", type=float, default=4.0)
    parser.add_argument("--pre_coupling", type=float, default=0.0)
    parser.add_argument("--sigma_p_fixed", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_path", type=str, default="coupled_pr_switching.h5")
    parser.add_argument("--plot_dir", type=str, default="plots")
    parser.add_argument("--n_plot_classes", type=int, default=8,
                        help="Number of example tracks to plot (one per class).")
    args = parser.parse_args()

    out = simulate_dataset(
        n_cells=args.n_cells,
        t_max=args.t_max,
        n_states=args.n_states,
        mode=args.mode,
        n_param_sets=args.n_param_sets,
        dt=args.dt,
        measurement_sd=args.measurement_sd,
        spikes=False,
        mask_length_mean=args.mask_length_mean,
        mask_length_sd=args.mask_length_sd,
        min_valid_length=args.min_valid_length,
        missing_prob=args.missing_prob,
        ramp_width=args.ramp_width,
        sigma_p_value=args.sigma_p_fixed,
        pre_decay_mult=args.lambda_pre_mult,
        pre_coupling=args.pre_coupling,
        seed=args.seed,
        device="cpu",
    )
    save_dataset(out, args.out_path)
    print(f"Wrote dataset to {args.out_path}")

    # -- diagnostic plots --------------------------------------------------
    os.makedirs(args.plot_dir, exist_ok=True)
    tracks   = out["tracks"]
    pad_mask = out["mask"]
    labels   = out["labels"]
    g1       = out["gamma1"]
    g2       = out["gamma2"]
    sr       = out["sigma_r"]
    n_states = out["n_states"]

    if args.mode == "class":
        # Plot one representative cell from each of the first N classes.
        n_classes_to_plot = min(args.n_plot_classes,
                                 out["param_sets"]["gamma1_set"].shape[0])
        for cid in range(n_classes_to_plot):
            idx = (labels == cid).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            i = int(idx[0])
            traj = tracks[i]
            valid = ~pad_mask[i]
            x, y = traj[valid].T
            per_state_strs = [
                f"s{s}: γ1={g1[i, s].item():.2f}, γ2={g2[i, s].item():.2f}, σr={sr[i, s].item():.2f}"
                for s in range(n_states)
            ]
            title = f"class {cid}\n" + "\n".join(per_state_strs)
            plt.figure()
            plt.plot(x.cpu().numpy(), y.cpu().numpy(), linewidth=1)
            plt.scatter(x[0].cpu(), y[0].cpu(), marker="o")
            plt.scatter(x[-1].cpu(), y[-1].cpu(), marker="x")
            plt.title(title)
            plt.axis("equal")
            plt.savefig(os.path.join(args.plot_dir, f"coupled_pr_switching_class_{cid}.png"))
            plt.clf()
            plt.close()
    else:
        # Per-cell mode: plot the first N cells.
        for i in range(min(args.n_plot_classes, tracks.shape[0])):
            traj = tracks[i]
            valid = ~pad_mask[i]
            x, y = traj[valid].T
            per_state_strs = [
                f"s{s}: γ1={g1[i, s].item():.2f}, γ2={g2[i, s].item():.2f}, σr={sr[i, s].item():.2f}"
                for s in range(n_states)
            ]
            title = f"cell {i}\n" + "\n".join(per_state_strs)
            plt.figure()
            plt.plot(x.cpu().numpy(), y.cpu().numpy(), linewidth=1)
            plt.scatter(x[0].cpu(), y[0].cpu(), marker="o")
            plt.scatter(x[-1].cpu(), y[-1].cpu(), marker="x")
            plt.title(title)
            plt.axis("equal")
            plt.savefig(os.path.join(args.plot_dir, f"coupled_pr_switching_cell_{i}.png"))
            plt.clf()
            plt.close()

    print(f"Wrote example plots to {args.plot_dir}/")
