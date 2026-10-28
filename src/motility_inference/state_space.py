"""Hierarchical Bayesian inference for target-associated OU movement changes.

The latent state is (velocity, position) on each axis.  Its exact integrated-OU
transition is used when coefficients are frozen over one recording interval.
Cell log-speed intercepts have a population Normal prior. Global parameters and
cell effects are sampled; latent paths are marginalized by a Kalman filter.

Distance gates are calculated from *observed* positions, never simulator truth.
Consequently this is a conditional-covariate approximation when localization
noise is nonzero. It is also a coarse approximation for simulations with
sub-frame varying gates, forces, binding, or spontaneous temporal changes. The
diagnostics flag these limitations; they are not corrected by a good R-hat.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import math
from time import perf_counter
from typing import Any

import numpy as np
from numba import njit
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from scipy.special import ndtr, ndtri

from .priors import PARAMETER_NAMES, PRIOR_LOW, PRIOR_HIGH


@njit(cache=True)
def ou_transition(speed: float, tau: float, jitter: float, dt: float):
    """Return phi, integrated drift h, and Qvv,Qvx,Qxx for stationary SD speed.

    dv = -v/tau dt + sqrt(2*speed**2/tau) dWv;
    dx = v dt + jitter dWx, with independent Brownian motions.
    """
    d = dt / tau
    one_minus_phi = -math.expm1(-d)
    one_minus_phi2 = -math.expm1(-2.0 * d)
    phi = math.exp(-d)
    h = tau * one_minus_phi
    qvv = speed * speed * one_minus_phi2
    qvx = speed * speed * tau * one_minus_phi * one_minus_phi
    if d < 1e-3:
        integral = d**3 * (2.0/3.0 + d * (-0.5 + d * (7.0/30.0 + d * (-1.0/12.0 + d * 31.0/1260.0))))
    else:
        integral = 2.0*d - 4.0*one_minus_phi + one_minus_phi2
    qxx = speed * speed * tau * tau * integral + jitter*jitter*dt
    return phi, h, qvv, qvx, qxx


@njit(cache=True)
def _cell_loglik(y, gate, intervals, theta, offset, localization_sd):
    """Conditional log p(y[1:] | y[0], observed gates, theta, offset)."""
    r = localization_sd * localization_sd
    speed0 = math.exp(theta[0] + offset + theta[3]*gate[0])
    mv = np.zeros(2)
    mx = y[0].copy()
    pvv = np.full(2, speed0*speed0)
    pvx = np.zeros(2)
    # Flat prior on initial true position, conditioned on its first observation.
    pxx = np.full(2, r)
    ll = 0.0
    for t in range(y.shape[0]-1):
        g = gate[t]
        speed = math.exp(theta[0] + offset + theta[3]*g)
        tau = math.exp(theta[1] + theta[4]*g)
        jitter = math.exp(theta[2] + theta[5]*g)
        phi, h, qvv, qvx, qxx = ou_transition(speed, tau, jitter, intervals[t])
        for axis in range(2):
            mx[axis] += h*mv[axis]
            mv[axis] *= phi
            a = phi*phi*pvv[axis] + qvv
            b = phi*(pvx[axis] + h*pvv[axis]) + qvx
            c = pxx[axis] + 2*h*pvx[axis] + h*h*pvv[axis] + qxx
            observation = y[t+1, axis]
            if math.isfinite(observation):
                innovation = observation - mx[axis]
                variance = c + r
                if variance <= 0 or not math.isfinite(variance):
                    return -np.inf
                ll -= 0.5*(math.log(2*math.pi*variance) + innovation*innovation/variance)
                mv[axis] += b/variance*innovation
                mx[axis] += c/variance*innovation
                pvv[axis] = max(a-b*b/variance, 0.0)
                pvx[axis] = b*r/variance
                pxx[axis] = c*r/variance
            else:
                pvv[axis], pvx[axis], pxx[axis] = a, b, c
    return ll


@njit(cache=True)
def _all_loglik(positions, gate, intervals, theta, offsets, localization_sd):
    result = 0.0
    for cell in range(positions.shape[0]):
        result += _cell_loglik(positions[cell], gate[cell], intervals, theta, offsets[cell], localization_sd)
    return result


def _config(data: dict) -> dict:
    value = data.get("config", {})
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return value
    raise TypeError("data['config'] must be a dataclass or dictionary")


def prepare_data(data: dict) -> dict:
    """Validate data and derive gates from observations, without reading truth."""
    from .simulator import proximity

    positions = np.ascontiguousarray(data["positions"], dtype=np.float64)
    targets = np.asarray(data["targets"], dtype=np.float64)
    times = np.asarray(data["times"], dtype=np.float64)
    if positions.ndim != 3 or positions.shape[-1] != 2 or positions.shape[1] < 3:
        raise ValueError("positions must have shape [cells, at least 3 frames, 2]")
    if times.shape != (positions.shape[1],) or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("times must be finite, strictly increasing and aligned to positions")
    if targets.ndim != 3 or targets.shape[1:] != positions.shape[1:] or targets.shape[0] < 1 or not np.all(np.isfinite(targets)):
        raise ValueError("targets must be finite [targets, frames, 2] aligned to positions")
    if not np.all(np.isfinite(positions[:, 0])) or np.any(np.isinf(positions)):
        raise ValueError("initial observations must be finite; later missing positions may be NaN")
    observed_covariates = positions.copy()
    missing = ~np.isfinite(observed_covariates)
    for cell in range(positions.shape[0]):
        for axis in range(2):
            finite = np.isfinite(observed_covariates[cell, :, axis])
            if finite.sum() < 3:
                raise ValueError("each coordinate of each cell needs at least 3 finite observations")
            if not np.all(finite):
                observed_covariates[cell, :, axis] = np.interp(times, times[finite], observed_covariates[cell, finite, axis])
    config = _config(data)
    contact_distance = config.get("contact_distance", config.get("killer_radius", 6)+config.get("target_radius", 6))
    prox = proximity(observed_covariates, targets, config.get("box_size", 120.0), contact_distance, config.get("interaction_length", 10.0))
    gate = prox["gate"] if isinstance(prox, dict) else prox[0]
    localization_sd = float(config.get("localization_sd", 0.15))
    heterogeneity_sd = float(config.get("heterogeneity_sd", 0.25))
    if not math.isfinite(localization_sd) or localization_sd < 0:
        raise ValueError("localization_sd must be finite and non-negative")
    if not math.isfinite(heterogeneity_sd) or heterogeneity_sd <= 0:
        raise ValueError("hierarchical inference requires positive heterogeneity_sd")
    limitations = []
    if localization_sd > 0:
        limitations.append("Distance gate conditions on noisy observed position; localization error in the covariate is not integrated out.")
    if missing.any():
        limitations.append("Missing positions are interpolated only to construct distance covariates; their observation updates are skipped.")
    if config.get("substeps", 1) != 1:
        limitations.append("Within-frame coefficient changes are approximated by the observed start-of-frame gate.")
    for name in ("repulsion", "adhesion", "binding_rate", "spontaneous_sd"):
        if config.get(name, 0) != 0:
            limitations.append(f"The fitted state-space model omits simulator component {name}.")
    if np.ptp(gate) < 0.25:
        limitations.append("Observed proximity has little variation; baseline and distance-response effects may be weakly identifiable.")
    return {"positions": positions, "gate": np.ascontiguousarray(gate), "intervals": np.ascontiguousarray(np.diff(times)),
            "localization_sd": localization_sd, "heterogeneity_sd": heterogeneity_sd,
            "limitations": limitations, "n_cells": positions.shape[0], "times": times}


def log_likelihood(data: dict, theta: np.ndarray, offsets: np.ndarray | None = None) -> float:
    prepared = prepare_data(data)
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (6,) or not np.all(np.isfinite(theta)):
        raise ValueError("theta must be a finite vector of length 6")
    offsets = np.zeros(prepared["n_cells"]) if offsets is None else np.asarray(offsets, dtype=float)
    if offsets.shape != (prepared["n_cells"],) or not np.all(np.isfinite(offsets)):
        raise ValueError("offsets must contain one finite value per cell")
    return float(_all_loglik(prepared["positions"], prepared["gate"], prepared["intervals"], theta, offsets, prepared["localization_sd"]))


def _hessian(fun, x, step=2e-4):
    n = len(x)
    out = np.empty((n, n))
    center = fun(x)
    for i in range(n):
        ei = np.zeros(n); ei[i] = step
        out[i, i] = (fun(x+ei)-2*center+fun(x-ei))/(step*step)
        for j in range(i):
            ej = np.zeros(n); ej[j] = step
            value = (fun(x+ei+ej)-fun(x+ei-ej)-fun(x-ei+ej)+fun(x-ei-ej))/(4*step*step)
            out[i, j] = out[j, i] = value
    return out


def summarize_samples(samples: np.ndarray, *, rope: float = math.log(1.2)) -> dict:
    """Marginal posterior intervals, sign probabilities and practical-null mass."""
    flat = np.asarray(samples).reshape(-1, 6)
    quantiles = np.quantile(flat, [0.025, 0.05, 0.5, 0.95, 0.975], axis=0)
    result = {}
    for j, name in enumerate(PARAMETER_NAMES):
        result[name] = {"mean": float(flat[:, j].mean()), "sd": float(flat[:, j].std()),
            "median": float(quantiles[2, j]), "ci90": [float(quantiles[1, j]), float(quantiles[3, j])],
            "ci95": [float(quantiles[0, j]), float(quantiles[4, j])],
            "prob_positive": float(np.mean(flat[:, j] > 0)), "prob_negative": float(np.mean(flat[:, j] < 0))}
        if j >= 3:
            result[name]["prob_in_rope"] = float(np.mean(np.abs(flat[:, j]) < rope))
            result[name]["rope_log_scale"] = float(rope)
            result[name]["contact_ratio_ci90"] = np.exp(quantiles[[1, 3], j]).tolist()
    return result


def fit(data: dict, *, draws: int = 1500, warmup: int = 1000, chains: int = 4,
        seed: int = 0, low=None, high=None, fixed_effects=None, progress: bool = False,
        initial_theta=None, rhat_threshold: float = 1.01, min_ess: int = 400) -> dict[str, Any]:
    """Sample the conditional-covariate hierarchical posterior by Metropolis.

    A mixture of preconditioned symmetric random-walk and multivariate Student-t
    independence proposals is used, with the exact proposal-density correction.
    A Laplace covariance only preconditions proposals; it is NOT the reported
    posterior. Random-walk scale/covariance adaptation stops at warmup. Chains
    use independent RNG streams. The population intercept also receives an exact
    conditional Gibbs refresh with cell baselines held fixed.

    ``fixed_effects=(0,0,0)`` fits a target-independent model. Model evidences are
    not computed; effect posterior sign probabilities are not Bayes factors.
    The known localization and between-cell SDs are benchmark assumptions.
    """
    if draws < 10 or warmup < 0 or chains < 2:
        raise ValueError("use draws>=10, warmup>=0 and at least 2 independent chains")
    if not 1 < rhat_threshold < 2 or min_ess < 1:
        raise ValueError("diagnostic thresholds require 1<rhat_threshold<2 and min_ess>=1")
    start = perf_counter()
    prepared = prepare_data(data)
    n = prepared["n_cells"]
    low = PRIOR_LOW.copy() if low is None else np.asarray(low, dtype=float).copy()
    high = PRIOR_HIGH.copy() if high is None else np.asarray(high, dtype=float).copy()
    if low.shape != (6,) or high.shape != (6,) or not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)) or np.any(low >= high):
        raise ValueError("low/high must define finite nonempty six-dimensional bounds")
    fixed = None if fixed_effects is None else np.asarray(fixed_effects, dtype=float)
    if fixed is not None and (fixed.shape != (3,) or not np.all(np.isfinite(fixed)) or np.any(fixed <= low[3:]) or np.any(fixed >= high[3:])):
        raise ValueError("fixed_effects must contain 3 values strictly inside prior bounds")
    active = np.arange(6+n) if fixed is None else np.r_[np.arange(3), np.arange(6, 6+n)]
    dim = len(active)
    full0 = np.r_[0.5*(low+high), np.zeros(n)]
    if initial_theta is not None:
        candidate = np.asarray(initial_theta, dtype=float)
        if candidate.shape != (6,) or not np.all(np.isfinite(candidate)) or np.any(candidate <= low) or np.any(candidate >= high):
            raise ValueError("initial_theta must be strictly inside prior bounds")
        full0[:6] = candidate
    if fixed is not None:
        full0[3:6] = fixed
    obs, gates, dt = prepared["positions"], prepared["gate"], prepared["intervals"]
    loc, hetero = prepared["localization_sd"], prepared["heterogeneity_sd"]

    def unpack(x):
        out = full0.copy()
        out[active] = x
        return out

    def objective(x, enforce_bounds=True):
        full = unpack(x)
        if not np.all(np.isfinite(full)):
            return np.inf
        if enforce_bounds and (np.any(full[:6] <= low) or np.any(full[:6] >= high)):
            return np.inf
        with np.errstate(over="ignore", invalid="ignore"):
            value = -_all_loglik(obs, gates, dt, full[:6], full[6:], loc) + 0.5*np.sum((full[6:]/hetero)**2)
        return float(value) if np.isfinite(value) else np.inf

    bounds = [(low[j]+1e-7, high[j]-1e-7) if j < 6 else (None, None) for j in active]
    optimized = minimize(objective, full0[active], method="L-BFGS-B", bounds=bounds,
                         options={"maxiter": 800, "ftol": 1e-10, "gtol": 1e-5})
    mode = optimized.x
    hessian = _hessian(lambda x: objective(x, False), mode)
    eigenvalues, eigenvectors = np.linalg.eigh(0.5*(hessian+hessian.T))
    covariance = (eigenvectors*(1/np.maximum(eigenvalues, 1e-3)))@eigenvectors.T
    # Limit proposal sizes for weakly identified/boundary dimensions; no effect
    # on the actual posterior density or MH acceptance calculation.
    max_sd = np.r_[0.5*(high-low), np.full(n, 2*hetero)][active]
    factors = np.minimum(1.0, max_sd/np.sqrt(np.diag(covariance)))
    covariance = factors[:, None]*covariance*factors[None, :]
    covariance += 1e-9*np.eye(dim)
    chol = np.linalg.cholesky(covariance)
    inv_chol = solve_triangular(chol, np.eye(dim), lower=True)
    df = 7.0
    independence_scale = 1.3

    def log_proposal(x):
        z = inv_chol@(x-mode)/independence_scale
        return -0.5*(df+dim)*np.log1p(z@z/df)

    output = np.empty((chains, draws, 6+n))
    logps = np.empty((chains, draws))
    acceptance = []
    seeds = np.random.SeedSequence(seed).spawn(chains)
    for chain_id in range(chains):
        rng = np.random.default_rng(seeds[chain_id])
        x = mode.copy()
        for _ in range(1000):
            candidate = mode + chol@rng.normal(size=dim)*0.6
            candidate_obj = objective(candidate)
            if np.isfinite(candidate_obj):
                x = candidate
                break
        lp = -objective(x)
        rw_chol = chol.copy()
        rw_scale = 2.38/np.sqrt(dim)
        history = np.empty((warmup, dim))
        rw_attempts = rw_accepted = ind_attempts = ind_accepted = 0
        window_attempts = window_accepted = 0
        for iteration in range(warmup+draws):
            independent = rng.random() < 0.3
            if independent:
                proposal = mode + independence_scale*(chol@rng.normal(size=dim))/np.sqrt(rng.chisquare(df)/df)
                correction = log_proposal(x)-log_proposal(proposal)
            else:
                proposal = x + rw_scale*(rw_chol@rng.normal(size=dim))
                correction = 0.0
                window_attempts += 1
            proposed_lp = -objective(proposal)
            accepted = np.log(rng.random()) < proposed_lp-lp+correction
            if accepted:
                x, lp = proposal, proposed_lp
                if not independent:
                    window_accepted += 1

            # Exact conditional refresh: cell baseline = population mean + offset.
            # This updates the global log-speed and every offset together while
            # preserving the likelihood and improving hierarchy mixing.
            full = unpack(x)
            baselines = full[0]+full[6:]
            conditional_mean = baselines.mean()
            conditional_sd = hetero/np.sqrt(n)
            a = ndtr((low[0]-conditional_mean)/conditional_sd)
            b = ndtr((high[0]-conditional_mean)/conditional_sd)
            if b-a > 1e-14:
                p = np.clip(a+(b-a)*rng.random(), 1e-15, 1-1e-15)
                new_mean = conditional_mean+conditional_sd*ndtri(p)
                full[0] = new_mean
                full[6:] = baselines-new_mean
                x = full[active]
                lp = -objective(x)

            if iteration < warmup:
                history[iteration] = x
                if (iteration+1) % 50 == 0 and window_attempts:
                    # Diminishing scale adaptation, never used after warmup.
                    learning = min(0.5, 2.0/np.sqrt((iteration+1)/50))
                    rw_scale *= np.exp(learning*(window_accepted/window_attempts-0.234))
                    window_attempts = window_accepted = 0
                if iteration >= max(100, 4*dim) and (iteration+1) % 100 == 0:
                    empirical = np.cov(history[:iteration+1].T)
                    adapted = 0.2*covariance + 0.8*empirical + 1e-9*np.eye(dim)
                    rw_chol = np.linalg.cholesky(adapted)
            else:
                index = iteration-warmup
                output[chain_id, index] = unpack(x)
                logps[chain_id, index] = lp
                if independent:
                    ind_attempts += 1; ind_accepted += int(accepted)
                else:
                    rw_attempts += 1; rw_accepted += int(accepted)
        acceptance.append({"random_walk": rw_accepted/max(rw_attempts, 1),
                           "independence": ind_accepted/max(ind_attempts, 1),
                           "random_walk_scale": float(rw_scale)})
        if progress:
            print(f"state-space chain {chain_id+1}/{chains} complete", flush=True)

    import arviz as az
    sample_dict = {name: output[:, :, j] for j, name in enumerate(PARAMETER_NAMES) if fixed is None or j < 3}
    sample_dict["cell_offset"] = output[:, :, 6:]
    idata = az.from_dict(posterior=sample_dict)
    rhats = az.rhat(idata, method="rank")
    ess_bulk = az.ess(idata, method="bulk")
    ess_tail = az.ess(idata, method="tail")
    diag = {"rhat": {}, "ess_bulk": {}, "ess_tail": {}, "acceptance": acceptance,
            "criteria": {"max_rhat": float(rhat_threshold), "min_ess_bulk": int(min_ess), "min_ess_tail": int(min_ess)}}
    for name in sample_dict:
        diag["rhat"][name] = np.asarray(rhats[name]).tolist()
        diag["ess_bulk"][name] = np.asarray(ess_bulk[name]).tolist()
        diag["ess_tail"][name] = np.asarray(ess_tail[name]).tolist()
    diag["max_rhat"] = float(max(np.max(rhats[name]) for name in sample_dict))
    diag["min_ess_bulk"] = float(min(np.min(ess_bulk[name]) for name in sample_dict))
    diag["min_ess_tail"] = float(min(np.min(ess_tail[name]) for name in sample_dict))
    diag["converged"] = bool(np.isfinite(diag["max_rhat"]) and diag["max_rhat"] < rhat_threshold and diag["min_ess_bulk"] >= min_ess and diag["min_ess_tail"] >= min_ess)
    diag["map_success"] = bool(optimized.success)
    diag["map_message"] = str(optimized.message)
    diag["hessian_min_eigenvalue"] = float(eigenvalues.min())
    diag["observation_model_limitations"] = prepared["limitations"]
    diag["exposure"] = {"gate_min": float(gates.min()), "gate_max": float(gates.max()),
                        "fraction_gate_above_half": float(np.mean(gates > 0.5)),
                        "fraction_gate_below_one_fifth": float(np.mean(gates < 0.2))}
    samples = output[:, :, :6]
    return {"method": "hierarchical_conditional_covariate_kalman_metropolis", "samples": samples,
            "cell_samples": output[:, :, 6:], "log_posterior": logps,
            "summary": summarize_samples(samples), "diagnostics": diag,
            "elapsed_seconds": float(perf_counter()-start), "seed": int(seed),
            "warmup": int(warmup), "draws": int(draws), "chains": int(chains),
            "fixed_effects": None if fixed is None else fixed.tolist(),
            "prior_low": low, "prior_high": high, "known_heterogeneity_sd": hetero,
            "known_localization_sd": loc, "map": unpack(mode)}


def predict_parameter_paths(data: dict, result: dict, *, max_samples: int = 1000, seed: int = 0) -> dict:
    """Posterior intervals for covariate-driven local parameters (not free paths).

    Returned [cell, frame] summaries condition on observed distance gates. They
    express parameter uncertainty, not a prediction interval for noisy speed.
    """
    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    prepared = prepare_data(data)
    theta = np.asarray(result["samples"]).reshape(-1, 6)
    offsets = np.asarray(result["cell_samples"]).reshape(-1, prepared["n_cells"])
    if offsets.shape[0] != theta.shape[0]:
        raise ValueError("global and cell posterior draws must align")
    rng = np.random.default_rng(seed)
    take = rng.choice(len(theta), size=min(max_samples, len(theta)), replace=False)
    theta, offsets = theta[take], offsets[take]
    gate = prepared["gate"][None]
    paths = {}
    for name, baseline, effect in (("speed", 0, 3), ("tau", 1, 4), ("jitter", 2, 5)):
        log_values = theta[:, baseline, None, None] + theta[:, effect, None, None]*gate
        if name == "speed":
            log_values = log_values+offsets[:, :, None]
        quantiles = np.quantile(np.exp(log_values), [0.05, 0.5, 0.95], axis=0)
        paths[name] = {"lower": quantiles[0], "median": quantiles[1], "upper": quantiles[2]}
    paths["gate"] = prepared["gate"]
    paths["times"] = prepared["times"]
    return paths
