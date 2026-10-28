"""Likelihood-free migration inference: prior-bank ABC and neural posterior.

The neural estimator is a custom PyTorch Gaussian-mixture density network, not
the third-party ``sbi`` package. Its likelihood is over parameters conditional
on observed trajectory summaries; no analytical trajectory likelihood is used.
Both methods approximate p(theta | summaries), not necessarily p(theta | data).
Independent dataset-level validation is essential before scientific use.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.special import expit, logit

from .priors import PRIOR_LOW, PRIOR_HIGH, PARAMETER_NAMES
from .summaries import summarize


def generate_bank(n: int, config=None, seed: int = 0, *, progress: bool = False,
                  theta: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Generate independent datasets from the prior (or supplied parameters).

    One row is one complete simulated movie. Parameter draws and trajectory
    seeds use independent RNG streams. The per-cell baseline effects are drawn
    afresh inside the simulator, hence marginalized by this bank.
    """
    from .simulator import SimConfig, simulate
    if n < 1:
        raise ValueError("n must be positive")
    config = SimConfig() if config is None else config
    theta_seed, sim_seed = np.random.SeedSequence(seed).spawn(2)
    rng_theta = np.random.default_rng(theta_seed)
    rng_sim = np.random.default_rng(sim_seed)
    if theta is None:
        theta = rng_theta.uniform(PRIOR_LOW, PRIOR_HIGH, size=(n, len(PRIOR_LOW)))
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (n, len(PRIOR_LOW)) or not np.isfinite(theta).all():
        raise ValueError("theta must be finite with shape (n, 6)")
    summaries = []
    start = time.perf_counter()
    for i in range(n):
        data = simulate(theta[i], config=config, seed=int(rng_sim.integers(0, 2**31 - 1)))
        summaries.append(summarize(data))
        if progress and ((i + 1) % max(1, n // 10) == 0):
            print(f"Simulation bank {i + 1}/{n}: {time.perf_counter() - start:.1f}s", flush=True)
    return theta, np.asarray(summaries)


def save_bank(path: str | Path, theta: np.ndarray, summaries: np.ndarray, config=None,
              seed: int | None = None) -> None:
    """Portable bank with its observation protocol; no pickle serialization."""
    import json
    meta = {"config": asdict(config) if is_dataclass(config) else config,
            "seed": seed, "parameter_names": PARAMETER_NAMES,
            "prior_low": PRIOR_LOW.tolist(), "prior_high": PRIOR_HIGH.tolist()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, theta=theta, summaries=summaries, metadata=json.dumps(meta))


def load_bank(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    import json
    with np.load(path, allow_pickle=False) as data:
        return data["theta"], data["summaries"], json.loads(str(data["metadata"]))


def abc_samples(summary: np.ndarray, theta: np.ndarray, summaries: np.ndarray,
                n_accept: int = 256, seed: int = 0) -> dict[str, Any]:
    """Empirical prior-bank rejection ABC, with an adaptive k-nearest tolerance.

    Every accepted point is an actual prior draw whose simulated summary is
    close to the observation. No regression adjustment or unsupported exact
    posterior claim is made. The tolerance and accepted fraction are returned
    because broad tolerance can materially bias the approximation.
    """
    theta, summaries, summary = map(np.asarray, (theta, summaries, summary))
    if summaries.ndim != 2 or theta.shape != (len(summaries), 6) or summary.shape != (summaries.shape[1],):
        raise ValueError("incompatible summary or parameter dimensions")
    if not 1 <= n_accept <= len(theta):
        raise ValueError("n_accept must lie between 1 and the bank size")
    if not np.isfinite(summaries).all() or not np.isfinite(summary).all():
        raise ValueError("summaries must be finite")
    scale = np.std(summaries, axis=0)
    active = scale > 1e-8
    distances = np.sqrt(np.mean(((summaries[:, active] - summary[active]) / scale[active]) ** 2, axis=1)) if active.any() else np.zeros(len(theta))
    # Stable ordering makes ties reproducible without inventing extra weights.
    indices = np.argsort(distances, kind="stable")[:n_accept]
    return {"samples": theta[indices].copy(), "indices": indices,
            "distances": distances[indices], "epsilon": float(distances[indices[-1]]),
            "acceptance_fraction": n_accept / len(theta),
            "effective_sample_size": float(n_accept), "method": "prior-bank rejection ABC"}


def _torch():
    import torch
    return torch


def _make_network(summary_dim: int, parameter_dim: int, components: int, hidden: int):
    torch = _torch()
    class MixtureNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(summary_dim, hidden), torch.nn.SiLU(),
                torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
                torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            )
            self.k, self.d = components, parameter_dim
            self.ntril = parameter_dim * (parameter_dim + 1) // 2
            self.output = torch.nn.Linear(hidden, components * (1 + parameter_dim + self.ntril))
            self.register_buffer("tril_rows", torch.tril_indices(parameter_dim, parameter_dim)[0])
            self.register_buffer("tril_cols", torch.tril_indices(parameter_dim, parameter_dim)[1])

        def forward(self, x):
            # Bounded inputs improve numerical behavior under stress, while
            # separate diagnostics still report out-of-training-range inputs.
            raw = self.output(self.net(torch.clamp(x, -12, 12)))
            raw = raw.reshape(-1, self.k, 1 + self.d + self.ntril)
            weights = torch.log_softmax(raw[:, :, 0], dim=-1)
            means = raw[:, :, 1:1 + self.d]
            packed = raw[:, :, 1 + self.d:]
            chol = torch.zeros((*packed.shape[:-1], self.d, self.d), dtype=packed.dtype, device=packed.device)
            chol[:, :, self.tril_rows, self.tril_cols] = packed
            diagonal = torch.nn.functional.softplus(torch.diagonal(chol, dim1=-2, dim2=-1)) + 0.025
            chol = chol - torch.diag_embed(torch.diagonal(chol, dim1=-2, dim2=-1)) + torch.diag_embed(diagonal)
            return weights, means, chol
    return MixtureNetwork()


def _mixture_log_prob(outputs, z):
    torch = _torch()
    log_weights, means, chol = outputs
    delta = z[:, None] - means
    whitened = torch.linalg.solve_triangular(chol, delta[..., None], upper=False)[..., 0]
    component_lp = -0.5 * (whitened.square().sum(-1) + z.shape[-1] * np.log(2 * np.pi)) - torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
    return torch.logsumexp(log_weights + component_lp, dim=-1)


class NeuralPosterior:
    """Conditional mixture posterior with hard prior support and joint draws."""
    def __init__(self, network, summary_mean, summary_scale, parameter_mean, parameter_scale,
                 *, components: int, hidden: int, history: dict | None = None):
        self.network = network.eval()
        self.summary_mean = np.asarray(summary_mean, dtype=float)
        self.summary_scale = np.asarray(summary_scale, dtype=float)
        self.parameter_mean = np.asarray(parameter_mean, dtype=float)
        self.parameter_scale = np.asarray(parameter_scale, dtype=float)
        self.components, self.hidden = components, hidden
        self.history = {} if history is None else history

    def _context(self, summary):
        summary = np.asarray(summary, dtype=float)
        if summary.shape != self.summary_mean.shape or not np.isfinite(summary).all():
            raise ValueError("one finite summary vector of the training dimension is required")
        return (summary - self.summary_mean) / self.summary_scale

    def sample(self, summary: np.ndarray, n: int = 1000, seed: int = 0) -> np.ndarray:
        if n < 1:
            raise ValueError("n must be positive")
        torch = _torch()
        with torch.no_grad():
            weights, means, chols = self.network(torch.as_tensor(self._context(summary)[None], dtype=torch.float32))
        weights, means, chols = weights[0].numpy(), means[0].numpy(), chols[0].numpy()
        probabilities = np.exp(weights.astype(float)); probabilities /= probabilities.sum()
        rng = np.random.default_rng(seed)
        which = rng.choice(self.components, n, p=probabilities)
        z = means[which] + np.einsum("nij,nj->ni", chols[which], rng.normal(size=(n, len(PRIOR_LOW))))
        unscaled = z * self.parameter_scale + self.parameter_mean
        return PRIOR_LOW + (PRIOR_HIGH - PRIOR_LOW) * expit(unscaled)

    def log_prob(self, theta: np.ndarray, summary: np.ndarray) -> np.ndarray:
        """Log density on original theta scale, including transform Jacobian."""
        torch = _torch()
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        if theta.shape[1] != len(PRIOR_LOW):
            raise ValueError("theta must have six columns")
        u = (theta - PRIOR_LOW) / (PRIOR_HIGH - PRIOR_LOW)
        valid = np.isfinite(u).all(1) & (u > 0).all(1) & (u < 1).all(1)
        result = np.full(len(theta), -np.inf)
        if valid.any():
            z = (logit(u[valid]) - self.parameter_mean) / self.parameter_scale
            context = np.broadcast_to(self._context(summary), (valid.sum(), len(self.summary_mean))).copy()
            with torch.no_grad():
                lp = _mixture_log_prob(self.network(torch.as_tensor(context, dtype=torch.float32)), torch.as_tensor(z, dtype=torch.float32)).numpy()
            log_jacobian = -np.log(PRIOR_HIGH - PRIOR_LOW).sum() - np.log(self.parameter_scale).sum() - np.log(u[valid]).sum(1) - np.log1p(-u[valid]).sum(1)
            result[valid] = lp + log_jacobian
        return result

    def input_diagnostics(self, summary: np.ndarray) -> dict:
        z = self._context(summary)
        return {"maximum_absolute_standardized_summary": float(np.max(np.abs(z))),
                "fraction_beyond_4_training_sd": float(np.mean(np.abs(z) > 4)),
                "warning": bool(np.any(np.abs(z) > 6))}

    def save(self, path: str | Path) -> None:
        torch = _torch()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {"state_dict": self.network.state_dict(), "summary_mean": self.summary_mean.tolist(),
                      "summary_scale": self.summary_scale.tolist(), "parameter_mean": self.parameter_mean.tolist(),
                      "parameter_scale": self.parameter_scale.tolist(), "components": self.components,
                      "hidden": self.hidden, "history": self.history, "prior_low": PRIOR_LOW.tolist(),
                      "prior_high": PRIOR_HIGH.tolist(), "format_version": 1}
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str | Path):
        torch = _torch()
        obj = torch.load(path, map_location="cpu", weights_only=True)
        if not np.allclose(obj["prior_low"], PRIOR_LOW) or not np.allclose(obj["prior_high"], PRIOR_HIGH):
            raise ValueError("checkpoint prior differs from current inference prior")
        network = _make_network(len(obj["summary_mean"]), len(PRIOR_LOW), obj["components"], obj["hidden"])
        network.load_state_dict(obj["state_dict"])
        return cls(network, obj["summary_mean"], obj["summary_scale"], obj["parameter_mean"], obj["parameter_scale"],
                   components=obj["components"], hidden=obj["hidden"], history=obj["history"])


def train_posterior(theta: np.ndarray, summaries: np.ndarray, seed: int = 0,
                    max_epochs: int = 300, *, components: int = 3, hidden: int = 128,
                    batch_size: int = 128, validation_fraction: float = 0.15,
                    patience: int = 35, learning_rate: float = 0.0015,
                    progress: bool = False) -> NeuralPosterior:
    """Train on dataset-level splits; restore best held-out likelihood weights.

    The validation split is used only for early stopping. Calibration and all
    scientific performance claims require another independent simulation set.
    Input standardization is fitted on training rows only. Theta is mapped into
    unconstrained logit coordinates; samples map back inside the shared prior.
    """
    torch = _torch()
    theta, summaries = np.asarray(theta, dtype=float), np.asarray(summaries, dtype=float)
    if summaries.ndim != 2 or theta.shape != (len(summaries), 6) or len(theta) < 40:
        raise ValueError("at least 40 paired dataset-level training rows are required")
    if not np.isfinite(theta).all() or not np.isfinite(summaries).all():
        raise ValueError("training data must be finite")
    if not 0 < validation_fraction < 0.5 or max_epochs < 1 or components < 1:
        raise ValueError("invalid training hyperparameters")
    u = (theta - PRIOR_LOW) / (PRIOR_HIGH - PRIOR_LOW)
    if np.any((u <= 0) | (u >= 1)):
        raise ValueError("training parameters must lie strictly inside the shared prior")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(theta))
    n_val = max(10, int(validation_fraction * len(theta)))
    val, train = indices[:n_val], indices[n_val:]
    sm = summaries[train].mean(0)
    ss = np.maximum(summaries[train].std(0), 1e-5)
    z = logit(u)
    zm, zs = z[train].mean(0), np.maximum(z[train].std(0), 0.1)
    x_t = torch.as_tensor((summaries - sm) / ss, dtype=torch.float32)
    z_t = torch.as_tensor((z - zm) / zs, dtype=torch.float32)
    history = {"train_nll": [], "validation_nll": [], "train_indices": train.tolist(),
               "validation_indices": val.tolist(), "seed": seed, "method": "custom bounded MDN posterior",
               "loss_scale": "standardized logit parameters; Jacobian constant w.r.t. network weights"}
    start = time.perf_counter()
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(min(2, old_threads))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            network = _make_network(summaries.shape[1], 6, components, hidden)
            optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate, weight_decay=1e-4)
            best, best_weights, best_epoch = np.inf, None, -1
            for epoch in range(max_epochs):
                network.train(); losses = []
                for start_index in range(0, len(train), batch_size):
                    if start_index == 0:
                        order = rng.permutation(train)
                    batch = order[start_index:start_index + batch_size]
                    loss = -_mixture_log_prob(network(x_t[batch]), z_t[batch]).mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite neural posterior training loss")
                    optimizer.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
                    optimizer.step(); losses.append((float(loss.detach()), len(batch)))
                network.eval()
                with torch.no_grad():
                    vloss = float(-_mixture_log_prob(network(x_t[val]), z_t[val]).mean())
                history["train_nll"].append(sum(v * n for v, n in losses) / len(train))
                history["validation_nll"].append(vloss)
                if vloss < best - 1e-4:
                    best, best_weights, best_epoch = vloss, deepcopy(network.state_dict()), epoch
                if progress and (epoch == 0 or (epoch + 1) % 25 == 0):
                    print(f"MDN epoch {epoch + 1}: held-out NLL {vloss:.3f}, best {best:.3f}", flush=True)
                if epoch - best_epoch >= patience:
                    break
            network.load_state_dict(best_weights)
    finally:
        torch.set_num_threads(old_threads)
    history.update({"best_epoch": best_epoch, "best_validation_nll": best,
                    "epochs": len(history["train_nll"]), "elapsed_seconds": time.perf_counter() - start})
    return NeuralPosterior(network, sm, ss, zm, zs, components=components, hidden=hidden, history=history)


def calibration_metrics(posterior: NeuralPosterior, theta: np.ndarray, summaries: np.ndarray,
                         n_draws: int = 1000, seed: int = 0) -> dict[str, np.ndarray]:
    """Independent prior-predictive SBC ranks and marginal interval coverage.

    Uniform ranks are necessary, not sufficient; an uninformative prior also
    calibrates. Return posterior means and widths to quantify informativeness.
    """
    theta, summaries = np.asarray(theta), np.asarray(summaries)
    ranks, means, lower, upper = [], [], [], []
    for i, (truth, summary) in enumerate(zip(theta, summaries)):
        draws = posterior.sample(summary, n_draws, seed + i)
        ranks.append(np.sum(draws < truth, axis=0))
        means.append(draws.mean(0)); lower.append(np.quantile(draws, 0.05, axis=0)); upper.append(np.quantile(draws, 0.95, axis=0))
    means, lower, upper = map(np.asarray, (means, lower, upper))
    return {"ranks": np.asarray(ranks), "posterior_mean": means, "lower90": lower, "upper90": upper,
            "covered90": (theta >= lower) & (theta <= upper), "rmse": np.sqrt(np.mean((theta - means) ** 2, axis=0)),
            "mean_width90": np.mean(upper - lower, axis=0)}
