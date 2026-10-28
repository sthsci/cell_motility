"""Scientific invariants for observed-only SBI and bounded posterior densities."""
import numpy as np
import pytest

from motility_inference.priors import PRIOR_LOW, PRIOR_HIGH
from motility_inference.summaries import summarize, observed_geometry
from motility_inference.sbi import abc_samples, train_posterior, NeuralPosterior


def example_data(seed=0):
    rng = np.random.default_rng(seed)
    n, t = 4, 50
    x = np.cumsum(rng.normal(size=(n, t, 2)), axis=1)
    targets = np.broadcast_to(np.array([[0., 0.], [30., 40.]])[:, None], (2, t, 2)).copy()
    return {"positions": x, "targets": targets, "times": np.arange(t) * 0.25,
            "config": {"box_size": 180., "contact_distance": 12., "interaction_length": 10.}}


def test_summaries_ignore_all_latent_truth():
    data = example_data()
    base, names = summarize(data, return_names=True)
    data.update(theta=np.full(6, np.nan), true_positions=np.full_like(data["positions"], np.nan),
                true_gate=np.full((4, 50), np.nan), cell_offsets=np.ones(4) * 100,
                true_velocity=np.zeros_like(data["positions"]))
    assert np.array_equal(base, summarize(data))
    assert len(names) == len(base) and len(set(names)) == len(names)
    assert np.isfinite(base).all()


def test_periodic_geometry_and_missing_observations():
    data = example_data()
    original = observed_geometry(data)
    shifted = {**data, "positions": data["positions"] + np.array([180., -180.])}
    assert np.allclose(original["gate"], observed_geometry(shifted)["gate"])
    data["positions"][0, 10:13] = np.nan
    assert np.isfinite(summarize(data)).all()
    data["times"][10] += 0.01
    with pytest.raises(ValueError, match="regular"):
        summarize(data)


def test_abc_returns_real_prior_bank_points():
    rng = np.random.default_rng(4)
    theta = rng.uniform(PRIOR_LOW, PRIOR_HIGH, size=(100, 6))
    summary = theta[:, [0, 2, 4]]
    result = abc_samples(summary[7], theta, summary, n_accept=8)
    assert result["indices"][0] == 7
    assert result["distances"][0] == 0
    assert np.array_equal(result["samples"], theta[result["indices"]])
    assert result["acceptance_fraction"] == 0.08


def test_neural_posterior_learns_and_roundtrips(tmp_path):
    # A simple independent observation model has an identifiable posterior.
    # The test checks learned conditioning, not a mirror of network internals.
    rng = np.random.default_rng(20)
    theta = rng.uniform(PRIOR_LOW, PRIOR_HIGH, size=(800, 6))
    unit = (theta - PRIOR_LOW) / (PRIOR_HIGH - PRIOR_LOW)
    summaries = unit + rng.normal(0, 0.035, unit.shape)
    posterior = train_posterior(theta, summaries, seed=3, max_epochs=65,
                                 components=1, hidden=48, patience=15)
    obs = np.full(6, 0.32)
    samples = posterior.sample(obs, 3000, seed=9)
    mean_u = ((samples - PRIOR_LOW) / (PRIOR_HIGH - PRIOR_LOW)).mean(0)
    assert np.mean((mean_u - obs) ** 2) < np.mean((0.5 - obs) ** 2) / 2
    assert np.all(samples > PRIOR_LOW) and np.all(samples < PRIOR_HIGH)
    assert np.isfinite(posterior.log_prob(samples[:10], obs)).all()
    # Independently verify the mixture density and bounded-transform Jacobian.
    import torch
    from scipy.special import logit, logsumexp
    from scipy.stats import multivariate_normal
    with torch.no_grad():
        logw, means, chols = posterior.network(torch.tensor(posterior._context(obs)[None], dtype=torch.float32))
    test_theta = samples[0]
    test_u = (test_theta - PRIOR_LOW) / (PRIOR_HIGH - PRIOR_LOW)
    test_z = (logit(test_u) - posterior.parameter_mean) / posterior.parameter_scale
    comp = []
    for k in range(posterior.components):
        ch = chols[0, k].numpy().astype(float)
        comp.append(float(logw[0, k]) + multivariate_normal.logpdf(test_z, means[0, k].numpy(), ch @ ch.T))
    jac = -np.log(PRIOR_HIGH - PRIOR_LOW).sum() - np.log(posterior.parameter_scale).sum() - np.log(test_u).sum() - np.log1p(-test_u).sum()
    assert np.allclose(posterior.log_prob(test_theta, obs), logsumexp(comp) + jac, atol=1e-4)
    assert np.isneginf(posterior.log_prob(PRIOR_LOW - 0.1, obs)).all()
    path = tmp_path / "posterior.pt"
    posterior.save(path)
    restored = NeuralPosterior.load(path)
    assert np.array_equal(samples[:100], restored.sample(obs, 3000, seed=9)[:100])
    assert set(posterior.history["train_indices"]).isdisjoint(posterior.history["validation_indices"])
