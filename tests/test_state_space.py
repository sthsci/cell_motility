"""Numerical likelihood and inference-contract checks, independent of the ABM."""
import numpy as np
import pytest
from scipy.integrate import quad_vec
from scipy.linalg import expm
from scipy.stats import multivariate_normal

from motility_inference.priors import BASE_THETA
from motility_inference.state_space import (
    _cell_loglik, ou_transition, prepare_data, fit, predict_parameter_paths,
)


@pytest.mark.parametrize("dt", [1e-5, 0.25, 4.0])
def test_integrated_ou_covariance_against_matrix_exponential(dt):
    speed, tau, jitter = 3.1, 1.2, 0.7
    drift = np.array([[-1/tau, 0.0], [1.0, 0.0]])
    diffusion = np.diag([2*speed**2/tau, jitter**2])
    expected_f = expm(drift*dt)
    expected_q = quad_vec(lambda s: expm(drift*s)@diffusion@expm(drift.T*s), 0, dt)[0]
    phi, h, qvv, qvx, qxx = ou_transition(speed, tau, jitter, dt)
    np.testing.assert_allclose([[phi, 0], [h, 1]], expected_f, atol=1e-12)
    np.testing.assert_allclose([[qvv, qvx], [qvx, qxx]], expected_q, rtol=1e-9, atol=1e-12)
    assert np.linalg.eigvalsh(expected_q).min() > 0


def test_kalman_matches_joint_gaussian_density():
    theta = BASE_THETA.copy()
    theta[3:] = [-0.8, -0.5, 0.4]
    offset = 0.15
    gate = np.array([0.1, 0.6, 0.8, 0.3, 0.0, 0.2])
    dt = np.array([0.2, 0.4, 0.15, 0.6, 0.25])
    rng = np.random.default_rng(4)
    y = np.cumsum(rng.normal(size=(len(gate), 2)), axis=0)
    sd = 0.2
    n = len(gate)
    # Construct the entire Gaussian covariance, rather than reproducing a filter.
    cov = np.zeros((2*n, 2*n))
    cov[:2, :2] = np.diag([np.exp(2*(theta[0]+offset+theta[3]*gate[0])), sd**2])
    for t in range(n-1):
        speed = np.exp(theta[0]+offset+theta[3]*gate[t])
        tau = np.exp(theta[1]+theta[4]*gate[t])
        jitter = np.exp(theta[2]+theta[5]*gate[t])
        A = np.array([[-1/tau, 0], [1, 0]])
        F = expm(A*dt[t])
        B = np.diag([2*speed**2/tau, jitter**2])
        Q = quad_vec(lambda s: expm(A*s)@B@expm(A.T*s), 0, dt[t])[0]
        current = slice(2*t, 2*t+2)
        future = slice(2*t+2, 2*t+4)
        cov[future, :2*t+2] = F@cov[current, :2*t+2]
        cov[:2*t+2, future] = cov[future, :2*t+2].T
        cov[future, future] = F@cov[current, current]@F.T+Q
    position_indices = np.arange(3, 2*n, 2)
    obs_cov = cov[np.ix_(position_indices, position_indices)] + sd**2*np.eye(n-1)
    expected = sum(multivariate_normal.logpdf(y[1:, axis], mean=np.full(n-1, y[0, axis]), cov=obs_cov) for axis in range(2))
    actual = _cell_loglik(y, gate, dt, theta, offset, sd)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-10)
    # Missing y has the marginal Gaussian density after dropping that coordinate.
    missing = y.copy(); missing[3, 0] = np.nan
    keep = np.arange(n-1) != 2
    expected_missing = multivariate_normal.logpdf(y[1:, 0][keep], mean=np.full(keep.sum(), y[0, 0]), cov=obs_cov[np.ix_(keep, keep)])
    expected_missing += multivariate_normal.logpdf(y[1:, 1], mean=np.full(n-1, y[0, 1]), cov=obs_cov)
    np.testing.assert_allclose(_cell_loglik(missing, gate, dt, theta, offset, sd), expected_missing, rtol=1e-11)


def _tiny_dataset():
    from motility_inference.simulator import SimConfig, simulate
    return simulate(BASE_THETA, SimConfig(n_killers=2, n_targets=2, n_frames=30), seed=314)


def test_truth_does_not_leak_into_covariates():
    data = _tiny_dataset()
    original = prepare_data(data)["gate"]
    for name in ("true_gate", "true_positions", "theta", "cell_offsets", "true_velocity"):
        if name in data:
            data[name] = np.full_like(data[name], 9999.0)
    np.testing.assert_array_equal(prepare_data(data)["gate"], original)


def test_short_chains_are_flagged_and_paths_align():
    data = _tiny_dataset()
    result = fit(data, draws=20, warmup=20, chains=2, seed=4, fixed_effects=[0, 0, 0])
    assert result["samples"].shape == (2, 20, 6)
    assert result["cell_samples"].shape == (2, 20, 2)
    assert not result["diagnostics"]["converged"]
    assert np.all(np.isfinite(result["samples"]))
    np.testing.assert_array_equal(result["samples"][..., 3:], 0)
    paths = predict_parameter_paths(data, result, max_samples=20)
    assert paths["speed"]["median"].shape == (2, 30)
    assert np.all(paths["speed"]["lower"] <= paths["speed"]["median"])
    assert np.all(paths["speed"]["median"] <= paths["speed"]["upper"])


def test_invalid_hierarchy_sd_rejected():
    data = _tiny_dataset()
    data["config"]["heterogeneity_sd"] = 0
    with pytest.raises(ValueError, match="heterogeneity_sd"):
        prepare_data(data)
