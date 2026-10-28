"""Scientific and numerical checks for the reduced simulator."""

from dataclasses import replace
import numpy as np
import pytest

from motility_inference.simulator import SimConfig, _joint_ou, from_mantishrimp_config, load_dataset, proximity, save_dataset, simulate


THETA = np.array([np.log(3.0), np.log(2.0), np.log(0.5), -0.7, -0.6, 0.8])


def test_reference_engine_equivalence_with_all_mechanisms():
    cfg = SimConfig(n_killers=4, n_targets=3, n_frames=18, substeps=2,
                    target_speed=0.4, repulsion=2.0, binding_rate=3.0,
                    adhesion=1.0, spontaneous_sd=0.2)
    a = simulate(THETA, cfg, seed=100)
    b = simulate(THETA, cfg, seed=100, engine="numpy")
    for key in ("positions", "targets", "true_velocity", "true_gate", "true_spontaneous", "true_bound_count"):
        np.testing.assert_allclose(a[key], b[key], rtol=2e-12, atol=2e-12)


def test_reproducible_without_mutating_inputs():
    theta = THETA.copy()
    cfg = SimConfig(n_killers=2, n_targets=1, n_frames=10)
    x = np.array([[119.0, 40.0], [10.0, 10.0]])
    targets = np.array([[60.0, 60.0]])
    offsets = np.array([0.1, -0.1])
    original = (theta.copy(), x.copy(), targets.copy(), offsets.copy())
    a = simulate(theta, cfg, 5, initial_positions=x, initial_targets=targets, cell_offsets=offsets)
    b = simulate(theta, cfg, 5, initial_positions=x, initial_targets=targets, cell_offsets=offsets)
    for arr, saved in zip((theta, x, targets, offsets), original):
        np.testing.assert_array_equal(arr, saved)
    np.testing.assert_array_equal(a["positions"], b["positions"])


def test_periodic_geometry_and_missing_observation():
    x = np.array([[[119., 10.], [121., 10.], [np.nan, 10.]]])
    q = np.array([[[1., 10.], [1., 10.], [1., 10.]]])
    gate, gap, direction, nearest = proximity(x, q, 120, 1, 10)
    np.testing.assert_allclose(gap[0, :2], [1, -1])
    np.testing.assert_allclose(direction[0, :2], [[1, 0], [0, 0]])
    assert np.isnan(gate[0, 2]) and nearest[0, 2] == -1


def test_unwrapped_tracks_are_continuous():
    cfg = SimConfig(n_killers=1, n_targets=1, n_frames=100, localization_sd=0,
                    heterogeneity_sd=0, box_size=30, contact_distance=3)
    theta = np.array([np.log(5), np.log(3), np.log(0.2), 0, 0, 0])
    data = simulate(theta, cfg, 101, initial_positions=np.array([[29., 15.]]),
                    initial_targets=np.array([[15., 15.]]))
    assert np.max(np.abs(np.diff(data["positions"], axis=1))) < 15
    assert np.any((data["positions"] < 0) | (data["positions"] >= cfg.box_size))


def test_joint_ou_moments_and_stationary_velocity():
    # This independently tests the exact transition against analytic moments.
    rng = np.random.default_rng(42)
    n = 100000
    tau, speed, jitter, dt = 2.0, 3.0, 0.5, 0.25
    phi, integral, lvv, lxv, lxx = _joint_ou(tau, speed, jitter, dt)
    z = rng.normal(size=(n, 3))
    initial_v = speed * z[:, 0]
    new_v = phi * initial_v + lvv * z[:, 1]
    displacement = integral * initial_v + lxv * z[:, 1] + lxx * z[:, 2]
    expected_var = 2 * speed**2 * tau * (dt - tau * (1 - np.exp(-dt / tau))) + jitter**2 * dt
    assert np.isclose(np.var(new_v), speed**2, rtol=0.02)
    assert np.isclose(np.var(displacement), expected_var, rtol=0.02)
    assert np.isclose(np.cov(new_v, initial_v)[0, 1], speed**2 * phi, rtol=0.02)
    assert np.isclose(np.cov(new_v, displacement)[0, 1], speed**2 * tau * (1-phi), rtol=0.02)


def test_observed_gate_uses_measurements_and_roundtrip(tmp_path):
    data = simulate(THETA, SimConfig(n_killers=3, n_targets=2, n_frames=20), 7)
    cfg = data["config"]
    gate = proximity(data["positions"], data["targets"], cfg["box_size"], cfg["contact_distance"], cfg["interaction_length"])[0]
    assert not np.array_equal(gate, data["true_gate"])
    path = save_dataset(data, tmp_path / "movie.npz", metadata={"purpose": "test"})
    loaded = load_dataset(path)
    np.testing.assert_array_equal(loaded["positions"], data["positions"])
    assert loaded["config"] == data["config"]
    assert loaded["metadata"]["purpose"] == "test"


def test_default_design_contains_near_and_far_observations():
    theta = THETA.copy()
    theta[3:] = 0
    for seed in range(601, 606):
        data = simulate(theta, seed=seed)
        gate = data["true_gate"]
        assert np.mean(gate > 0.5) > 0.05
        assert np.mean(gate < 0.2) > 0.05


def test_reference_abm_mapping():
    from mantishrimp.config import SimulationConfig, MotilityConfig
    original = SimulationConfig(duration=1.03, max_dt=0.1,
                                motility=MotilityConfig(target_translation_noise=0))
    theta, reduced = from_mantishrimp_config(original)
    np.testing.assert_allclose(np.exp(theta[:3]), [150 * .75 / np.sqrt(5), 1 / 2.5, .75])
    assert reduced.contact_distance == 22
    assert np.isclose((reduced.n_frames - 1) * reduced.frame_dt, 1.03)
    assert reduced.frame_dt <= original.max_dt
    assert reduced.adhesion == original.mechanics.adhesion_kt


@pytest.mark.parametrize("changes", [{"frame_dt": 0}, {"substeps": 0}, {"localization_sd": -1}, {"n_killers": 1.5}, {"contact_distance": 1000}])
def test_invalid_design_rejected(changes):
    with pytest.raises(ValueError):
        simulate(THETA, replace(SimConfig(), **changes))
