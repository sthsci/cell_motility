"""Regression and independent-generator checks for the supplied ABM extension."""

from dataclasses import replace
import numpy as np
import pytest

from mantishrimp.config import CytotoxicityConfig, DomainConfig, MechanicsConfig, MotilityConfig, SimulationConfig
from mantishrimp.simulation import Simulation
from motility_inference.reference import simulate_reference
from motility_inference.simulator import SimConfig


def test_zero_effects_preserve_original_seeded_reference():
    # Recorded before introducing proximity effects: includes adaptive steps,
    # force feedback, synapse formation, damage, and exhaustion.
    cfg = SimulationConfig(n_killers=1, n_targets=1, duration=.2, max_dt=.04,
                           seed=71, domain=DomainConfig(0, 240, 0, 240))
    result = Simulation(cfg).run(initial_killer_positions=np.array([[40., 40.]]),
                                 initial_target_positions=np.array([[62., 40.]]))
    expected = [[38.00071000691171, 39.64377923703114, -.2267583125495467, -.0779376251885904, .9841133570492872],
                [60.70843016480245, 40.08487753702917, .0008406927198313773, .007849558152065384, 1.0]]
    np.testing.assert_allclose(result.final_cells[["x", "y", "polarity_x", "polarity_y", "state"]], expected, rtol=1e-12, atol=1e-12)
    assert list(result.events["event"]) == ["contact_started", "synapse_formed"]
    assert result.snapshots["step"].max() == 11


def test_signed_effect_validation():
    MotilityConfig(killer_delta_speed=-1, killer_delta_tau=-2, killer_delta_jitter=-.5).validate()
    with pytest.raises(ValueError):
        MotilityConfig(target_interaction_length=0).validate()
    with pytest.raises(ValueError):
        MotilityConfig(killer_delta_tau=np.nan).validate()


def test_large_near_target_jitter_increases_step_variance():
    # No active motion or force: same random draws isolate the new Brownian
    # proximity response. A very long interaction length holds gate near one.
    base = SimulationConfig(
        n_killers=1, n_targets=1, duration=1, max_dt=.005, seed=82,
        max_drift_displacement=1e6, domain=DomainConfig(0, 240, 0, 240),
        motility=MotilityConfig(killer_speed=0, target_speed=0,
                               killer_polarity_noise=0, target_polarity_noise=0,
                               killer_translation_noise=.5, target_translation_noise=0,
                               target_interaction_length=1e6),
        mechanics=MechanicsConfig(repulsion_kk=0,repulsion_tt=0,repulsion_kt=0,
                                  adhesion_kt=0,binding_rate=0),
        cytotoxicity=CytotoxicityConfig(killing_rate_max=0),
    )
    def steps(config):
        result = Simulation(config).run(initial_killer_positions=np.array([[40.,40.]]),
                                        initial_target_positions=np.array([[62.,40.]]))
        return np.diff(result.snapshots.query("cell_type == 'killer'")[["x", "y"]].to_numpy(), axis=0)
    null = steps(base)
    changed = steps(replace(base, motility=replace(base.motility, killer_delta_jitter=np.log(3))))
    assert np.mean(changed**2) / np.mean(null**2) > 8.9


def test_bridge_has_expected_grid_and_hierarchy():
    theta = np.array([np.log(3), np.log(2), np.log(.5), -.5, -.7, .6])
    cfg = SimConfig(n_killers=4,n_targets=3,n_frames=13,heterogeneity_sd=.25)
    data = simulate_reference(theta, cfg, seed=36)
    assert data["positions"].shape == (4,13,2)
    assert data["targets"].shape == (3,13,2)
    np.testing.assert_allclose(np.diff(data["times"]),cfg.frame_dt)
    assert np.std(data["cell_offsets"]) > 0
    ref = data["reference_config"]
    motility = ref["motility"]
    mapped_sd = motility["killer_speed"] * motility["killer_polarity_noise"] / np.sqrt(2*motility["killer_polarity_decay"])
    np.testing.assert_allclose(mapped_sd,np.exp(theta[0]))
    assert ref["cytotoxicity"]["killing_rate_max"] == 0
    assert np.isfinite(data["positions"]).all()


def test_invalid_speed_factors_rejected():
    cfg = SimulationConfig(n_killers=1,n_targets=1,duration=.01)
    with pytest.raises(ValueError):
        Simulation(cfg).run(killer_speed_factors=np.array([-1.]))
