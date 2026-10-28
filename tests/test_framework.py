"""Safeguards against applying a trained posterior outside its observation model."""
from dataclasses import asdict, replace
import json
import numpy as np
import pytest

from motility_inference.framework import (
    validate_observations, effect_probabilities, infer_neural, infer_state_space,
)
from motility_inference.simulator import SimConfig, simulate
from motility_inference.priors import BASE_THETA


def observed_data(config=None):
    config = SimConfig(n_killers=3, n_targets=3, n_frames=40) if config is None else config
    generated = simulate(BASE_THETA, config, seed=945)
    # Public wrapper must work without any ground truth.
    result = {key: generated[key] for key in ("positions", "targets", "times", "config")}
    # Deliberately provide observed near/far variation for the interface tests;
    # posterior inference is mocked, so these are not recovery experiments.
    result["positions"][:] = result["targets"][0][None]
    result["positions"][:, :, 0] += 12+np.linspace(0, 30, config.n_frames)[None]
    return result


class FakePosterior:
    warn = False
    def input_diagnostics(self, summary):
        return {"warning": self.warn, "maximum_absolute_standardized_summary": 7.0 if self.warn else 1.0}
    def sample(self, summary, n=1000, seed=0):
        result = np.tile(BASE_THETA, (n, 1))
        result[:, 3:] = [-0.8, -0.8, 0.8]
        return result


@pytest.fixture
def posterior(monkeypatch):
    from motility_inference.sbi import NeuralPosterior
    model = FakePosterior()
    monkeypatch.setattr(NeuralPosterior, "load", classmethod(lambda cls, path: model))
    return model


def test_neural_noise_shift_requires_explicit_opt_in(posterior):
    baseline = SimConfig(n_killers=3, n_targets=3, n_frames=40)
    data = observed_data(replace(baseline, localization_sd=0.6))
    with pytest.raises(ValueError, match="localization_sd"):
        infer_neural(data, "unused.pt", training_config=baseline)
    result = infer_neural(data, "unused.pt", training_config=baseline, allow_protocol_shift=True)
    assert result["decision"] == "protocol_shift_unvalidated"
    assert result["abstained"] and result["supported_effects"] == []
    assert len(result["candidate_effects"]) == 3


def test_actual_arrays_override_untrusted_config_claims(posterior):
    data = observed_data()
    data["positions"] = data["positions"][:2]
    with pytest.raises(ValueError, match="disagrees with actual arrays"):
        infer_neural(data, "unused.pt", training_config=data["config"], allow_protocol_shift=True)
    data = observed_data()
    data["times"] *= 2
    with pytest.raises(ValueError, match="timestamps disagree"):
        validate_observations(data)


def test_protocol_file_and_advisory_are_truth_free(tmp_path, posterior):
    data = observed_data()
    (tmp_path/"bank_metadata.json").write_text(json.dumps({"config": data["config"]}))
    first = infer_neural(data, tmp_path/"posterior.pt")
    data.update(theta=np.full(6, np.nan), true_positions=np.zeros_like(data["positions"]),
                true_gate=np.full((3, 40), 999), cell_offsets=np.full(3, 999))
    second = infer_neural(data, tmp_path/"posterior.pt")
    assert first["decision"] == "advisory_effect"
    assert first["effect_probabilities"] == second["effect_probabilities"]
    assert first["supported_effects"] == []


def test_missingness_rejected_or_requires_stress_opt_in(posterior):
    data = observed_data()
    data["positions"][0, :9] = np.nan
    with pytest.raises(ValueError, match="exceeds 20%"):
        validate_observations(data)
    data = observed_data()
    data["positions"][0, 5:7] = np.nan
    with pytest.raises(ValueError, match="missing_observations"):
        infer_neural(data, "unused.pt", training_config=data["config"])
    result = infer_neural(data, "unused.pt", training_config=data["config"], allow_protocol_shift=True)
    assert result["decision"] == "protocol_shift_unvalidated"
    data["positions"][0, 2:] = np.nan
    with pytest.raises(ValueError, match="at least 3 finite"):
        validate_observations(data)


def test_summary_warning_abstains_even_under_matching_protocol(posterior):
    data = observed_data()
    posterior.warn = True
    result = infer_neural(data, "unused.pt", training_config=data["config"])
    assert result["decision"] == "summary_support_caution"
    assert result["abstained"]


def test_state_space_only_supports_after_gate_and_no_known_omission(monkeypatch):
    from motility_inference import state_space
    def fake_fit(data, **kwargs):
        return {"samples": FakePosterior().sample(None, 500).reshape(2, 250, 6),
                "diagnostics": {"converged": kwargs.get("passed", True)}}
    monkeypatch.setattr(state_space, "fit", fake_fit)
    data = observed_data()
    assert infer_state_space(data)["decision"] == "supported_effect"
    assert infer_state_space(data, passed=False)["decision"] == "inconclusive"
    # A correctly supplied changed measurement noise scale is allowed for SSM.
    data["config"]["localization_sd"] = 0.6
    assert infer_state_space(data)["decision"] == "supported_effect"
    data["config"]["spontaneous_sd"] = 0.5
    result = infer_state_space(data)
    assert result["decision"] == "inconclusive"
    assert result["candidate_effects"] and not result["supported_effects"]
    assert "spontaneous_sd" in result["diagnostics"]["known_omitted_components"]


def test_effects_do_not_require_a_truth_vector():
    samples = np.tile(BASE_THETA, (500, 1))
    result = effect_probabilities(samples)
    assert all(value["prob_in_rope"] == 1 for value in result.values())
    assert not any(value["exceeds_posterior_threshold"] for value in result.values())


def test_independent_generator_is_an_explicit_shift(posterior, monkeypatch):
    from motility_inference import state_space
    data = observed_data()
    data["generator"] = "mantishrimp_euler_polarity"
    data["reference_substeps"] = 4
    with pytest.raises(ValueError, match="generator"):
        infer_neural(data, "unused.pt", training_config=data["config"])
    result = infer_neural(data, "unused.pt", training_config=data["config"], allow_protocol_shift=True)
    assert result["decision"] == "protocol_shift_unvalidated"
    monkeypatch.setattr(state_space, "fit", lambda *a, **k: {
        "samples": FakePosterior().sample(None, 100), "diagnostics": {"converged": True}})
    result = infer_state_space(data)
    assert result["decision"] == "inconclusive"
    assert result["diagnostics"]["known_generator_mismatch"]


def test_absent_near_far_contrast_abstains(posterior, monkeypatch):
    from motility_inference import state_space
    data = observed_data()
    data["positions"][:] = data["targets"][0][None]
    data["positions"][:, :, 0] += 12.5
    result = infer_neural(data, "unused.pt", training_config=data["config"])
    assert result["decision"] == "insufficient_proximity_contrast"
    monkeypatch.setattr(state_space, "fit", lambda *a, **k: {
        "samples": FakePosterior().sample(None, 100), "diagnostics": {"converged": True}})
    result = infer_state_space(data)
    assert result["decision"] == "inconclusive"
    assert result["diagnostics"]["exposure"]["warning"]
