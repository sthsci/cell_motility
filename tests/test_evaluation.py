import numpy as np
from motility_inference.evaluation import aggregate_metrics, posterior_metrics, wilson_interval


def test_dataset_not_frame_denominator():
    rng = np.random.default_rng(51)
    record = posterior_metrics(rng.normal(size=(4, 300, 6)), np.zeros(6))
    aggregate = aggregate_metrics([record] * 3)
    assert aggregate["n"] == 3
    assert aggregate["coverage90"] == [1.] * 6
    assert aggregate["coverage90_wilson"][0][0] < .5


def test_zero_events_still_uncertain():
    low, high = wilson_interval(0, 20)
    assert abs(low) < 1e-12
    assert .1 < high < .2


def test_directional_effects_and_equivalence_are_distinct():
    samples = np.zeros((500, 6))
    null = posterior_metrics(samples, np.zeros(6))
    assert not null["any_detected"]
    assert null["prob_equivalent"] == [1., 1., 1.]
    samples[:, 3:] = [-.8, -.8, .8]
    positive = posterior_metrics(samples, samples[0])
    assert positive["detected"] == [True, True, True]
