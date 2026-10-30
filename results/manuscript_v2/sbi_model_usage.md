# Using the manuscript noise-conditioned research checkpoint

The primary noise-conditioned checkpoint is
`results/manuscript_v2/sbi_conditioned_127001.pt`. Its input has **83 values**:
the existing 82 observed-track summaries followed by **log(known localization
standard deviation in µm)**. The noise standard deviation is an acquisition
nuisance supplied to inference. It is not estimated by this network.

The original `framework.infer_neural` wrapper was written for the 82-feature,
fixed-noise checkpoint. **Do not pass the 83-feature checkpoint to that wrapper.**
The example below uses the correct context and makes its synthetic protocol
checks explicit. It is a research example, not an adapter that validates an
experimental imaging protocol.

## Training domain

All models in this family used the following fixed settings, except the stated
localization-noise range. These are assumptions, not measured biological facts.

| Setting | Training value |
|---|---|
| Dimension and boundary | 2D; periodic 180 × 180 µm square |
| Killer trajectories | 12 per independent movie |
| Target trajectories | 9, stationary and observed without noise |
| Recorded frames and interval | 201 frames; 0.25 min/frame; 50 min duration |
| Internal simulator steps | One per recorded frame |
| Contact distance | 12 µm |
| Exponential interaction length | 10 µm |
| Cell random-intercept SD | 0.25 log units, affecting active amplitude only |
| Localization SD | Known; training draws log-uniformly from 0.05 to 0.70 µm |
| Observation error | Independent Gaussian position errors; no motion blur |
| Target initial geometry | Perturbed 3 × 3 lattice; ±9 µm coordinate perturbations |
| Killer initialization | Uniform outside initial target-contact zones |
| Intrinsic velocity initialization | Local stationary Gaussian distribution |
| Mechanical and temporal alternatives | Repulsion, binding, adhesion and spontaneous drift absent |
| Missingness and identities | Complete finite tracks with correct identities |
| Position representation | Unwrapped killer coordinates; minimum-image target distances |
| Target physics | Penetrable proximity zones in this baseline |

The six-dimensional prior is identical to the original inference package:
baseline active SD 2–5 µm/min, persistence 0.8–3 min and jitter amplitude
0.4–1.2 µm/√min, each uniform on its log scale; proximity log-ratio effects
`delta_speed` and `delta_tau` uniform on [−1.2, 0.5], and `delta_jitter` uniform
on [−0.5, 1.2]. Population heterogeneity and localization error are not extra
parameters in the six-dimensional posterior.

The marginalized ablation `sbi_marginalized_127001.pt` was trained on the same
varying-noise bank but uses 82 features. It integrates over the training noise
distribution rather than conditioning on a supplied noise value. That is a
different assumption, even when a particular prediction looks similar.

## Executable synthetic example

Run from the repository root after installing the project. The example creates
a synthetic high-noise dataset, discards all latent truth before inference,
checks the declared protocol and observable array structure, then reports
conditional posterior probabilities. It does not create a biological claim.

```python
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np

from motility_inference.simulator import SimConfig, simulate
from motility_inference.priors import scenario_theta
from motility_inference.summaries import summarize
from motility_inference.sbi import NeuralPosterior

root = Path.cwd()
checkpoint = root / "results/manuscript_v2/sbi_conditioned_127001.pt"

# Simulation is only a demonstration source, not part of inference.
generated = simulate(
    scenario_theta("combined"),
    replace(SimConfig(), localization_sd=0.60),
    seed=199910001,
)
observed = {key: generated[key] for key in
            ("positions", "targets", "times", "config")}
del generated  # Inference below has no generating parameters or latent paths.

# Declared-protocol checks: metadata cannot establish that an omitted physical
# process is actually absent from experimental observations.
expected = asdict(SimConfig())
cfg = observed["config"]
missing = sorted(set(expected) - set(cfg))
if missing:
    raise ValueError(f"Missing protocol fields: {missing}")
for key, value in expected.items():
    if key != "localization_sd" and not np.isclose(
        float(cfg[key]), float(value), rtol=0, atol=1e-12
    ):
        raise ValueError(f"Unsupported protocol setting: {key}")
noise_sd = float(cfg["localization_sd"])
if not np.isfinite(noise_sd) or not 0.05 <= noise_sd <= 0.70:
    raise ValueError("Known localization SD lies outside the trained range")

positions = np.asarray(observed["positions"])
targets = np.asarray(observed["targets"])
times = np.asarray(observed["times"])
if positions.shape != (12, 201, 2) or targets.shape != (9, 201, 2):
    raise ValueError("Unsupported track count, frame count or dimension")
if times.shape != (201,) or not np.allclose(
    times - times[0], np.arange(201) * 0.25, rtol=0, atol=1e-10
):
    raise ValueError("Unsupported observation times")
if not all(np.isfinite(x).all() for x in (positions, targets, times)):
    raise ValueError("Missing/non-finite measurements were not validated here")
if not np.allclose(targets, targets[:, :1, :], rtol=0, atol=1e-10):
    raise ValueError("Moving targets are outside this checkpoint's training")

context = np.r_[summarize(observed), np.log(noise_sd)]
posterior = NeuralPosterior.load(checkpoint)
if len(posterior.summary_mean) != 83:
    raise ValueError("This example requires the conditioned 83-feature model")
draws = posterior.sample(context, n=1500, seed=199910002)
threshold = np.log(1.2)
probabilities = {
    "lower_active_amplitude": float(np.mean(draws[:, 3] < -threshold)),
    "shorter_persistence": float(np.mean(draws[:, 4] < -threshold)),
    "higher_jitter": float(np.mean(draws[:, 5] > threshold)),
}
print("Research posterior probabilities conditional on this model:", probabilities)
print("Summary-support diagnostic:", posterior.input_diagnostics(context))
```

This example checks counts, timestamps, finite values, stationary observed
targets and declared nuisance fields. It cannot verify units, correct identities,
unwrapped coordinates, absence of mechanical forces, the physical observation
model or the initial-position sampling mechanism. Those assumptions require
scientific validation before use on experimental data. A quiet summary-support
diagnostic does not establish that an observation is in the modeled domain.

To regenerate the study, use:

```bash
python scripts/run_manuscript_sbi.py --stage all
```

Existing banks, checkpoints and completed evaluation cases are reused. Use a
new `--output` directory for a fresh complete reproduction. The first numeric
seed is the designated primary model; do not select a checkpoint by favorable
test or experimental results.
