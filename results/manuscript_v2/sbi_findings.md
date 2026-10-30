# Additional SBI manuscript experiments

**Synthetic follow-up study; all scientific decisions are exploratory and informed by the v1 study.**

## Design and reproducibility

The plan was saved before generating the new bank and evaluation movies. Three new fixed-noise training seeds share the original 16,000-movie bank. Three noise-conditioned and three noise-marginalized models share a new 24,000-movie bank, with localization SD independently drawn log-uniformly from 0.05 to 0.7 µm. The conditioned model appends known log localization SD to the 82 observed-only summaries; the marginalized ablation uses the same movies without this scalar. Model architectures and optimization hyperparameters are unchanged. First numerical seeds were designated primary before evaluation. No network was selected by test performance. Training-seed sensitivity is conditional on a common bank and includes split/initialization/optimization variability.

Independent evaluation uses 1,024 matched-noise prior-predictive movies and 1,024 varying-noise prior-predictive movies. Scenario datasets are shared across models; they must never be pooled as additional independent observations. Separate paired comparisons regenerate exactly the 336 SSM follow-up movies. Matching position hashes and parameter truth are saved/checked as SSM results become available.

## Matched-protocol primary network

| Scenario | Independent movies | Any-effect count | 95% Wilson interval | Slowdown | Shorter persistence | Increased jitter |
|---|---:|---:|---:|---:|---:|---:|
| null | 256 | 2 | [0.2%, 2.8%] | 0.4% | 0.0% | 0.4% |
| varied_null | 256 | 1 | [0.1%, 2.2%] | 0.0% | 0.4% | 0.0% |
| slowdown | 128 | 124 | [92.2%, 98.8%] | 96.9% | 0.8% | 0.0% |
| decorrelation | 128 | 42 | [25.3%, 41.3%] | 0.0% | 32.8% | 0.0% |
| jitter | 128 | 128 | [97.1%, 100.0%] | 0.0% | 0.0% | 100.0% |
| combined | 128 | 128 | [97.1%, 100.0%] | 89.1% | 13.3% | 100.0% |
| weak | 128 | 45 | [27.4%, 43.8%] | 11.7% | 3.1% | 25.8% |

Nominal 90% coverage across parameters is 89.9%–91.6% in 1,024 prior-predictive movies. This is marginal prior-predictive calibration, not proof of conditional or joint calibration. The combined scenario reliably produces an any-effect decision, but persistence detection remains weak. The v1 32-movie conclusion that every combined slowdown was recovered does not persist exactly in the larger follow-up.

## Observation-noise intervention

| Model | High-noise null flag count / 128 | 95% Wilson interval | High-noise combined any-effect count / 128 |
|---|---:|---:|---:|
| frozen_v1 | 128/128 | [97.1%, 100.0%] | 128/128 |
| conditioned_127001 | 1/128 | [0.1%, 4.3%] | 123/128 |
| conditioned_127002 | 1/128 | [0.1%, 4.3%] | 124/128 |
| conditioned_127003 | 2/128 | [0.4%, 5.5%] | 123/128 |
| marginalized_127001 | 3/128 | [0.8%, 6.7%] | 121/128 |
| marginalized_127002 | 1/128 | [0.1%, 4.3%] | 119/128 |
| marginalized_127003 | 1/128 | [0.1%, 4.3%] | 118/128 |

Primary conditioned-model marginal 90% coverage across 1,024 varying-noise movies is 89.9%–92.4%. Its baseline-jitter log-parameter RMSE is 0.091, compared with 0.161 when noise is marginalized without the known-noise context. Corresponding mean 90% interval widths are 0.289 and 0.508. Broadening the simulation support already repairs most false attribution; the extra known-noise context improves jitter precision. These results do not isolate the scalar as the sole cause of improvement over the original fixed-noise model.

The high-noise primary conditioned combined detection rates are 67.2% slowdown, 1.6% shorter persistence and 94.5% increased jitter. Thus recovery of any effect (96.1%) must not be equated with recovery of every mechanism. The method uses the correct known localization SD; it is not a method for estimating unknown localization error.

## Training-seed sensitivity

| Fixed-noise model | Calibration 90% coverage range | Fixed-null any rate | Persistence-only rate | Combined persistence rate |
|---|---:|---:|---:|---:|
| frozen_v1 | 89.6%–92.9% | 1.2% | 35.2% | 3.9% |
| fixed_117001 | 89.9%–91.6% | 0.8% | 32.8% | 13.3% |
| fixed_117002 | 91.6%–93.1% | 2.3% | 47.7% | 13.3% |
| fixed_117003 | 89.3%–92.6% | 1.6% | 43.8% | 9.4% |

## Limits and implementation notes

- Fixed-null and varied-baseline-null uncertainty is across whole movies. Training-seed ranges are descriptive and are not confidence intervals.
- The varied baseline null varies only the first three motility parameters; other nuisances remain fixed. It is not a stress test of mechanics, heterogeneity SD, target observation noise or geometry.
- Known-noise contexts use the correct acquisition metadata, appended after observed-only summaries. Tests confirm that dropping/mutating latent synthetic truth leaves the context unchanged.
- New 83-dimensional checkpoints must be called with `observed_context(data, condition_noise=True)` from the manuscript script; the original fixed-protocol application wrapper expects the original 82-dimensional feature scheme. The expanded models are currently research checkpoints, not an experimental-data adapter.
- All tests use the simplified penetrable-target proximity model. Source-ABM transfer/mechanical misspecification remain separate v1 evidence.
- `sbi_summary.json`, `sbi_scenario_summary.csv` and `sbi_parameter_summary.csv` expose aggregates; full evaluation JSON retains each movie’s truth, posterior metrics and seed. Training histories and model checkpoints are preserved.
