# Implementation guide: target-associated migration inference

This framework is a synthetic validation study. It estimates effective movement
responses and tests identifiability; it does not report a discovery about real
killer cells. Actual numerical results and failure cases are in
`results/synthetic_v1/REPORT.md`.

## Scientific definition

“Trapped and more jiggled” has several possible meanings. We distinguish:

- Reduced persistent movement amplitude.
- Shorter directional persistence.
- Increased short-timescale translational fluctuations.
- Physical binding/repulsion, which changes movement without requiring any of
  the preceding intrinsic OU parameters to change.

The first three are fitted effects. Physical adhesion, moving targets,
spontaneous temporal variation and tracking noise are separate stress cases.
The framework does not infer a chemical sensing pathway, and currently contains
no explicit target-directed chemotactic force in its fitted baseline.

## Observation schema

Each dataset is a dictionary containing:

| Field | Shape | Meaning |
|---|---|---|
| `positions` | killer × frame × 2 | Observed unwrapped killer positions |
| `targets` | target × frame × 2 | Target positions, periodic coordinates |
| `times` | frame | Observation times in minutes |
| `config` | mapping | Acquisition, geometry and assumed nuisance settings |

Synthetic datasets also contain truth labels such as `theta`, `cell_offsets`
and `true_positions`. **Inference and summaries do not read these fields.**
Tests explicitly modify truth labels to verify invariance. NPZ serialization
uses arrays and JSON metadata without pickle objects.

The benchmark uses 2D periodic geometry with a fixed contact distance. It does
not directly support arbitrary masks, 3D images or nonperiodic chambers. An
experimental adapter must specify actual timestamps, calibration and geometry.

## Reduced movement model

For killer i at a given integration step, let d be its minimum-image distance
to the nearest target, and R the sum of the assumed cell radii. Define

\[
g_i(t)=\exp[-\max(d_i(t)-R,0)/\ell].
\]

The model is

\[
dv_i=-v_i/\tau_i\,dt+\sqrt{2S_i^2/\tau_i}\,dW_v,
\qquad dx_i=v_i\,dt+J_i\,dW_x+F_i\,dt,
\]

with independent Brownian processes per spatial axis. Its response functions are

\[
\log S_i=\mu_S+u_i+\delta_Sg_i,\quad
\log\tau_i=\mu_\tau+\delta_\tau g_i,\quad
\log J_i=\mu_J+\delta_Jg_i,
\qquad u_i\sim N(0,0.25^2).
\]

Observed coordinates satisfy y=x+Normal(0, sigma_obs²). In the baseline,
sigma_obs=0.15 µm and F=0. S is the stationary **per-coordinate active velocity
standard deviation**, not instantaneous or mean scalar speed. J has units
µm/sqrt(minute), tau has units minutes, and deltas are log contact/far ratios.
This parameterization avoids separately estimating an arbitrary latent
polarity scale and its displacement coupling.

Both methods use the same six-dimensional uniform prior. Baseline S ranges
2–5 µm/min, tau 0.8–3 min, and J 0.4–1.2 µm/sqrt(min). Delta_S and delta_tau
range −1.2 to 0.5; delta_J ranges −0.5 to 1.2. All baseline priors are uniform
on log scales. The asymmetry reflects the hypothesis but retains zero and the
opposite direction; it is a synthetic design choice requiring sensitivity
analysis before application.

Coefficients are frozen within each internal timestep. The velocity and its
integrated displacement are sampled jointly from the exact constant-coefficient
OU transition, retaining their cross covariance. This removes Euler error for
free OU motion, but spatially changing parameters and mechanical forces still
require a timestep study. Positions remain unwrapped; interaction distances
use periodic minimum images, preventing artificial large displacements.

The accelerated kernel uses Numba, numeric arrays and pre-generated random
numbers. It avoids dataframe construction, per-agent Python objects and event
logging in the loop. Optional mechanical calculations use bounded killer–target
pair loops, scaling O(N_k N_t T); very large dense populations would benefit
from neighbor lists. The reference engine uses identical draws to test exact
implementation agreement.

The baseline has penetrable targets because repulsion is zero. It is therefore
an identifiability benchmark for proximity-dependent motion, not a physical
contact simulation. Optional Hookean repulsion and stochastic bonds restore
soft interactions, with explicit discretization checks. Reduced-model targets
move independently; the original source ABM retains reciprocal forces.

## Supplied ABM connection

`src/mantishrimp/config.py` now exposes optional proximity-response coefficients
and `Simulation.run` accepts positive per-killer speed factors. Local polarity
decay scales as exp(−delta_tau*g), polarity noise as exp(−delta_tau*g/2),
coupling as exp(delta_speed*g), and translation noise as exp(delta_jitter*g).
This keeps local stationary polarity variance fixed when persistence changes.
Zero-effect defaults retain the original arithmetic and random-number path;
a golden regression captured before editing checks that behavior, including
binding, adaptive stepping, mechanical feedback, damage and exhaustion.

`from_mantishrimp_config` maps the source OU scales to the reduced model using
tau=1/decay and S=coupling*polarity_noise/sqrt(2*decay). The independent
`simulate_reference` bridge executes the source ABM with fine Euler steps and
the same observed-data schema. It disables damage/exhaustion for the movement
question but preserves requested mechanical forces. Source initial polarity is
zero, and changing polarity coupling differs mathematically from changing an
OU velocity amplitude. Transfer validation deliberately retains these
differences rather than claiming the models are identical.

## Hierarchical state-space inference

`state_space.fit` integrates latent velocity and position with a Kalman
likelihood and samples the six global parameters plus cell random intercepts.
The sampler uses warmup-only adaptation, Hessian-preconditioned proposals,
proper Metropolis proposal corrections, and a hierarchy-preserving conditional
intercept update. A Laplace approximation is used for proposal construction;
it is not substituted for the posterior samples.

Four independently seeded chains are checked with rank-normalized R-hat and
bulk/tail effective sample sizes. The benchmark requires R-hat<1.01 and both
ESS values at least 400. Failed fits receive more samples; unresolved cases are
retained as failures, with conditional-on-success summaries and unresolved-case
performance bounds. Good convergence cannot rescue a misspecified likelihood.

The hierarchy SD and observation SD are known nuisance settings in this study.
Distance is computed from observed positions: this is a conditional-covariate
approximation when localization error is nonzero. Missing-position observation
updates can be skipped, but interpolation used for their distance covariates
introduces another approximation. The primary model does not infer arbitrary
latent temporal parameter drift, separate contact-only jumps, binding states,
or unknown interaction length. Returned parameter paths are posterior response
curves evaluated along distance histories, not independent per-frame estimates.

`fixed_effects=(0,0,0)` fits a target-independent baseline. Posterior effect
probabilities are not Bayes factors; no marginal likelihood is calculated.
The likelihood updates latent-state distributions sequentially, but global
parameter inference uses complete movies retrospectively. This release does
not implement an online parameter filter.

## Simulation-based inference

`summaries.summarize` constructs 82 observed-only features, including proximity
strata, multiscale displacements, autocorrelation/turning, radial motion, dwell
and exposure, and within-cell contrasts. Each training row represents a complete
independent simulated movie. Cell random effects are drawn afresh and therefore
marginalized through simulation.

Two likelihood-free routes are implemented:

1. Empirical rejection ABC retains nearest prior-bank draws in standardized
   summary space. Report its tolerance and acceptance fraction. Correlated
   high-dimensional summaries and finite tolerance make it an approximation.
2. A custom PyTorch mixture-density network learns a joint conditional posterior.
   Full Cholesky covariance matrices preserve parameter correlations. A logit
   transformation enforces prior support; density evaluation includes the
   Jacobian. Early stopping uses a held-out movie split. This implementation is
   not the separately distributed `sbi` library.

Both methods approximate p(theta|summaries), not necessarily p(theta|full tracks).
Check simulation-based calibration, interval width, RMSE, posterior predictive
discrepancies and null decisions on separately generated datasets. A prior-only
posterior can be calibrated but uninformative; coverage alone is insufficient.

Use `framework.infer_neural` for ordinary application: it checks actual array
shapes/timing and the declared acquisition protocol against training metadata.
Known shifts such as changed localization noise require retraining or explicit
exploratory override. The low-level `.sample` remains available for deliberate
stress tests. No summary-support check can guarantee detection of an unknown
model mismatch.

## Validation and decisions

The default study uses 16,000 prior-bank movies, a separate 256-movie
prior-predictive test, 32 replicates per named neural/ABC scenario, a 24-dataset
state-space recovery pilot, and repeated null/combined state-space fits.
Independent-source transfer and timestep studies are saved separately.

Each declared directional decision requires marginal posterior probability
>0.95 for a change beyond log(1.2): decreased active amplitude, decreased
persistence, or increased jitter. Any-effect is the OR of these three
separate decisions; their false-alarm rate
must be measured rather than assumed to equal 5%. An inconclusive result is
not evidence of equivalence. Mechanical-only cases genuinely change observed
movement and test intrinsic-mechanism attribution, not absence of all behavior.

Split at the independent movie level, preserve failures, and avoid selecting
only successful encounters. Report Wilson uncertainty for rates across movies.
The state-space and neural nuisance assumptions must be explicit: knowing the
correct observation noise gives the likelihood model information unavailable
to an unadapted neural estimator trained under a different noise level.

## Reproduction

See the root README for installation and stage commands. The benchmark writes
software versions and source SHA-256 hashes alongside seeds, design settings,
posteriors, calibration records, figures and a generated report. Notebooks are
regenerated by `scripts/build_notebooks.py` and executed by
`scripts/execute_notebooks.py`. They inspect the saved evidence without
automatically launching the complete simulation/training campaign.
