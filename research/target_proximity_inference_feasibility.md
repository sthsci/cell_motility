# Inferring changes in killer-cell migration near targets

Research assessment, 28 October 2026. This is a feasibility assessment based on the local code and published methods, not an analysis of experimental trajectories. The user confirmed that the reference is Metzner et al. (2015), “Superstatistical analysis and modelling of heterogeneous random walks,” and that target positions or masks are available at each time point. The exact representation, synchronization, dimensionality, sampling interval, and encounter durations still need inspection.

**Recommendation:** use a hierarchical Bayesian movement model with explicit distance/contact effects as the primary analysis. Retain the existing position observation model and OU/Kalman machinery where its assumptions apply. Compare a smooth response against a small switching-state or change-point model. Use freely drifting parameters only after recovery tests establish that the tracks contain enough information.

The statistical methods are established, and the confirmed target information supports direct modeling of proximity and direction. A complete validated killer–target analysis still needs a suitable model, verified synchronization, and validation at the experimental sampling rate.

With these annotations, build a per-frame record of killer position, target identity and position, separation, and target bearing. If target masks are available, measure killer-centroid-to-target-boundary distance; true surface-to-surface separation requires a killer mask or a justified size estimate as well. Contact should be annotated or inferred with explicit geometric uncertainty, rather than equated to an arbitrary centroid-distance threshold. Retain the killer displacement and target displacement separately so that target motion is not misinterpreted as directed killer movement.

## What we can estimate

Separate three scientific questions:

1. Does movement change over time within a cell?
2. Are changes associated with distance to targets, contact, or time since contact?
3. Does the cell actively orient toward a target, rather than encounter it by chance or remain nearby after slowing down?

Time-varying persistence and stochastic activity address the first question. Target covariates address the second. The third additionally needs a directional-bias term and controls for target-independent movement changes and geometry. A distance association alone cannot establish a sensing mechanism.

Useful estimands include persistence time, active movement amplitude, short-timescale random motion, signed target-directed bias, probability of occupying a movement state, and change onset/duration. These are effective movement properties, not direct molecular measurements. Infer a small subset initially rather than allowing every quantity to vary independently at every frame.

## Relevant established approaches

| Approach | Appropriate question | Practical route and limitation |
|---|---|---|
| Hierarchical covariate-dependent state-space model | Does persistence or movement amplitude vary with distance/contact, and how much do cells differ? | Extend the existing NumPyro model with a low-dimensional response curve and cell/movie effects. Best first choice for the scientific question. |
| Continuous parameter evolution | What residual within-cell variation remains beyond the shared response? | Smooth splines, random walks, OU processes, or Gaussian-process priors. Flexible, but short trajectories may be dominated by the smoothness prior. |
| Hidden Markov / switching movement model | Do cells alternate between persistent movement, slow movement, and arrest? | State-specific movement parameters; let distance/contact affect transition probabilities. Interpret inferred states statistically before assigning biological names. |
| Hidden semi-Markov model | Does the time spent in a state matter? | Explicit dwell-time distributions, useful when an ordinary homogeneous HMM's geometric durations are unsuitable. Additional fitting work is required. |
| Bayesian change-point model | When did a discrete change begin, and how large was it? | Infer onset and pre/post parameters, with an autoregressive or state-space likelihood. A change need not occur exactly at annotated contact. |
| Simulation-based inference | Is the biologically required simulator too complex for a tractable likelihood? | Useful for detailed geometry, cell shapes, interactions, or signaling. It adds simulation and validation costs when the current Gaussian likelihood is already available. |

Metzner et al. (2015) infer time-varying persistence and innovation activity in a displacement AR(1) model, including gradual and abrupt changes and forward/backward inference. Mark et al. (2018) extend this with inference and comparison of parameter-dynamics models. These are related precedents; neither by itself identifies a target-dependent mechanism. [Metzner 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4491834/), [Mark 2018](https://www.nature.com/articles/s41467-018-04241-5).

The AR(1) activity parameter is an innovation amplitude, not speed. For a locally stationary scalar AR(1), variance is proportional to activity squared divided by one minus persistence squared. Thus a speed change can arise from changes in either parameter. Their discrete displacement persistence is not numerically interchangeable with the latent OU persistence parameter in this repository.

## Software choices

- **bayesloop:** the closest packaged baseline for low-dimensional changing-parameter inference. Its documented transitions include gradual random walks, regime switches, and change points; it supports offline and online studies and model evidence. Grid scaling makes it more attractive for a few parameters than for a large joint latent-state model. Modeling localization error and target covariates still requires care. [Official documentation](https://docs.bayesloop.com/en/latest/).
- **Existing NumPyro/Kalman implementation:** the most direct project-specific route. It already marginalizes latent polarity and fits kinetic parameters. The new biological response model and hierarchy would need to be written and validated.
- **Dynamax:** JAX implementations of HMMs and linear/nonlinear Gaussian state-space inference, including filtering and smoothing. Its documentation also demonstrates Bayesian parameter inference. It is an algorithmic foundation, not a complete killer-cell application. [Official documentation](https://probml.github.io/dynamax/).
- **particles:** documented particle filters, smoothing, particle MCMC, and SMC². Appropriate when nonlinear target dependence or non-Gaussian dynamics justify the computational cost. SMC² learns global parameters sequentially; time-varying biological parameters must still be specified as latent processes. [Official features](https://particles-sequential-monte-carlo-in-python.readthedocs.io/en/latest/features.html).
- **momentuHMM:** an established movement-analysis benchmark supporting correlated/biased movement, covariates, and attraction/repulsion centers, including moving centers. Its usual estimation is maximum likelihood, so it should not be described as a fully Bayesian replacement. [Primary software paper](https://doi.org/10.1111/2041-210X.12995), [official vignette](https://stat.ethz.ch/CRAN/web/packages/momentuHMM/vignettes/momentuHMM.pdf).

A standard Bayesian online change-point detector should not be applied to correlated displacement data with an independent-observation likelihood. Use a suitable dependent likelihood, or carefully modeled innovations. [Adams–MacKay original paper](https://www.cs.princeton.edu/~rpa/pubs/adams2007changepoint.pdf).

## What the repository already does

The current physical model is

\[
dp=a p\,dt+s_0\,dW_0,\qquad dx=b p\,dt+s_1\,dW_1,
\]

independently on each coordinate axis. The observed positions also have measurement noise. The implementation computes exact OU transition matrices for coefficients held constant within a sampling interval and uses a Kalman marginal likelihood in NumPyro.

- [Transition matrices and time dependence](../kalman_mcmc/kalman_analytical.py): `build_FQ_timevarying_for_track` uses a supplied `onset_idx`, a fixed sigmoid width, and a constrained pre/post relationship. The noise scales are constant through time. This does not infer an unknown parameter path or establish a target effect.
- [Bayesian model](../kalman_mcmc/mcmc_model.py): currently supports shared population parameters or labeled class parameters with hierarchical priors. It does not yet provide cell-specific response curves.
- [Data loader](../kalman_mcmc/data.py): reads tracks, masks, onset, and labels. NK/tumour groups can be selected, but there is no synchronized killer–target pairing, distance, direction, or contact model.
- [Simulator](../ou_sim.py): already supports multiple imposed regimes and switch times, useful for recovery experiments.
- [MDN training](../sbi/train_onset_mdn.py): estimates a fixed parameter vector using pooled trajectory representations. ABC-SMC populations and sequential neural-inference rounds are computational refinement stages; they do not automatically estimate biological changes over time.

No experimental trajectory files were found in the inspected repository. Default numerical settings do not establish the actual frame interval, spatial units, or localization accuracy.

## A concrete model extension

For example, write persistence time as

\[
\log\tau_i(t)=\alpha+u_i+f(d_i(t))+g(s_i(t)),
\]

where \(u_i\) represents cell heterogeneity, \(d_i(t)\) is target distance, and \(s_i(t)\) is time relative to an independently annotated encounter/contact. Start with either a distance response or an encounter-time response; these covariates can be strongly correlated, so fitting both flexibly may be unidentifiable. Add movie/donor variation where the experimental replication supports it.

A separate signed directional term should describe motion toward or away from the target. This prevents target-directed motion from being absorbed into apparent persistence or noise. A velocity formulation with persistence time and active-velocity amplitude is often easier to interpret than separate latent polarity scale and polarity-to-position coupling.

The current polarity model has a scale ambiguity: rescaling latent polarity by \(c\), its noise by \(c\), and coupling by \(1/c\) leaves the observable dynamics unchanged when the initial polarity distribution is rescaled consistently. Fix the polarity scale or fit identifiable combinations. An arbitrary fixed initial-state prior should not be treated as biological information resolving this ambiguity. Also, replacing \(b(t)p(t)\) by an OU velocity is not an exact reparameterization when \(b\) varies: it introduces an additional drift term involving \(b'(t)/b(t)\).

**Computational boundary:** prescribed parameter paths and known target regressors can preserve a conditional linear-Gaussian model, allowing a Kalman likelihood inside Bayesian fitting. However, distance and normalized target bearing depend nonlinearly on latent cell position. Treating their noisy measured values as fixed is an approximation. If this error matters, use joint latent-position inference, nonlinear filtering/smoothing, or particles. A linear attraction proportional to target position minus cell position can preserve linearity under appropriate coefficient assumptions, but may be biologically too restrictive.

For recorded movies, infer the posterior using all frames and reconstruct paths with smoothing. Filtering uses only frames available up to the present and is appropriate for live decisions. Smoothing can sharpen retrospective estimates but also spread apparent changes across an event boundary; validate timing before interpreting an inferred pre-contact response as anticipation.

## The strongest direct killer–target precedent

Metzner et al. (2021) analyze NK–K562 trajectories and compare target-directed turning against randomized-target references. They validate their test with blind migration, spontaneous switching, and target-guided search. This is a valuable independent check on inferred attraction, although it is not a continuous Bayesian parameter estimator. Their discussion also reports that fitting a misspecified interaction model can yield incorrect interaction estimates. [Detecting long-range interactions between migrating cells](https://www.nature.com/articles/s41598-021-94458-0).

Target-contact-associated arrest is biologically plausible, but spontaneous arrest must also be allowed. A direct NK study reports transient movement modes and arrest both with and without target contact; only its abstract was reviewed here. [Khorshidi et al. 2011](https://pubmed.ncbi.nlm.nih.gov/21687858/).

## Minimum validation before biological interpretation

1. **Audit data:** timestamps, units, 2D versus 3D acquisition, localization error, missed frames, track swaps, pre/during/post encounter coverage, and independent movies/donors. Verify that the confirmed per-frame target positions or masks align with the killer tracks. Surface separation is preferable to centroid distance for physical contact, but requires both cell boundaries or a justified geometric approximation.
2. **Define target assignment:** nearest-target identity can switch abruptly without a biological switch. Evaluate consistent pairing or a multi-target influence model. Separate pre-contact movement, contact mechanics, and post-contact motion.
3. **Fit a target-independent baseline:** include cell heterogeneity and spontaneous temporal changes. Comparing a target-dependent model only against constant motion would be an inadequate null.
4. **Run recovery tests:** simulate constant motion, spontaneous switching, smooth proximity responses, contact arrest, and directional bias with realistic observation noise and tracking gaps. Test distinguishability, false-change rates, onset error, and posterior interval coverage.
5. **Validate predictions:** hold out entire cells or movies as appropriate; do not split random frames from the same trajectory. Check movement direction, displacement distributions, persistence, dwell times, and encounter behavior.
6. **Check sensitivity:** vary smoothness priors, model family, target assignment, localization uncertainty, and geometry-preserving target randomizations. Propagate uncertainty into population comparisons rather than treating smoothed parameter means as error-free observations.

Selecting only successful encounters, or aligning all tracks to closest approach, can manufacture apparent approach patterns. Slow cells can also be overrepresented near targets because they remain there longer. Include non-encounters and null simulations through the same selection pipeline. Changes near contact may additionally reflect target motion, segmentation errors, crowding, or signals from neighboring killer cells.

Feasibility depends on timescales: if persistence is much shorter than the frame interval, it is difficult to separate from rapid random motion; if it is much longer than the observed segment, its duration is poorly bounded. An estimated value at every frame does not imply independent information at every frame. When encounters are brief, a pooled population response may be identifiable even if individual parameter paths are not.

**First implementation decision:** compare a shared hierarchical distance/contact response against a target-independent time-varying null, using the existing likelihood where valid. Use bayesloop as a low-dimensional exploratory benchmark, and add a switching alternative if the trajectories exhibit distinct pauses/runs. Choose the justified temporal resolution through recovery tests rather than from the plotting interval.
