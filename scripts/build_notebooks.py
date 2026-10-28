#!/usr/bin/env python3
"""Regenerate four clean educational notebooks without running computation.

Run from any directory with ``python /path/to/scripts/build_notebooks.py``.
The notebooks locate the project from their current working directory and read
the actual benchmark artifacts; outputs are deliberately left unexecuted here.
"""
from pathlib import Path
from textwrap import dedent

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(dedent(text).strip())


SETUP = r'''
from pathlib import Path
from dataclasses import asdict, replace
from io import BytesIO
import json
import os
import sys

candidates = [Path.cwd(), *Path.cwd().parents]
ROOT = next((p for p in candidates if (p / 'pyproject.toml').is_file()
             and (p / 'src' / 'motility_inference').is_dir()), None)
if ROOT is None:
    raise RuntimeError('Launch this notebook from the project or its notebooks directory.')
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
RESULTS = ROOT / 'results' / 'synthetic_v1'
RUN_TRAINING = False  # Large simulation banks and MCMC are never launched by default.
(ROOT / '.cache').mkdir(exist_ok=True)
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.cache'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import display, Image
from motility_inference.simulator import SimConfig, simulate, load_dataset, proximity
from motility_inference.priors import PARAMETER_NAMES, PRIOR_LOW, PRIOR_HIGH, scenario_theta

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 110})

def show(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=130, bbox_inches='tight')
    display(Image(data=buffer.getvalue()))
    plt.close(fig)

def read_json(name):
    path = RESULTS / name
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        print(f'{name} is being updated; rerun this cell after the benchmark finishes.')
        return None

print('Project:', ROOT)
print('Cached results:', RESULTS)
'''


def notebook1():
    return [
        md(r'''
        # 1 · What does “trapped and more jiggly” mean?

        This is a **synthetic methods experiment**, not a conclusion about real killer cells.
        We ask whether inference can recover known target-associated changes and avoid
        attributing spontaneous movement changes or mechanical trapping to those effects.

        The supplied `mantishrimp` ABM is retained, with optional proximity-response fields
        whose zero defaults preserve its original behavior. The reduced model keeps OU motility,
        periodic geometry, and optional target movement, repulsion, and stochastic adhesion.
        Cytotoxicity, exhaustion, per-step event tables, and within-type cell forces are omitted.
        Consequently its timing against the full ABM is **not** an equivalent-model speedup.
        The NumPy-reference versus Numba timing below does compare the same reduced model.
        '''),
        code(SETUP),
        md(r'''
        For nearest-target surface gap $d_i(t)$, define
        $$
        g_i(t)=\exp[-\max(d_i(t),0)/\ell].
        $$
        The three effects enter log parameters: $\log s_i=\log s_0+u_i+\delta_sg_i$,
        $\log\tau_i=\log\tau_0+\delta_\tau g_i$, and
        $\log j_i=\log j_0+\delta_jg_i$, with $u_i\sim N(0,\sigma_u^2)$.
        The gate is one at contact and tends to zero far away; $\exp(\delta)$ is a contact/far ratio.

        - **Active amplitude $s$:** stationary SD of each OU velocity coordinate, not mean observed speed.
        - **Persistence time $\tau$:** time scale of active-velocity memory.
        - **Jitter $j$:** translational Brownian amplitude, with units length / square-root time.

        Reduced amplitude, shorter persistence, and increased jitter can yield slower net
        progress with more local fluctuations. They are separate mechanisms, not synonyms.
        Localization noise also produces apparent jiggling and is modeled separately.
        '''),
        code('''
        config = SimConfig()
        experiments = {name: simulate(scenario_theta(name), config, seed=103)
                       for name in ('null', 'combined')}
        table = []
        for name, data in experiments.items():
            gate, gap, _, _ = proximity(data['positions'], data['targets'],
                config.box_size, config.contact_distance, config.interaction_length)
            table.append({'scenario': name, 'near_fraction_gate_above_half': (gate > .5).mean(),
                          'far_fraction_gate_below_point2': (gate < .2).mean(),
                          'amplitude_contact_ratio': np.exp(data['theta'][3]),
                          'persistence_contact_ratio': np.exp(data['theta'][4]),
                          'jitter_contact_ratio': np.exp(data['theta'][5])})
        display(pd.DataFrame(table).set_index('scenario'))
        display(pd.Series(asdict(config), name='simulation design'))
        '''),
        code('''
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for ax, (name, data) in zip(axes, experiments.items()):
            gate, _, _, _ = proximity(data['positions'], data['targets'],
                config.box_size, config.contact_distance, config.interaction_length)
            # Points avoid drawing artificial connecting lines across periodic seams.
            points = data['positions'] % config.box_size
            ax.scatter(points[..., 0].ravel(), points[..., 1].ravel(),
                       c=gate.ravel(), cmap='viridis', s=2, vmin=0, vmax=1, alpha=.65)
            for target in data['targets'][:, 0]:
                ax.add_patch(plt.Circle(target, config.contact_distance / 2,
                                       facecolor='#ce6c47', alpha=.65))
            ax.set(xlim=(0, config.box_size), ylim=(0, config.box_size),
                   xlabel='x (µm)', ylabel='y (µm)', title=name)
            ax.set_aspect('equal')
        show(fig)
        '''),
        code('''
        data = experiments['combined']
        gate, gap, _, _ = proximity(data['positions'], data['targets'],
            config.box_size, config.contact_distance, config.interaction_length)
        # Choose a track with exposure variation for illustration, not performance evaluation.
        cell = int(np.argmax(np.std(gate, axis=1)))
        truth_gate = data['true_gate'][cell]
        fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
        for ax, label, b, e in zip(axes, ('Active amplitude (µm/min)', 'Persistence (min)',
                                         'Jitter (µm/√min)'), (0, 1, 2), (3, 4, 5)):
            offset = data['cell_offsets'][cell] if b == 0 else 0
            values = np.exp(data['theta'][b] + offset + data['theta'][e] * truth_gate)
            ax.plot(data['times'], values, color='#167d8d')
            ax.set_ylabel(label)
        axes[-1].set_xlabel('Time (min)')
        axes[0].set_title(f'Synthetic ground truth for cell {cell}; not inferred parameters')
        show(fig)
        '''),
        md('''
        The joint OU velocity/displacement transition is exact for coefficients frozen within
        one integration step. Spatially changing coefficients and mechanical forces still
        require a substep-convergence check. Observation cadence and integration step are
        different settings. Target-independent spontaneous amplitude changes and
        adhesion are retained as stress tests.
        '''),
        code('''
        speed = read_json('speed.json')
        if speed is None:
            print('No cached timing study. Run: python scripts/run_synthetic_benchmark.py --stage speed')
        else:
            display(pd.Series({
                'same reduced-model speedup': speed['same_model_speedup'],
                'paired engine maximum error': speed['paired_engine_max_absolute_difference'],
                'Numba median seconds': speed['timings']['numba']['median_seconds'],
                'NumPy reference median seconds': speed['timings']['numpy']['median_seconds'],
            }))
            print(speed['supplied_abm_comparison_note'])
        '''),
        md('''
        The source ABM now also exposes `killer_delta_speed`, `killer_delta_tau`, and
        `killer_delta_jitter`, with `target_interaction_length` setting the response scale.
        Its explicit polarity/Euler update and initialization remain distinct from the
        reduced integrated-velocity simulator. Independent source-ABM validation below is
        therefore a transfer test, not another draw from the training generator.
        '''),
        code('''
        from mantishrimp.config import MotilityConfig
        source_defaults = MotilityConfig()
        display(pd.Series({name: getattr(source_defaults, name) for name in
            ('target_interaction_length', 'killer_delta_speed', 'killer_delta_tau', 'killer_delta_jitter')},
            name='optional source-ABM response defaults'))
        reference = read_json('reference_validation.json')
        if reference:
            rows = []
            for case, result in reference.items():
                for method, aggregate in result['aggregate'].items():
                    rows.append({'source_ABM_case': case, 'method': method, 'datasets': aggregate['n'],
                                 'any_direction_detected': aggregate['any_detection_rate']})
            display(pd.DataFrame(rows))
            print('These raw transfer-test decisions are exploratory. The guarded API identifies the known generator mismatch and abstains.')
        else:
            print('Independent source-ABM validation is not available yet; no transfer claim is made.')
        '''),
        md('''
        Continue to notebook 2 for likelihood-based inference and notebook 3 for
        likelihood-free inference. Their performance is assessed on independent whole
        simulated movies in notebook 4. Frames within a movie are not independent replicates.
        ''')]


def notebook2():
    return [
        md(r'''
        # 2 · Hierarchical state-space inference

        The model uses observed positions with a separate measurement-error term. Latent
        OU velocity and position are integrated out by a Kalman likelihood; MCMC estimates
        six population parameters and one baseline-amplitude offset per cell.

        $$
        dv_i=-v_i\,dt/\tau_i+\sqrt{2s_i^2/\tau_i}\,dW_v,
        \qquad dx_i=v_i\,dt+j_i\,dW_x,
        \qquad y_i=x_i+\epsilon_i.
        $$

        The hierarchy pools information across cells. Its between-cell SD and localization
        SD are **fixed benchmark nuisance values**, not inferred experimental calibrations.
        '''),
        code(SETUP),
        md(r'''
        **Conditional-covariate approximation:** the likelihood uses gates calculated from
        observed positions and targets. The Gaussian transition is exact given these gates,
        but the full model with uncertain positions determining their own gates is nonlinear.
        Thus this is not an exact likelihood for every spatial simulator. A good R-hat does
        not remove this model approximation. Fine-substep, noise, and mechanics stress tests
        explicitly assess departures from its assumptions.

        A Laplace covariance preconditions the Metropolis sampler; it is not the reported
        posterior. Adaptation stops after warmup and independent chains are diagnosed using
        rank R-hat and bulk/tail effective sample sizes.
        '''),
        code('''
        from motility_inference.state_space import fit, log_likelihood, predict_parameter_paths
        case = 'combined'
        data_path = RESULTS / 'examples' / f'{case}.npz'
        fit_path = RESULTS / 'examples' / f'{case}_ssm.npz'
        cached = data_path.is_file() and fit_path.is_file()
        if cached:
            data = load_dataset(data_path)
            with np.load(fit_path, allow_pickle=False) as z:
                fitted = {'samples': z['samples'].copy(), 'cell_samples': z['cell_samples'].copy()}
            record = (read_json('state_space.json') or {}).get(case, {})
            diagnostics = record.get('diagnostics', {})
            print('Loaded completed benchmark fit; no MCMC rerun.')
        else:
            # A bounded computational example. Its short chains cannot establish recovery.
            data = simulate(scenario_theta(case), SimConfig(n_killers=3, n_frames=41), seed=201)
            fitted = fit(data, draws=80, warmup=80, chains=2, seed=202)
            diagnostics = fitted['diagnostics']
            print('DEMONSTRATION ONLY: cached benchmark absent; short-chain result is not accepted evidence.')
        candidate = scenario_theta('null')
        print('Illustrative log likelihood at a null candidate with zero cell offsets:',
              log_likelihood(data, candidate))
        print('This single value is not a model comparison or a Bayes factor.')
        display(pd.Series({k: diagnostics.get(k, 'not available')
                           for k in ('max_rhat', 'min_ess_bulk', 'min_ess_tail', 'converged')}))
        sampling_pass = bool(diagnostics.get('converged', False))
        if not sampling_pass:
            print('EXPLORATORY ONLY: this representative fit did not pass the sampling gate. Do not treat its intervals as accepted results.')
        '''),
        code('''
        samples = fitted['samples'].reshape(-1, 6)
        low, median, high = np.quantile(samples, [.05, .5, .95], axis=0)
        table = pd.DataFrame({'parameter': PARAMETER_NAMES, 'synthetic_truth': data['theta'],
                              'posterior_median': median, 'lower90': low, 'upper90': high})
        display(table.set_index('parameter'))
        fig, axes = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
        for ax, j, label in zip(axes, (3, 4, 5), ('Amplitude effect', 'Persistence effect', 'Jitter effect')):
            ax.hist(samples[:, j], bins=35, density=True, color='#167d8d', alpha=.75)
            ax.axvline(data['theta'][j], color='#c65335', label='synthetic truth')
            ax.axvline(0, color='grey', ls=':')
            ax.set(xlabel='Log contact/far ratio', title=label)
        axes[0].legend(fontsize=8)
        if not sampling_pass:
            fig.suptitle('Exploratory illustration: sampling diagnostic gate failed')
        show(fig)
        '''),
        md('''
        A parameter path here is the estimated **distance-response function evaluated along
        the observed trajectory**. It is not a freely inferred independent value at every
        frame. The following bands express pointwise parameter uncertainty, not prediction
        intervals for noisy observed speed and not simultaneous confidence bands.
        '''),
        code('''
        paths = predict_parameter_paths(data, fitted, max_samples=256, seed=203)
        cell = int(np.argmax(np.std(paths['gate'], axis=1)))
        fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
        for ax, name, label, b, e in zip(axes, ('speed', 'tau', 'jitter'),
                ('Active amplitude (µm/min)', 'Persistence (min)', 'Jitter (µm/√min)'), (0,1,2), (3,4,5)):
            ax.fill_between(paths['times'], paths[name]['lower'][cell],
                            paths[name]['upper'][cell], color='#167d8d', alpha=.2)
            ax.plot(paths['times'], paths[name]['median'][cell], color='#167d8d', label='posterior median')
            offset = data['cell_offsets'][cell] if b == 0 else 0
            truth = np.exp(data['theta'][b] + offset + data['theta'][e] * data['true_gate'][cell])
            ax.plot(data['times'], truth, color='#c65335', ls='--', label='synthetic truth')
            ax.set_ylabel(label)
        axes[0].legend(fontsize=8)
        axes[-1].set_xlabel('Time (min)')
        if not sampling_pass:
            fig.suptitle('Exploratory parameter paths: sampling diagnostic gate failed')
        show(fig)
        '''),
        md('''
        The global interval may include zero because the data are weak, not because the null
        is established. The benchmark separately reports posterior probability of a practical
        directional effect and probability of lying within a predefined equivalence region.
        For real data, localization accuracy, heterogeneity, contact geometry, and the response
        length scale would need calibration or sensitivity analysis.

        Background: [Metzner et al. (2015)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4491834/)
        established time-varying migration inference. This model is a separate implementation
        with explicit target covariates. [ArviZ diagnostics](https://python.arviz.org/en/stable/api/diagnostics.html)
        describe the reported sampling checks.
        ''')]


def notebook3():
    return [
        md('''
        # 3 · Simulation-based inference

        This route does not evaluate the trajectory likelihood. It simulates many independent
        movies from a stated prior and learns how their **observed-data summaries** relate to
        parameter values. Two implementations are included:

        - Prior-bank rejection ABC: retain simulated movies whose summaries are nearby.
        - A **custom PyTorch mixture-density network (MDN)**: learn a joint conditional
          density on six parameters. This is not the third-party `sbi` package.

        Both target a summary posterior, which can lose information relative to full data.
        The neural posterior uses a bounded transform, includes its density Jacobian, and
        generates correlated parameter draws through full component covariance matrices.
        '''),
        code(SETUP),
        code('''
        import torch
        torch.set_num_threads(2)
        from motility_inference.summaries import summarize
        from motility_inference.sbi import generate_bank, train_posterior, NeuralPosterior, abc_samples
        bank_path = RESULTS / 'training_bank.npz'
        if bank_path.is_file():
            with np.load(bank_path, allow_pickle=False) as z:
                bank_theta, bank_summaries = z['theta'].copy(), z['summaries'].copy()
                bank_config = SimConfig(**json.loads(str(z['config_json'])))
            print('Loaded independent dataset-level simulation bank:', bank_summaries.shape)
        else:
            bank_config = SimConfig()
            bank_theta, bank_summaries = generate_bank(256, bank_config, seed=301)
            print('DEMONSTRATION ONLY: 256-movie bank; this is not the completed validation study.')
        example_path = RESULTS / 'examples' / 'combined.npz'
        data = load_dataset(example_path) if example_path.is_file() else simulate(scenario_theta('combined'), bank_config, seed=302)
        if data['config'] != asdict(bank_config):
            raise ValueError('Example observation protocol differs from the simulation bank.')
        observed, names = summarize(data, return_names=True)
        print('Observed-only summary dimensions:', len(observed))
        display(pd.DataFrame({'feature': names, 'value': observed}).head(20))
        '''),
        md('''
        Summaries include distance-stratified multi-lag MSD, increment correlations,
        turning/reversals, radial displacement, within-cell contrasts, and exposure/dwell.
        Distance strata use the beginning of each interval, not eventual successful contact.
        Latent velocity, truth parameters, true gates, and random cell effects are excluded.
        Lags are in frames, so a different imaging cadence requires a matched simulation bank.
        '''),
        code('''
        abc = abc_samples(observed, bank_theta, bank_summaries,
                          n_accept=min(256, max(16, len(bank_theta)//20)))
        print('ABC accepted fraction:', abc['acceptance_fraction'])
        print('ABC standardized-summary tolerance:', abc['epsilon'])
        print('These are actual accepted prior samples; nonzero tolerance makes ABC approximate.')
        checkpoint = RESULTS / 'neural_posterior.pt'
        model = None
        if checkpoint.is_file() and bank_path.is_file():
            model = NeuralPosterior.load(checkpoint)
            print('Loaded trained posterior; no training performed.')
        elif RUN_TRAINING:
            model = train_posterior(bank_theta, bank_summaries, seed=303,
                                    max_epochs=60, hidden=64, components=2)
            print('Small demonstration training only; no validated performance claim.')
        else:
            print('Neural checkpoint absent. Training is disabled by RUN_TRAINING=False.')
        neural_draws = model.sample(observed, n=1500, seed=304) if model is not None else None
        if model is not None:
            display(pd.Series(model.input_diagnostics(observed)))
        '''),
        md('''
        Use the high-level entry point for application. It checks actual array dimensions,
        time spacing, near/far information, and the observation protocol against the bank.
        Matching inputs can yield an **advisory**, not experimental proof. A protocol shift
        raises an error by default. Deliberately allowing a stress test still returns an
        explicit abstention and no supported effects.
        '''),
        code('''
        from motility_inference.framework import infer_neural
        if checkpoint.is_file() and (RESULTS / 'bank_metadata.json').is_file():
            observed_only = {key: data[key] for key in ('positions', 'targets', 'times', 'config')}
            advisory = infer_neural(observed_only, checkpoint, n_draws=500, seed=305)
            display(pd.Series({'matched_decision': advisory['decision'],
                               'abstained': advisory['abstained'],
                               'candidate_effects': advisory['candidate_effects'],
                               'supported_effects': advisory['supported_effects']}))
            noise_path = RESULTS / 'examples' / 'noisy_null.npz'
            noisy = load_dataset(noise_path) if noise_path.is_file() else simulate(
                scenario_theta('null'), replace(bank_config, localization_sd=0.6), seed=306)
            noisy_observed = {key: noisy[key] for key in ('positions', 'targets', 'times', 'config')}
            try:
                infer_neural(noisy_observed, checkpoint, n_draws=200, seed=307)
            except ValueError as error:
                print('Expected protocol guard:', error)
            stress = infer_neural(noisy_observed, checkpoint, allow_protocol_shift=True,
                                  n_draws=300, seed=308)
            display(pd.Series({'stress_decision': stress['decision'], 'abstained': stress['abstained'],
                               'supported_effects': stress['supported_effects']}))
        else:
            print('Guard demonstration requires the trained checkpoint and bank_metadata.json.')
        '''),
        code('''
        fig, axes = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
        for ax, j, label in zip(axes, (3,4,5), ('Amplitude effect', 'Persistence effect', 'Jitter effect')):
            bins = np.linspace(PRIOR_LOW[j], PRIOR_HIGH[j], 35)
            ax.hist(abc['samples'][:, j], bins=bins, density=True, alpha=.45,
                    color='#d89b39', label='ABC')
            if neural_draws is not None:
                ax.hist(neural_draws[:, j], bins=bins, density=True, histtype='step',
                        linewidth=2, color='#167d8d', label='Neural posterior')
            ax.axvline(data['theta'][j], color='#c65335', label='synthetic truth')
            ax.axvline(0, color='grey', ls=':')
            ax.set(title=label, xlabel='Log contact/far ratio')
        axes[0].legend(fontsize=8)
        show(fig)
        '''),
        code('''
        training = read_json('training.json')
        if training is not None:
            history = training['history']
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.plot(history['train_nll'], label='training')
            ax.plot(history['validation_nll'], label='early-stopping validation')
            ax.axvline(history['best_epoch'], color='grey', ls=':')
            ax.set(xlabel='Epoch (zero based)', ylabel='NLL in standardized logit coordinates')
            ax.legend()
            show(fig)
        else:
            print('No saved training curve.')
        print('Reproduce training explicitly from the project root:')
        print('python scripts/run_synthetic_benchmark.py --stage bank --n-train 16000')
        print('python scripts/run_synthetic_benchmark.py --stage train --epochs 300')
        print('python scripts/run_synthetic_benchmark.py --stage evaluate --n-test 256 --replicates 32')
        '''),
        md('''
        Early-stopping validation chooses weights; it is **not** the independent calibration
        set. Calibration, null false positives, and alternatives are evaluated on entirely
        new simulated movies. Coverage alone is insufficient: an unchanged prior can have
        correct coverage, so notebook 4 also reports widths and recovery errors.

        An input-range warning is only a crude diagnostic; its absence does not establish
        model adequacy. Adhesion, spontaneous state changes, or altered noise can make
        confident inference wrong. Sequential rounds of SBI would refine a computational
        posterior and should not be confused with inference at successive biological times.

        General workflow reference: [SBI diagnostics and validation guidance](https://sbi.readthedocs.io/en/latest/how_to_guide.html).
        ''')]


def notebook4():
    return [
        md('''
        # 4 · What did the synthetic validation establish?

        All numbers below are read from saved benchmark artifacts. Missing artifacts produce
        an explanatory message, not invented results. The unit of replication is an independent
        simulated movie; uncertainty in coverage and detection rates is computed across movies.

        The prespecified decision is posterior probability greater than 0.95 for an effect
        beyond a factor of 1.2 in the hypothesized direction: lower active amplitude,
        shorter persistence, or greater jitter. A null false-positive rate and an alternative
        detection rate are different quantities. Absence of detection is not evidence of equivalence.
        '''),
        code(SETUP),
        code('''
        from motility_inference.evaluation import wilson_interval, aggregate_metrics
        calibration = read_json('calibration.json')
        ssm_calibration = read_json('state_space_calibration.json')
        scenarios = read_json('scenarios.json')
        ssm_cases = read_json('state_space.json')
        ssm_repeated = read_json('state_space_scenarios.json')
        rows = []
        def append_calibration(label, aggregate):
            if not aggregate or not aggregate.get('n', 0):
                return
            for j, parameter in enumerate(PARAMETER_NAMES):
                interval = aggregate['coverage90_wilson'][j]
                rows.append({'method': label, 'parameter': parameter, 'datasets': aggregate['n'],
                             'coverage90': aggregate['coverage90'][j],
                             'coverage_Wilson_low': interval[0], 'coverage_Wilson_high': interval[1],
                             'RMSE': aggregate['rmse'][j], 'mean_interval_width': aggregate['mean_width90'][j]})
        if calibration:
            for method, aggregate in calibration['aggregate'].items():
                append_calibration(method, aggregate)
        if ssm_calibration:
            accepted = [r for r in ssm_calibration['records'] if r.get('diagnostics', {}).get('converged', False)]
            attempted = ssm_calibration.get('n_attempted', len(ssm_calibration['records']))
            passed = ssm_calibration.get('n_passed', len(accepted))
            failed = ssm_calibration.get('n_failed', attempted - passed)
            display(pd.Series({'state_space_attempted': attempted, 'sampling_passed': passed,
                               'sampling_failed_or_unresolved': failed}))
            append_calibration('state_space_passed_only', ssm_calibration['aggregate'])
            print(ssm_calibration.get('aggregation_note', 'Primary state-space metrics condition on passing diagnostics.'))
            if failed:
                print('Passing fits are a selected subset; all_fits_exploratory is not accepted calibration evidence.')
            unresolved = ssm_calibration.get('coverage90_unresolved_bounds')
            if unresolved is not None:
                display(pd.DataFrame(unresolved, index=PARAMETER_NAMES,
                                     columns=['empirical_coverage_if_all_unresolved_fail',
                                              'empirical_coverage_if_all_unresolved_pass']))
                print('Unresolved bounds describe possible outcomes of unfinished/failed fits; they are not confidence intervals.')
        if rows:
            calibration_table = pd.DataFrame(rows)
            display(calibration_table.round(3))
        else:
            print('No independent calibration artifact yet. Run the evaluate/state-space stages.')
        '''),
        code('''
        if calibration:
            fig, axes = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
            records = calibration['records']['neural']
            ranks = np.array([r['rank_fraction'] for r in records])
            for ax, j, label in zip(axes, (3,4,5), ('Amplitude effect', 'Persistence effect', 'Jitter effect')):
                ax.hist(ranks[:, j], bins=np.linspace(0,1,11), color='#167d8d', alpha=.8)
                ax.axhline(len(ranks)/10, color='grey', ls=':')
                ax.set(title=label, xlabel='SBC fractional rank', ylabel='Independent movies')
            show(fig)
        else:
            print('SBC plot awaits independent prior-predictive simulations.')
        '''),
        md('''
        Uniform SBC ranks are a necessary diagnostic, not proof of correctness. The model
        may discard information and still calibrate; coverage, width, recovery error, and
        predictive adequacy must be considered together. These finite simulation counts
        provide Monte Carlo estimates, not exact operating characteristics.
        '''),
        code('''
        scenario_rows = []
        if scenarios:
            for scenario, result in scenarios.items():
                for method, aggregate in result['aggregate'].items():
                    n = aggregate['n']
                    # Recompute intervals from independent per-movie decisions.
                    successes = sum(int(r['any_detected']) for r in result[method])
                    low, high = wilson_interval(successes, n)
                    scenario_rows.append({'scenario': scenario, 'method': method,
                                          'same_training_design': result['in_domain'], 'datasets': n,
                                          'any_direction_detected': successes/n,
                                          'Wilson_low': low, 'Wilson_high': high,
                                          'amplitude_detection': aggregate['detection_rate'][0],
                                          'persistence_detection': aggregate['detection_rate'][1],
                                          'jitter_detection': aggregate['detection_rate'][2]})
            display(pd.DataFrame(scenario_rows).round(3))
        else:
            print('Scenario replicates have not been generated.')
        '''),
        code('''
        if ssm_cases:
            rows = []
            for scenario, result in ssm_cases.items():
                d = result['diagnostics']
                rows.append({'scenario': scenario, 'Rhat_max': d['max_rhat'],
                             'ESS_bulk_min': d['min_ess_bulk'], 'ESS_tail_min': d['min_ess_tail'],
                             'sampling_pass': d['converged'],
                             'amplitude_detected': result['detected'][0],
                             'persistence_detected': result['detected'][1],
                             'jitter_detected': result['detected'][2]})
            display(pd.DataFrame(rows).round(3))
            print('These are representative fits. Repeated-dataset operating characteristics are reported separately below.')
        else:
            print('State-space example fits are not available yet.')
        '''),
        code('''
        repeated_rows = []
        if ssm_repeated:
            for scenario, checked in ssm_repeated.items():
                aggregate = checked.get('aggregate', {})
                attempted = checked.get('n_attempted', len(checked.get('records', [])))
                passed = checked.get('n_passed', aggregate.get('n', 0))
                failed = checked.get('n_failed', attempted - passed)
                bounds = checked.get('any_detection_unresolved_bounds', [np.nan, np.nan])
                wilson = aggregate.get('any_detection_wilson', [np.nan, np.nan])
                repeated_rows.append({'scenario': scenario, 'attempted': attempted, 'passed': passed,
                    'failed_or_unresolved': failed,
                    'any_detection_among_passed': aggregate.get('any_detection_rate', np.nan),
                    'Wilson_low_among_passed': wilson[0], 'Wilson_high_among_passed': wilson[1],
                    'overall_unresolved_lower': bounds[0], 'overall_unresolved_upper': bounds[1],
                    'amplitude_detection_among_passed': aggregate.get('detection_rate', [np.nan]*3)[0],
                    'persistence_detection_among_passed': aggregate.get('detection_rate', [np.nan]*3)[1],
                    'jitter_detection_among_passed': aggregate.get('detection_rate', [np.nan]*3)[2]})
            display(pd.DataFrame(repeated_rows).round(3))
            print('Rates condition on sampling success. Counts and unresolved-outcome bounds preserve failed fits in the assessment.')
            print('Stress-case detections are exploratory model-misspecification results; convergence does not validate the movement model.')
        else:
            print('Repeated state-space scenarios have not completed; do not infer rates from the representative-fit table.')
        '''),
        code('''
        reference = read_json('reference_validation.json')
        reference_rows = []
        if reference:
            for scenario, result in reference.items():
                for method, aggregate in result['aggregate'].items():
                    interval = aggregate['any_detection_wilson']
                    reference_rows.append({'independent_generator_case': scenario, 'method': method,
                        'attempted': aggregate['n'], 'passed': np.nan, 'failed': np.nan,
                        'any_detection': aggregate['any_detection_rate'],
                        'Wilson_low': interval[0], 'Wilson_high': interval[1]})
                checked = result.get('state_space_checked')
                if checked:
                    aggregate = checked['aggregate']
                    interval = aggregate.get('any_detection_wilson', [np.nan, np.nan])
                    reference_rows.append({'independent_generator_case': scenario,
                        'method': 'state_space_passed_only', 'attempted': checked['n_attempted'],
                        'passed': checked['n_passed'], 'failed': checked['n_failed'],
                        'any_detection': aggregate.get('any_detection_rate', np.nan),
                        'Wilson_low': interval[0], 'Wilson_high': interval[1]})
            display(pd.DataFrame(reference_rows).round(3))
            print('Sampling-pass counts apply only to MCMC; they are not applicable to ABC/neural draws.')
            print('The independent source ABM uses different polarity/Euler dynamics and initialization. These are raw transfer diagnostics, not accepted guarded classifications.')
        else:
            print('Independent source-ABM reference validation is not available yet.')
        '''),
        md('''
        **Read stress tests as model criticism.** The baseline bank does not train over
        spontaneous within-cell changes, adhesion, altered localization error, moving
        targets, or fine-step discretization. These deliberate departures test whether
        conclusions fail under plausible alternatives. In the noise stress, state-space
        inference is told the altered observation-error value whereas the frozen neural
        posterior retains its original training protocol; it is not a fully matched comparison.

        Mechanical confinement can change effective movement without changing the intrinsic
        OU coefficients. An inferred near-target effect in that scenario does not establish
        active target sensing. The framework must be expanded before making that claim.
        '''),
        code('''
        ppc_rows = []
        for scenario in ('null', 'combined', 'mechanical_null'):
            ppc = read_json(f'examples/{scenario}_ppc.json')
            if ppc:
                for method, result in ppc.items():
                    ppc_rows.append({'scenario': scenario, 'method': method,
                                     'fraction_summaries_outside90': result['fraction_features_outside90']})
        if ppc_rows:
            display(pd.DataFrame(ppc_rows).round(3))
            print('Summary discrepancies are correlated and reused for SBI fitting; this is not an independent omnibus test.')
        else:
            print('Posterior predictive checks await the ppc stage.')
        '''),
        md('''
        A useful next decision follows the observed failures:

        1. Require adequate within-cell near/far exposure and calibrated observation error.
        2. Use independent replication to establish which effects and magnitudes are recoverable.
        3. Fit or marginalize mechanical/spontaneous mechanisms when stress tests expose confounding.
        4. Validate a revised model on new simulations rather than repeatedly tuning to the same test set.
        5. Only then apply the frozen protocol to experimental trajectories.

        Synthetic recovery establishes what the inference can recover **under a stated model**;
        it cannot answer whether real killer cells change their migration around targets.
        The related [NK–target interaction study](https://www.nature.com/articles/s41598-021-94458-0)
        motivates target-independent controls and separate checks on directional interactions.
        ''')]


def main():
    out = ROOT / 'notebooks'
    out.mkdir(exist_ok=True)
    builders = {
        '01_simulator_and_question.ipynb': notebook1,
        '02_hierarchical_state_space.ipynb': notebook2,
        '03_simulation_based_inference.ipynb': notebook3,
        '04_validation_and_results.ipynb': notebook4,
    }
    for name, builder in builders.items():
        notebook = nbf.v4.new_notebook(cells=builder())
        notebook.metadata = {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.12'},
        }
        nbf.validate(notebook)
        nbf.write(notebook, out / name)
        print(out / name)


if __name__ == '__main__':
    main()
