# Synthetic benchmark commands

Run commands from the repository root. The scripts use `src/` directly, so an
editable package installation is optional. Install the dependencies declared in
`pyproject.toml` in your chosen Python environment.

1. `run_synthetic_benchmark.py --stage all` creates the simulation bank, trains
   the neural model, performs independent recovery/stress testing, fits the
   state-space model, tests the supplied ABM, extends unresolved MCMC runs, and
   performs posterior predictive checks.
2. `check_timestep.py` measures weak timestep sensitivity across 128 independent
   movies per setting.
3. `render_benchmark_report.py` generates the report, CSV tables and PNG/SVG
   figures from the recorded results.
4. `execute_notebooks.py` executes and saves the tutorial notebooks using a real
   Jupyter kernel. `build_notebooks.py` regenerates their clean source versions.

All published benchmark results are synthetic; no experimental trajectories
are used. Default output is `results/synthetic_v1/`. Seed partitions, priors,
software versions, source hashes, parameter truth and failed diagnostics are
preserved. The full run can take several minutes, mainly because MCMC quality
checks may trigger longer runs. Large neural training is disabled inside the
notebooks by default.
