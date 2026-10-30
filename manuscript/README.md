# Nature Communications-style manuscript package

Start with [the main manuscript](nature_communications/main.pdf) and [Supplementary Information](nature_communications_supplement/main.pdf). Both editable LaTeX projects use `main.tex` and a local `figures/` directory. Author and affiliation fields are blank as requested. This is a synthetic computational research draft, not an experimental claim or a submitted/accepted paper.

The article follows the checked Nature Communications initial-submission requirements: standard `article` class, Computer Modern text, single-column manuscript, numbered embedded references, short abstract, Results/Discussion/Methods, separate data and code statements, declarations, figure legends and tables. It does not imitate the production layout. The official-source record is [journal_requirements.md](../research/journal_requirements.md), checked 28 October 2026; the currently served formatting PDF itself was revised in 2021.

## Contents

- Main manuscript: detailed introduction, six Results subsections, Discussion, reproducible Methods, 14 verified references, 5 main figures and 1 table.
- Supplement: derivations, additional methods and limitations, 8 numerical tables and 8 figures.
- Figure formats: vector PDF, editable SVG and 400 dpi PNG, with a CSV source-data file for every figure.
- `numeric_claim_ledger.json`: numerical replacements derived from saved records and their source hashes.
- `format_audit.json` and `output_checks.json`: word/legend limits, compilation, font embedding, core tests and artifact checks.
- `../research/paper_evidence.md`: source locators and claims, marked assistant-checked with human verification pending.
- `../results/manuscript_v2`: plans, nine new neural checkpoints, simulation banks, per-movie evaluation records, 336 state-space fits, pairing checks and expanded results.

## What the added experiments establish

The expansion generated 24,000 broader-noise training movies and 4,048 independent evaluation movies, including 336 shared with the state-space study. The state-space analysis regenerated those exact 336 movies for fitting; they are not additional independent datasets. All 336 fits met the prespecified numerical diagnostics after saved retries. Four fixed-noise neural networks, including the original frozen model, evaluated the same held-out movies separately. Three conditioned and three marginalized noise networks share an expanded bank and evaluation ensembles; their outputs are not pooled as independent replication.

The frozen original neural model has 89.6–92.9% marginal 90% interval coverage over 1,024 new prior-predictive movies. State-space coverage is 89.6–94.8% over 96 new movies. Strong combined slowdown/jitter is recoverable, but persistence remains much weaker. Fixed-null state-space declarations are 0/96; varied-baseline-null declarations are 3/48, with undercoverage of baseline persistence in that subset. Marginal calibration is not conditional calibration.

The fixed-noise model flags 128/128 high-noise null movies. Broader-noise training reduces this to 3/128 for the primary marginalized model and 1/128 for the primary known-noise model. Most of the false-flag reduction comes from broader training; the supplied noise value further improves jitter precision. This is not a general guarantee against misspecification. Mechanical and spontaneous temporal variation remain attribution failures, and the selected predictive summaries can miss omitted mechanics.

## Reproduce

From the repository root, with the project dependencies installed:

```bash
python scripts/run_synthetic_benchmark.py --stage all
python scripts/check_timestep.py
python scripts/run_manuscript_sbi.py
python scripts/run_manuscript_ssm.py
python scripts/build_paper_figures.py
python scripts/build_manuscript.py
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd manuscript/nature_communications/main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd manuscript/nature_communications_supplement/main.tex
python scripts/check_paper_outputs.py
```

The expanded simulation scripts resume from their saved artifacts and preserve v1 results. See each script's `--help` and the immutable JSON plans for seeds and settings. Three state-space workers completed the recorded study in about 17 minutes on this machine; runtime depends on hardware and diagnostic retries. Scientific thresholds were not changed in response to results.

To rebuild only the paper from the included results, begin with `build_paper_figures.py`. Figure and manuscript builders read saved evidence rather than re-running inference. The manuscript builder embeds only cited references in first-citation order, so `main.tex` compiles without BibTeX. The two projects also compiled successfully using the Codex editor's built-in compiler.

The noise-conditioned research checkpoint expects 83 inputs. Follow [sbi_model_usage.md](../results/manuscript_v2/sbi_model_usage.md); the original application wrapper expects 82 and must not be used interchangeably. The original experimental-data limitations remain: 2D periodic geometry, noiseless target positions, fixed hierarchy/contact settings and no inferred physical adhesion mechanism.

## Before actual submission

Author names, affiliations, contributions, funding and competing interests require human completion. Public code/data archive identifiers and a license have not been invented. Scientific claims, bibliography and AI-use disclosure require human review. No manuscript or data have been submitted to a journal or publicly deposited. These outstanding author decisions are explicitly retained in the draft rather than disguised as finalized declarations.
