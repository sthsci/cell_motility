# Literature evidence for the synthetic motility manuscript

Checked on **2026-10-28** by the Codex research agent. These records are
**assistant-checked; accountable human claim and bibliographic verification is
pending**. No source is represented as human-verified. The bibliography is
`research/paper_references.bib`. This is a focused evidence selection, not a
systematic review or a novelty certification.

Primary article pages, author manuscripts, or abstracts deposited by the
publishers in PubMed were opened and read. Nature/PMC intermittently returned
authentication or browser-check pages; where this happened, the record below
identifies the alternative successfully opened source. Search snippets alone
were not used to establish scientific support.

## Biological motivation and prior trajectory inference

| ID / citation key | Verified record and locator | Supported proposition and limit |
|---|---|---|
| E01 / Metzner2015 | Metzner et al. *Nat. Commun.* **6**, 7516 (2015), DOI 10.1038/ncomms8516. [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC4491834/), Abstract; Introduction; Results, Bayesian inference; Fig. 3. | Sequential Bayesian inference can recover evolving activity and persistence in an autoregressive walk, and was applied to tumour-cell trajectories. This is precedent for temporal motility inference, not evidence of a killer–target response. |
| E02 / Mark2018 | Mark et al. *Nat. Commun.* **9**, 1803 (2018), DOI 10.1038/s41467-018-04241-5. [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC5935699/), Abstract; Results, Bayesian updating and Inference of time-varying parameters. | Grid-based filtering/smoothing with parameter-evolution models can evaluate model evidence and compare low-dimensional temporal hypotheses. Our retrospective coefficient inference does not reproduce this algorithm or calculate its model evidence. |
| E03 / Metzner2021 | Metzner et al. *Sci. Rep.* **11**, 15031 (2021), DOI 10.1038/s41598-021-94458-0. [Opened publisher-deposited abstract](https://pubmed.ncbi.nlm.nih.gov/34294808/), Abstract; figure descriptions. | An encounter is not itself evidence of purposeful pursuit. The authors compare directional statistics with target-randomized references and report donor/target-dependent NK interactions. Their estimand is directed attraction/repulsion, distinct from local speed, persistence and jitter changes. |
| E04 / Olofsson2014 | Olofsson et al. *Front. Immunol.* **5**, 80 (2014), DOI 10.3389/fimmu.2014.00080. [Full text](https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2014.00080/full), Abstract; Results, Motile scanning and speed in contact; Discussion. | Human NK cells show heterogeneous migration/contact dynamics, including movement while conjugated and motile scanning. Contact is therefore not synonymous with zero movement. The study compares activation conditions; it does not validate the present stochastic model. |
| E05 / Frontiers2014Erratum | Frontiers Production Office. *Front. Immunol.* **5**, 451 (2014), DOI 10.3389/fimmu.2014.00451. [Correction](https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2014.00451/full), Reason for Erratum; corrected Results. | Several numerical speeds were damaged by typesetting in E04. The correction states that conclusions are unchanged. Use qualitative motivation, or check every imported number against this correction. |
| E06 / Olofsson2019 | Olofsson et al. *Sci. Rep.* **9**, 10672 (2019), DOI 10.1038/s41598-019-46958-3. [Opened publisher-deposited abstract](https://pubmed.ncbi.nlm.nih.gov/31337806/), Abstract. | Three-dimensional collagen microwells support prolonged tracking of NK–target interactions; NK trajectories show between-cell heterogeneity and alternation between movement and arrest. A 2D proximity benchmark is not a validated replacement for that physical setting. |

## Statistical foundations and validation

| ID / citation key | Verified record and locator | Supported proposition and limit |
|---|---|---|
| E07 / Johnson2008 | Johnson, London, Lea & Durban. *Ecology* **89**, 1208–1215 (2008), DOI 10.1890/07-1032.1. [Opened article PDF](https://www.whoi.edu/cms/files/JohnsonEtal_Ecology2008_57189.pdf), pp. 1208–1210, Abstract and Methods. [DOI metadata](https://pubmed.ncbi.nlm.nih.gov/18543615/). | Integrating an Ornstein–Uhlenbeck velocity process yields a continuous-time movement model amenable to state-space inference and measurement-error treatment. It is established movement methodology; cell-specific distance effects are the present adaptation. |
| E08 / Kalman1960 | Kalman. *J. Basic Eng.* **82**, 35–45 (1960), DOI 10.1115/1.3662552. [Opened article PDF](https://eceweb1.rutgers.edu/~gajic/pdffiles/519KalmanFiltering/kalman1960.pdf), Abstract and state-transition/filter derivation. | Recursive linear filtering supplies latent-state means and covariances. Exact Gaussian marginalization requires the specified linear-Gaussian conditional model; noisy position-derived distance remains an approximation in our likelihood. |
| E09 / Cranmer2020 | Cranmer, Brehmer & Louppe. *Proc. Natl Acad. Sci. USA* **117**, 30055–30062 (2020), DOI 10.1073/pnas.1912789117. [Opened author abstract](https://arxiv.org/abs/1911.01429), linked published DOI; publisher metadata available in [PMC record](https://pmc.ncbi.nlm.nih.gov/articles/PMC7720103/). | Simulation-based inference addresses inverse problems when simulation is available but likelihood evaluation is difficult. This review provides context, not performance evidence for this implementation. |
| E10 / Papamakarios2016 | Papamakarios & Murray. *Adv. Neural Inf. Process. Syst.* **29**, 1028–1036 (2016). [Official proceedings abstract](https://papers.neurips.cc/paper_files/paper/2016/hash/6aca97005c68f1206823815f66102863-Abstract.html), Abstract. | Conditional density estimation is an established likelihood-free posterior strategy; nonzero-tolerance ABC approximates a smoothed posterior. Our prior-bank mixture network is a custom implementation, not a claim of reproducing the full sequential algorithm. |
| E11 / TejeroCantero2020 | Tejero-Cantero et al. *J. Open Source Softw.* **5**, 2505 (2020), DOI 10.21105/joss.02505. [Official record](https://joss.theoj.org/papers/10.21105/joss.02505) and [opened paper](https://joss.theoj.org/papers/10.21105/joss.02505.pdf), Summary and Motivation. | The `sbi` toolkit makes neural simulation-based inference accessible. Cite only for the software ecosystem; the current custom PyTorch implementation does not use the distributed `sbi` package. |
| E12 / Lueckmann2021 | Lueckmann, Boelts, Greenberg, Gonçalves & Macke. *Proc. Mach. Learn. Res.* **130**, 343–351 (2021). [Official proceedings](https://proceedings.mlr.press/v130/lueckmann21a.html), Abstract and bibliographic metadata. | Performance depends on tasks and metrics; no single tested inference method is uniformly best. This motivates multiple diagnostics and comparators rather than a universal-superiority claim. |
| E13 / Talts2018 | Talts, Betancourt, Simpson, Vehtari & Gelman. *Preprint at arXiv* 1804.06788 (2018; v2 revised 2020), DOI 10.48550/arXiv.1804.06788. [Opened record](https://arxiv.org/abs/1804.06788), Abstract. | Simulation-based calibration checks Bayesian computation using simulation and posterior samples. In our paper, distinguish marginal prior-predictive diagnostics from fixed-parameter decision rates and from proof of model adequacy. |
| E14 / Vehtari2021 | Vehtari, Gelman, Simpson, Carpenter & Bürkner. *Bayesian Anal.* **16**, 667–718 (2021), DOI 10.1214/20-BA1221. [Opened author record](https://arxiv.org/abs/1903.08008), Abstract, related DOI, corrected v5 history. | Rank-based convergence diagnostics and quantile-specific efficiency measures address shortcomings of traditional R-hat. Passing numerical diagnostics does not validate the data-generating assumptions. |
| E15 / Ward2022 | Ward, Cannon, Beaumont, Fasiolo & Schmon. *Adv. Neural Inf. Process. Syst.* **35**, 33845–33859 (2022), DOI 10.52202/068431-2453. [Official proceedings abstract](https://papers.nips.cc/paper_files/paper/2022/hash/db0eac6747e3631eb91095cd76065611-Abstract-Conference.html), Abstract; page range from publisher's [proceedings record](https://www.proceedings.com/068431-2453.html). | Naive neural posterior estimation can be unreliable under misspecification; model criticism and explicit discrepancy modeling are remedies. The proposed robust method is not implemented here; cite as future methodological direction. |
| E16 / Morris2019 | Morris, White & Crowther. *Stat. Med.* **38**, 2074–2102 (2019), DOI 10.1002/sim.8086. [Opened publisher full text](https://onlinelibrary.wiley.com/doi/10.1002/sim.8086), Abstract; Sections 3–6 and Table 1. | Simulation studies should specify aims, generating mechanisms, estimands, methods and performance measures, and quantify Monte Carlo uncertainty. Follow-up design informed by initial results must be identified as such. |
| E17 / Sailynoja2022 | Säilynoja, Bürkner & Vehtari. *Stat. Comput.* **32**, 32 (2022), DOI 10.1007/s11222-022-10090-6. [Opened publisher full text](https://link.springer.com/article/10.1007/s11222-022-10090-6), Abstract; Sections 1.1–1.4, 2.1–2.2. | Discrete posterior ranks can be checked against uniformity using ECDF plots and simultaneous bands. Pointwise bin or ECDF intervals are not simultaneous envelopes; posterior autocorrelation changes finite-sample rank behavior. Cite as a diagnostic design reference, and describe the actual implemented bands rather than implying its specific algorithm was used. |

## Claim boundaries for the manuscript

1. The contribution is a reproducible synthetic feasibility and stress-testing
   framework for proximity-associated motility inference. Neither unprecedented
   Bayesian methodology nor a biological discovery has been established.
2. Metabolic, receptor-signaling, adhesion, cytotoxic or chemotactic mechanisms
   cannot be assigned from a fitted effective movement coefficient alone.
3. Present time-dependent plots are distance-response curves evaluated along
   trajectories. Unrestricted latent parameter dynamics and online parameter
   filtering are not part of the current implementation.
4. Numerical simulator agreement is implementation verification. Predictive
   recovery under the same generator, source-ABM transfer and experimental
   validation are distinct levels of evidence.
5. Per-effect detection and interval coverage must accompany any-effect rates;
   noise robustness must state which nuisance information each method receives.
6. Do not describe `sbi` or bayesloop as executed dependencies. Their papers
   establish relevant approaches, not the code used in these simulations.

The human authors must verify sources, author order, contributions, funding,
competing interests and data/code deposition before submission. A manuscript
draft with explicit placeholders can be completed without fabricating them.
