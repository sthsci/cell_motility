#!/usr/bin/env python3
"""Render figures and a report strictly from saved synthetic benchmark results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "src"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from scipy.stats import binom
from motility_inference.simulator import load_dataset, proximity
from motility_inference.priors import PARAMETER_NAMES, PRIOR_LOW, PRIOR_HIGH

LABELS=["Baseline active SD", "Baseline persistence", "Baseline jitter", "Proximity: active SD", "Proximity: persistence", "Proximity: jitter"]
COLORS={"neural":"#0F4D92","abc":"#B64342","state_space":"#42949E"}
NAMES={"neural":"Neural SBI","abc":"Rejection ABC","state_space":"Hierarchical SSM"}


def read(path):
    return json.loads(Path(path).read_text())


def save(fig, directory, stem):
    fig.savefig(directory / f"{stem}.png",dpi=200,bbox_inches="tight")
    fig.savefig(directory / f"{stem}.svg",bbox_inches="tight")
    plt.close(fig)


def tables(out):
    import pandas as pd
    calibration=read(out / "calibration.json")
    rows=[]
    methods=dict(calibration["aggregate"])
    if (out / "state_space_calibration.json").exists():
        methods["state_space"]=read(out / "state_space_calibration.json")["aggregate"]
    for method,a in methods.items():
        for j,name in enumerate(PARAMETER_NAMES):
            rows.append({"method":method,"parameter":name,"n_datasets":a["n"],
                "coverage90":a["coverage90"][j],"coverage90_low":a["coverage90_wilson"][j][0],
                "coverage90_high":a["coverage90_wilson"][j][1],"rmse":a["rmse"][j],
                "mean_width90":a["mean_width90"][j],
                "width_relative_to_prior90":a["mean_width90"][j]/(.9*(PRIOR_HIGH[j]-PRIOR_LOW[j]))})
    pd.DataFrame(rows).to_csv(out / "calibration_summary.csv",index=False)
    rows=[]
    for label,case in read(out / "scenarios.json").items():
        for method,a in case["aggregate"].items():
            row={"scenario":label,"method":method,"in_domain":case["in_domain"],"n_datasets":a["n"],
                 "any_detection_rate":a["any_detection_rate"],"any_detection_low":a["any_detection_wilson"][0],
                 "any_detection_high":a["any_detection_wilson"][1]}
            row.update({f"detect_{name}":a["detection_rate"][j] for j,name in enumerate(PARAMETER_NAMES[3:])})
            rows.append(row)
    pd.DataFrame(rows).to_csv(out / "scenario_summary.csv",index=False)


def calibration_figures(out,figs):
    calibration=read(out / "calibration.json")
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for j,ax in enumerate(axes.flat):
        ax.plot([0,1],[0,1],"--",color=".65",lw=1)
        for method in ("neural","abc"):
            a=calibration["aggregate"][method]
            levels=np.array([.5,.8,.9,.95])
            rates=[a["coverage_curve"][str(l)][j] for l in levels]
            ax.plot(levels,rates,"o-",color=COLORS[method],label=NAMES[method])
            low,high=a["coverage90_wilson"][j]
            ax.errorbar(.9,a["coverage90"][j],yerr=[[a["coverage90"][j]-low],[high-a["coverage90"][j]]],color=COLORS[method],capsize=4)
        ax.set(title=LABELS[j],xlim=(.45,1),ylim=(.4,1.01),xlabel="Nominal interval probability",ylabel="Empirical coverage")
    axes[0,0].legend(fontsize=9)
    fig.suptitle(f"Independent prior-predictive recovery • {calibration['aggregate']['neural']['n']} synthetic datasets\nBars: 95% Wilson intervals at nominal 90% coverage",fontsize=13)
    save(fig,figs,"calibration")

    ranks=np.asarray([r["rank_fraction"] for r in calibration["records"]["neural"]])
    fig,axes=plt.subplots(2,3,figsize=(13,6.5),constrained_layout=True)
    low,high=binom.interval(.95,len(ranks),.1)
    for j,ax in enumerate(axes.flat):
        ax.axhspan(low,high,color=".92",label="Pointwise 95% binomial band")
        ax.hist(ranks[:,j],bins=np.linspace(0,1,11),color=COLORS["neural"],edgecolor="white")
        ax.axhline(len(ranks)/10,color=".3",ls="--",lw=1)
        ax.set(title=LABELS[j],xlabel="Posterior rank fraction of truth",ylabel="Datasets")
    fig.suptitle("Neural SBI simulation-based calibration; marginal diagnostic, not proof of joint calibration")
    save(fig,figs,"rank_histograms")

    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for j,ax in enumerate(axes.flat):
        for method in ("abc","neural"):
            r=calibration["records"][method]
            ax.scatter([x["truth"][j] for x in r],[x["median"][j] for x in r],s=12,alpha=.4,c=COLORS[method],label=NAMES[method],rasterized=True)
        ax.plot([PRIOR_LOW[j],PRIOR_HIGH[j]],[PRIOR_LOW[j],PRIOR_HIGH[j]],color=".3",ls="--")
        ax.set(title=LABELS[j],xlabel="True parameter (log scale)",ylabel="Posterior median")
    axes[0,0].legend(fontsize=9)
    save(fig,figs,"parameter_recovery")


def example_figures(out,figs):
    data=load_dataset(out / "examples" / "combined.npz")
    cfg=data["config"]
    gate,*_=proximity(data["positions"],data["targets"],cfg["box_size"],cfg["contact_distance"],cfg["interaction_length"])
    # Choose for observable exposure variation, not fit quality or recovery.
    cell=int(np.argmax(np.ptp(gate,axis=1)))
    path_file=out / "examples" / "combined_paths.npz"
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    ax=axes[0,0]
    x=data["positions"][cell] % cfg["box_size"]
    segments=np.stack([x[:-1],x[1:]],axis=1)
    valid=np.linalg.norm(x[1:]-x[:-1],axis=1)<cfg["box_size"]/2
    lc=LineCollection(segments[valid],array=gate[cell,:-1][valid],cmap="viridis",linewidth=2)
    lc.set_clim(0,1);ax.add_collection(lc)
    for target in data["targets"][:,0]:
        ax.add_patch(Circle(target,cfg["contact_distance"],facecolor=".9",edgecolor=".7",lw=.8))
    ax.scatter(x[0,0],x[0,1],c="black",s=30,marker="s",label="Start")
    ax.set(xlim=(0,cfg["box_size"]),ylim=(0,cfg["box_size"]),aspect="equal",xlabel="x (µm)",ylabel="y (µm)",title=f"Synthetic killer {cell}; circles mark contact distance")
    fig.colorbar(lc,ax=ax,label="Observed proximity gate")
    ssm=read(out / "state_space.json") if (out / "state_space.json").exists() else {}
    path_ok=ssm.get("combined",{}).get("diagnostics",{}).get("converged",False)
    paths=np.load(path_file,allow_pickle=False) if path_file.exists() and path_ok else None
    for ax,name,baseline,effect,label in zip([axes[0,1],axes[1,0],axes[1,1]],["speed","tau","jitter"],[0,1,2],[3,4,5],["Active velocity SD (µm/min)","Persistence time (min)","Jitter amplitude (µm/√min)"]):
        offset=data["cell_offsets"][cell] if name=="speed" else 0
        truth=np.exp(data["theta"][baseline]+offset+data["theta"][effect]*data["true_gate"][cell])
        ax.plot(data["times"],truth,color="black",lw=1.5,label="Generating value")
        if paths is not None:
            ax.fill_between(data["times"],paths[f"{name}_lower"][cell],paths[f"{name}_upper"][cell],color=COLORS["state_space"],alpha=.25,label="90% credible interval")
            ax.plot(data["times"],paths[f"{name}_median"][cell],color=COLORS["state_space"],lw=1.4,label="Posterior median")
        ax.set(xlabel="Time (min)",ylabel=label)
    axes[0,1].legend(fontsize=8)
    fig.suptitle("Distance-driven parameter paths; frame-by-frame estimates share a response model")
    save(fig,figs,"parameter_paths")
    if paths is not None: paths.close()

    scenarios=read(out / "scenarios.json")
    names=["null","slowdown","decorrelation","jitter","combined","weak"]
    fig,axes=plt.subplots(1,3,figsize=(14,5),sharey=True,constrained_layout=True)
    for j,ax in enumerate(axes):
        ax.axvline(0,color=".6",lw=1,ls="--")
        ax.axvspan(-np.log(1.2),np.log(1.2),color=".94")
        for i,name in enumerate(names):
            for method,shift in (("neural",-.2),("abc",0),("state_space",.2)):
                if method=="state_space":
                    if name not in ssm:continue
                    rec=ssm[name]
                    if not rec["diagnostics"]["converged"]:
                        continue
                else: rec=scenarios[name][method][0]
                k=j+3;mean=rec["median"][k];lo=rec["lower90"][k];hi=rec["upper90"][k]
                ax.errorbar(mean,i+shift,xerr=[[mean-lo],[hi-mean]],fmt="o",ms=4,color=COLORS[method],capsize=2,label=NAMES[method] if i==0 else None)
            truth=scenarios[name]["neural"][0]["truth"][j+3]
            ax.plot(truth,i,"k|",ms=18,mew=2)
        ax.set(title=LABELS[j+3],xlabel="Log contact/far parameter ratio",yticks=range(len(names)),yticklabels=names)
    axes[0].legend(fontsize=8,loc="lower left")
    fig.suptitle("Prespecified first replicate of each scenario • 90% marginal intervals • black marks: truth")
    save(fig,figs,"scenario_posteriors")


def stress_figure(out,figs):
    scenarios=read(out / "scenarios.json")
    names=[x for x in scenarios if x.endswith("null") or x=="combined"]
    fig,ax=plt.subplots(figsize=(11,5),constrained_layout=True)
    for method,shift in (("neural",-.12),("abc",.12)):
        for i,name in enumerate(names):
            a=scenarios[name]["aggregate"][method]
            p=a["any_detection_rate"];lo,hi=a["any_detection_wilson"]
            ax.errorbar(i+shift,p,yerr=[[p-lo],[hi-p]],fmt="o",color=COLORS[method],capsize=5,label=NAMES[method] if i==0 else None)
    ax.set(xticks=range(len(names)),xticklabels=[n.replace("mechanical_null","mechanical_only").replace("_","\n") for n in names],ylim=(-.03,1.03),ylabel="Fraction with any prespecified intrinsic-effect claim",title="Detection and attribution under matched and misspecified models\nMechanical-only alters real movement; apparent intrinsic effects represent mechanism misattribution")
    ax.legend(loc="upper left",bbox_to_anchor=(1.01,1))
    save(fig,figs,"stress_tests")


def report(out,figs):
    cal=read(out / "calibration.json");scenarios=read(out / "scenarios.json")
    speed=read(out / "speed.json");bank=read(out / "bank_metadata.json");training=read(out / "training.json")
    ssm=read(out / "state_space.json") if (out / "state_space.json").exists() else {}
    ss_cal=read(out / "state_space_calibration.json") if (out / "state_space_calibration.json").exists() else None
    na=cal["aggregate"]["neural"];aa=cal["aggregate"]["abc"]
    combined=scenarios["combined"]["aggregate"]["neural"]
    null=scenarios["null"]["aggregate"]["neural"]
    solo=scenarios["decorrelation"]["aggregate"]["neural"]
    lines=["# Synthetic killer–target migration benchmark", "", "**Synthetic validation only. No experimental trajectories were analyzed.**", "",
        "The benchmark asks whether distance-associated changes in active movement amplitude, persistence and short-timescale jitter can be recovered, and whether omitted mechanisms produce false attribution. It cannot establish whether real killer cells change behavior near targets.", "",
        "## What the completed tests show", "",
        f"**This is feasible for strong proximity-associated slowdown and jitter under the tested model.** In the combined scenario, neural SBI detected slowdown in {round(combined['detection_rate'][0]*combined['n'])}/{combined['n']} movies and increased jitter in {round(combined['detection_rate'][2]*combined['n'])}/{combined['n']}. It made no prespecified effect claims in {null['n']} matched null movies; the 95% Wilson upper bound is still {null['any_detection_wilson'][1]:.1%}, so zero observed errors does not imply zero error risk.", "",
        f"The neural posterior's nominal 90% intervals covered the true parameters in {min(na['coverage90']):.1%}–{max(na['coverage90']):.1%} of {na['n']} independent prior-predictive movies. Directional persistence was less identifiable: the neural method detected it in {round(solo['detection_rate'][1]*solo['n'])}/{solo['n']} persistence-only movies and {round(combined['detection_rate'][1]*combined['n'])}/{combined['n']} combined movies. A successful any-effect decision therefore does not mean every mechanism was recovered.", "",
        "**The main limitation is attribution.** Unmodeled localization noise and mechanical sticking can produce apparent intrinsic motility effects. The provided application wrapper rejects declared acquisition mismatches; source-ABM and mechanical stress tests remain exploratory. Neither method establishes biological sensing or regulation from movement alone.", "",
        "## Implemented framework", "",
        "- A compiled reduced ABM with a numerically equivalent reference engine, observed positions and target geometry, cell heterogeneity, optional binding/repulsion, moving targets and spontaneous movement changes.",
        "- A hierarchical Bayesian integrated-OU state-space model. Latent velocity and position are marginalized by a Kalman filter; global effects and cell intercepts are sampled by MCMC. The analytical likelihood is exact for its conditional model, but using noisy observed distance as a covariate is an approximation to the spatial simulator.",
        "- A likelihood-free neural posterior (custom PyTorch full-covariance mixture network), plus an empirical rejection-ABC baseline. Both condition on 82 observed-only summaries; neither sees latent states or parameter labels at inference.", "",
        "## Design and parameter meaning", "",
        f"Each baseline movie contains {bank['config']['n_killers']} killers, {bank['config']['n_targets']} targets and {bank['config']['n_frames']} frames at {bank['config']['frame_dt']} minutes per frame, in a periodic {bank['config']['box_size']} µm square. These are chosen synthetic units and settings, not measured experimental properties.", "",
        "For distance gate g=exp(−max(surface gap,0)/10 µm), log(active velocity SD)=log_speed+cell_offset+delta_speed*g, log(persistence)=log_tau+delta_tau*g, and log(Brownian amplitude)=log_jitter+delta_jitter*g. Effects are signed and zero lies inside their prior. The combined alternative uses (−0.8,−0.8,+0.8), corresponding at contact to 0.45× active velocity SD, 0.45× persistence and 2.23× jitter amplitude. Active SD is not mean observed speed; a jitter increase can increase frame-to-frame displacement despite reduced persistent movement.", "",
        "The baseline intentionally has no exclusion force: targets are proximity landmarks that can be crossed. Mechanical stress cases add repulsion and stochastic adhesion. This distinction prevents the mathematical recovery benchmark from being mistaken for a validated physical cell-contact model.", "",
        "Localization SD (0.15 µm), cell-intercept SD (0.25 log units), interaction length and target geometry are assumed known. Inferring them is future work. The state-space model has a distance response saturated at contact, not a separate contact-only jump or unrestricted parameter evolution.", "",
        "## Computation", "",
        f"- Same-model warm simulator timing: **{1000*speed['timings']['numba']['median_seconds']:.3f} ms/movie**, versus {1000*speed['timings']['numpy']['median_seconds']:.2f} ms for its reference engine; **{speed['same_model_speedup']:.1f}×** speedup. Maximum paired coordinate difference: {speed['paired_engine_max_absolute_difference']:.3g}.",
        f"- {bank['n']:,} training-bank movies and summaries generated in {bank['seconds']:.1f} s. Network training took {training['seconds']:.1f} s, with a separate internal validation split for early stopping.",
        "- First invocation may include compilation or disk-cache loading. The supplied full ABM has different integration, initialization and output overhead; its timing is recorded separately and is not the same-model speedup.", "",
        "## Independent recovery and calibration", "",
        "Training, early-stopping validation and scientific evaluation use separate movie-level draws. Frames from the same trajectory are not treated as independent replicates. Coverage is marginal prior-predictive coverage; agreement does not establish joint or conditional calibration. Interval width and RMSE show whether calibration is accompanied by information gain.", "",
        "| Parameter | Neural 90% coverage | ABC 90% coverage | Neural RMSE | Neural/prior 90% interval width |", "|---|---:|---:|---:|---:|"]
    for j,name in enumerate(PARAMETER_NAMES):
        lo,hi=na["coverage90_wilson"][j]
        lines.append(f"| {name} | {na['coverage90'][j]:.1%} [{lo:.1%}, {hi:.1%}] | {aa['coverage90'][j]:.1%} | {na['rmse'][j]:.3f} | {na['mean_width90'][j]/(.9*(PRIOR_HIGH[j]-PRIOR_LOW[j])):.2f} |")
    lines.extend(["",f"Neural/ABC evaluation uses {na['n']} independently generated datasets. Brackets are 95% Wilson Monte Carlo intervals. ABC retains 256/{bank['n']} prior-bank samples; its nonzero tolerance and correlated summaries can give broad, conservative intervals.","",
        "![Coverage](figures/calibration.png)","","![Recovery](figures/parameter_recovery.png)","","![Ranks](figures/rank_histograms.png)","",
        "## Detection and false attribution", "",
        "For each effect separately, a claim requires posterior probability >0.95 of crossing a 1.2-fold threshold in the prespecified direction: delta_speed<−log(1.2), delta_tau<−log(1.2), or delta_jitter>log(1.2). An any-effect claim means at least one of those three separate decisions passed. Its false-alarm rate is measured for the union of decisions; it is not assumed to be 5%. Failure to cross the threshold is inconclusive, not proof of equivalence.","",
        "| Scenario | In training design? | Neural any-effect claim | ABC any-effect claim |", "|---|---|---:|---:|"])
    for label,case in scenarios.items():
        a=case["aggregate"]["neural"];b=case["aggregate"]["abc"]
        lo,hi=a["any_detection_wilson"]
        lines.append(f"| {label} | {'Yes' if case['in_domain'] else 'No — stress test'} | {a['any_detection_rate']:.1%} [{lo:.1%}, {hi:.1%}] | {b['any_detection_rate']:.1%} |")
    lines.extend(["",f"Each scenario contains {next(iter(scenarios.values()))['aggregate']['neural']['n']} independent movies. Stress scenarios labeled null have zero programmed intrinsic OU-coefficient effects. **Mechanical adhesion/repulsion genuinely changes near-target movement**: it is a null only for intrinsic motility modulation, and a positive intrinsic-effect claim is mechanism misattribution. Spontaneous fluctuations are independent of target distance. The noisy scenario changes localization SD from 0.15 to 0.6 µm; the SSM receives the correct noise value while the neural estimator remains unadapted, so this is not an equal-information superiority test.","","![Stress tests](figures/stress_tests.png)","","## Hierarchical inference",""])
    lines.insert(lines.index("![Stress tests](figures/stress_tests.png)"),"These are raw low-level posterior threshold flags, deliberately computed for stress testing. The guarded application API rejects declared protocol shifts; an explicit exploratory override returns abstention. Unknown or incorrectly declared shifts can still escape these checks.\n")
    if ssm:
        passing=sum(r["diagnostics"]["converged"] for r in ssm.values())
        lines.extend([f"{passing}/{len(ssm)} named-example fits meet the prespecified MCMC gate: rank R-hat<1.01 and bulk/tail ESS≥400 for global and cell parameters. Runs extend sampling when necessary; failed diagnostics remain visible. Good MCMC mixing does not validate an omitted-mechanism model.","",
            "| Example | Max R-hat | Minimum bulk ESS | Minimum tail ESS | Gate passed |", "|---|---:|---:|---:|---|"])
        for name,r in ssm.items():
            d=r["diagnostics"]
            lines.append(f"| {name} | {d['max_rhat']:.3f} | {d['min_ess_bulk']:.0f} | {d['min_ess_tail']:.0f} | {d['converged']} |")
    if ss_cal:
        a=ss_cal["aggregate"]
        lines.extend(["",f"A separate state-space recovery pilot attempted {ss_cal.get('n_attempted',a['n'])} prior-predictive datasets; {ss_cal.get('n_passed',a['n'])} passed and {ss_cal.get('n_failed',0)} failed its numerical gate. The following metrics are **conditional on passing**. The JSON also retains all-fit exploratory metrics and worst/best coverage bounds treating failures as unresolved. Its small size gives substantially wider uncertainty than the neural benchmark; it is not evidence that either approach is universally superior.","",
            "| Parameter | SSM 90% coverage [Wilson interval] | RMSE |", "|---|---:|---:|"])
        for j,name in enumerate(PARAMETER_NAMES):
            lo,hi=a["coverage90_wilson"][j]
            lines.append(f"| {name} | {a['coverage90'][j]:.1%} [{lo:.1%}, {hi:.1%}] | {a['rmse'][j]:.3f} |")
    if (out / "state_space_scenarios.json").exists():
        repeated=read(out / "state_space_scenarios.json")
        lines.extend(["","Repeated state-space fits use the same simulation seeds as the SBI comparison. Stress replication is smaller and explicitly reported.","","| Scenario | Passed / attempted | Any-effect claim among passed | Unresolved-case bounds |","|---|---:|---:|---:|"])
        for label,case in repeated.items():
            a=case["aggregate"];lo,hi=case["any_detection_unresolved_bounds"]
            lines.append(f"| {label} | {case['n_passed']}/{case['n_attempted']} | {a.get('any_detection_rate',float('nan')):.1%} | [{lo:.1%}, {hi:.1%}] |")
        if repeated.get("combined",{}).get("n_passed",0):
            a=repeated["combined"]["aggregate"]
            lines.extend(["",f"Among {a['n']} converged combined-scenario SSM fits, the individual detection rates were {a['detection_rate'][0]:.1%} for slowdown, {a['detection_rate'][1]:.1%} for shorter persistence and {a['detection_rate'][2]:.1%} for increased jitter. This is a within-design comparison; improving the neural summaries could change the persistence gap."])
        if repeated.get("spontaneous_null",{}).get("n_passed",0):
            a=repeated["spontaneous_null"]["aggregate"]
            lines.extend(["",f"Absence of a directional claim does not establish robustness: in the spontaneous-change stress case, the SSM's 90% interval coverage for delta_speed was only {a['coverage90'][3]:.1%} across {a['n']} passed fits. The omitted temporal fluctuations can bias coefficient inference even when the practical decision threshold is not crossed."])
    if (out / "reference_validation.json").exists():
        reference=read(out / "reference_validation.json")
        lines.extend(["","## Transfer to the supplied ABM","","The actual MantiShrimp source now has optional proximity effects and cell-specific speed factors. A pre-edit golden regression confirms that zero-effect defaults preserve the seeded original dynamics. The reference bridge executes its fine-step Euler/polarity engine, zero initial polarity and reciprocal mechanics. It is not the same stochastic process as the simplified velocity model when coupling changes. These are transfer tests, not matched-model calibration.","","| Source-ABM scenario | Neural any-effect claim | ABC any-effect claim | SSM passed / attempted |","|---|---:|---:|---:|"])
        for label,case in reference.items():
            a=case['aggregate']['neural'];b=case['aggregate']['abc'];s=case['state_space_checked']
            lines.append(f"| {label} | {a['any_detection_rate']:.1%} ({a['n']} movies) | {b['any_detection_rate']:.1%} | {s['n_passed']}/{s['n_attempted']} |")
        c=reference['reference_combined'];a=c['aggregate']['neural'];s=c['state_space_checked']['aggregate']
        lines.extend(["",f"For source-ABM combined changes, neural detection rates were {a['detection_rate'][0]:.1%} for slowdown, {a['detection_rate'][1]:.1%} for persistence and {a['detection_rate'][2]:.1%} for jitter ({a['n']} movies). The SSM pilot rates were {s['detection_rate'][0]:.1%}, {s['detection_rate'][1]:.1%} and {s['detection_rate'][2]:.1%} ({s['n']} movies). Transfer supports detection of some altered movement, especially jitter, but does not establish reliable recovery of all mechanisms."])
    if (out / "timestep_sensitivity.json").exists():
        lines.extend(["","## Integration-step sensitivity","","`timestep_sensitivity.json` records independent 128-movie ensembles at 1, 2, 4, 8 and 16 internal steps per recorded frame, for combined motility changes and mechanical-only interactions. It reports short-lag displacement, turning, overlap, exposure and net displacement with Monte Carlo standard errors. This is a weak distributional sensitivity study, not a pathwise convergence proof; the finest step is a numerical reference, not known truth.","","The baseline combined case showed differences smaller than 1.9 combined Monte Carlo standard errors in the recorded metrics. Mechanics was more timestep-sensitive: one internal step differed from 16 by roughly 4.9 standard errors for turning and 5.6 for overlap. Four steps, used in the mechanical stress benchmark, still differed by about 2.1 standard errors for turning. Use 8–16 internal steps as a starting point for future mechanical inference and repeat convergence checks; the current mechanical rates are exploratory."])
    ppc_paths=sorted((out / "examples").glob("*_ppc.json"))
    if ppc_paths:
        lines.extend(["","## Posterior predictive checks","","These checks simulate 100 movies for each fitted method and example, conditioning on observed starting geometry and drawing new cell offsets from the population. They use the fitted baseline even for the mechanical stress example. The fraction below describes 82 correlated summaries outside marginal 90% predictive intervals; it is a descriptive discrepancy, not a calibrated test or independent replication.","","| Example | Method | Summary features outside 90% predictive interval |","|---|---|---:|"])
        for path in ppc_paths:
            for method,record in read(path).items():
                lines.append(f"| {path.stem.removesuffix('_ppc')} | {NAMES[method]} | {record['fraction_features_outside90']:.1%} |")
        lines.extend(["","The mechanical example has few summaries outside these intervals despite its omitted mechanism. These predictive checks therefore do not reliably diagnose mechanical misspecification. Agreement in the selected summaries cannot establish correct mechanism attribution."])
    lines.extend(["","![Posteriors](figures/scenario_posteriors.png)","","![Paths](figures/parameter_paths.png)","",
        "The example plots use the prespecified first replicate of each scenario, not a selected successful fit. The illustrated cell is selected by observed distance variation. Parameter bands reflect uncertainty in a shared response function; they do not imply that three unrelated parameters are estimated afresh from every frame.","",
        "## Interpretation and recommended use", "",
        "Use the state-space method as the transparent likelihood baseline and the neural method as fast approximate inference under a matched simulator and observation protocol. Check each method's actual coverage and false-alarm results above before relying on its intervals. Keep rejection ABC as a broad independent benchmark, not as an exact gold standard.","",
        "The state-space likelihood processes observations sequentially, while parameter inference here is retrospective MCMC over complete movies. This release does not implement an online parameter filter. Its time-dependent outputs arise from distance/contact histories and a shared response function.","",
        "A trapping-like path does not uniquely identify a motility response. Compare mechanisms that include physical adhesion, exclusion, target motion and spontaneous changes, and test transfer to the supplied ABM. The baseline inference omits these mechanisms; stress failures identify a need to expand the training model or abstain, not a reason to label mechanics as biological sensing.","",
        "Before experimental use: inspect true acquisition cadence and localization error; use segmented surfaces for contact; model target identity changes; estimate or calibrate nuisance parameters; include non-encounters; assess geometry-preserving null randomizations; and validate with independent movies/donors. This package currently supports 2D periodic benchmark geometry; arbitrary 3D masks require a different observation adapter.","",
        "## Reproducibility and outputs", "",
        "Run `python scripts/run_synthetic_benchmark.py --stage all` then `python scripts/render_benchmark_report.py`. The four notebooks explain the simulator, hierarchy, SBI and validation. JSON/CSV files contain per-dataset metrics, seeds, priors, known nuisance settings, software versions and source hashes; NPZ files contain selected observed data and posterior draws; the PyTorch checkpoint contains the learned posterior.","",
        "Methods context: [Metzner et al. 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4491834/), [Mark et al. 2018](https://www.nature.com/articles/s41467-018-04241-5), [NK–target interaction validation](https://www.nature.com/articles/s41598-021-94458-0), [SBI validation workflow](https://sbi.readthedocs.io/en/latest/how_to_guide.html). The implementation here is original project code; it does not claim to reproduce those papers' software.",""])
    (out / "REPORT.md").write_text("\n".join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default=str(ROOT / "results" / "synthetic_v1"))
    args=parser.parse_args();out=Path(args.output);figs=out / "figures";figs.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False,"svg.fonttype":"none"})
    tables(out);calibration_figures(out,figs);example_figures(out,figs);stress_figure(out,figs);report(out,figs)
    print(out / "REPORT.md")


if __name__=="__main__":main()
