#!/usr/bin/env python3
"""Additional, independently seeded synthetic SBI experiments for the manuscript.

The plan is written before any computation. Results are per trained network;
shared evaluation movies and shared training banks are never pooled as extra
independent replications. No experimental observations are used.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from motility_inference.simulator import SimConfig, simulate
from motility_inference.priors import PARAMETER_NAMES, PRIOR_LOW, PRIOR_HIGH, sample_prior, scenario_theta
from motility_inference.summaries import summarize
from motility_inference.sbi import train_posterior, NeuralPosterior
from motility_inference.evaluation import posterior_metrics, aggregate_metrics


def safe(x):
    if isinstance(x,dict): return {str(k):safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [safe(v) for v in x]
    if isinstance(x,np.ndarray): return safe(x.tolist())
    if isinstance(x,np.integer): return int(x)
    if isinstance(x,(float,np.floating)): return float(x) if np.isfinite(x) else None
    if isinstance(x,np.bool_): return bool(x)
    return x


def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(safe(obj),indent=2,allow_nan=False))
    tmp.replace(path)


def say(message): print(message,flush=True)


def plan():
    return {
        "study":"Follow-up synthetic validation informed by synthetic_v1; not preregistered",
        "created_at_utc":datetime.now(timezone.utc).isoformat(),
        "experimental_data":False,
        "configuration":asdict(SimConfig()),
        "prior_low":PRIOR_LOW,"prior_high":PRIOR_HIGH,"parameters":PARAMETER_NAMES,
        "fixed_training_bank":"results/synthetic_v1/training_bank.npz",
        "fixed_training_seeds":[117001,117002,117003],
        "primary_models":{"fixed":"fixed_117001","conditioned":"conditioned_127001","marginalized":"marginalized_127001","legacy":"frozen_v1"},
        "primary_selection_note":"First numeric seed designated primary before training/evaluation; other seeds are sensitivity analyses, without best-network selection.",
        "frozen_v1_network":"results/synthetic_v1/neural_posterior.pt",
        "matched_evaluation_counts":{"calibration":1024,"null":256,"varied_null":256,
            "slowdown":128,"decorrelation":128,"jitter":128,"combined":128,"weak":128},
        "paired_ssm_evaluation":{"calibration":{"n":96,"parameter_seed":710001,"simulation_start":711000},
            "fixed_null":{"n":96,"scenario":"null","simulation_start":721000},
            "combined":{"n":96,"scenario":"combined","simulation_start":731000},
            "varied_baseline_null":{"n":48,"parameter_seed":740001,"effects_zero":True,"simulation_start":741000}},
        "varied_null_definition":"First three parameters independently drawn from their prior; all three proximity effects fixed to zero; other nuisance settings fixed.",
        "noise_training":{"n":24000,"localization_sd_distribution":"log-uniform(0.05,0.7) micrometres",
            "theta_noise_independent":True,"training_seeds":[127001,127002,127003],
            "known_noise_context":"log(localization_sd) appended to 82 observed-only summaries",
            "ablation":"Three matched training-seed models use the identical bank but omit the known-noise context (marginalized noise)."},
        "noise_evaluation_counts":{"calibration_noise":1024,"null_sd015":128,"null_sd060":128,
            "combined_sd015":128,"combined_sd060":128},
        "posterior_draws":1500,
        "decision":"OR of three marginal probabilities >0.95: delta_speed<-log(1.2), delta_tau<-log(1.2), delta_jitter>log(1.2)",
        "intervals":"Marginal equal-tail 90% intervals, Wilson 95% intervals across independent movies",
        "training_hyperparameters":{"max_epochs":300,"hidden":128,"components":3,"patience":40,
            "batch_size":128,"learning_rate":0.0015,"validation_fraction":0.15},
        "seed_partitions":{"matched_simulation_start":1500000000,"noise_evaluation_start":1500100000,
            "matched_parameter_start":1600000000,"noise_parameter_start":1600100000,
            "noise_training_parameters":1700000000,"noise_training_levels":1700000001,
            "noise_training_simulation_start":1800000000},
        "analysis_limits":["Training-seed variability is conditional on the fixed simulation bank, not independent-bank uncertainty.",
            "All compared networks share test movies; never pool networks as independent datasets.",
            "Noise conditioning supplies known nuisance information and broader training support; it does not estimate unknown localization error.",
            "All results concern the specified simplified 2D proximity model; no claim of cellular mechanism is made."]}


def provenance(out):
    versions={}
    for package in ("numpy","scipy","torch","numba","pandas"):
        versions[package]=importlib.metadata.version(package)
    paths=list((ROOT/"src/motility_inference").glob("*.py"))+[Path(__file__).resolve()]
    write(out/"sbi_provenance.json",{"timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "python":platform.python_version(),"platform":platform.platform(),"versions":versions,
        "source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "v1_bank_sha256":hashlib.sha256((ROOT/"results/synthetic_v1/training_bank.npz").read_bytes()).hexdigest(),
        "v1_model_sha256":hashlib.sha256((ROOT/"results/synthetic_v1/neural_posterior.pt").read_bytes()).hexdigest()})


def observed_context(data,condition_noise=False):
    features=summarize(data)
    if condition_noise:
        features=np.r_[features,np.log(float(data["config"]["localization_sd"]))]
    return features


def validate_context(out):
    data=simulate(scenario_theta("combined"),SimConfig(),seed=1500200000)
    observed={k:data[k] for k in ("positions","targets","times","config")}
    np.testing.assert_array_equal(observed_context(data,True),observed_context(observed,True))
    changed=dict(data,theta=np.full(6,12345.),cell_offsets=np.full(12,-12345.),
                 true_positions=np.full_like(data["true_positions"],777.))
    np.testing.assert_array_equal(observed_context(data,True),observed_context(changed,True))
    assert observed_context(data,True).shape==(83,)
    write(out/"sbi_context_validation.json",{"checks":["observations-only context equals full synthetic context",
        "mutating theta, cell offsets and latent positions leaves context unchanged","83 features when known log noise appended"],
        "passed":True,"noise_source":"known synthetic acquisition metadata, not inferred truth parameter"})


def noise_bank(out):
    path=out/"sbi_noise_bank.npz"
    if path.exists(): return
    n=24000;theta=sample_prior(n,1700000000)
    noise=np.exp(np.random.default_rng(1700000001).uniform(np.log(.05),np.log(.7),n))
    x=np.empty((n,83));t0=time.perf_counter()
    for i in range(n):
        data=simulate(theta[i],replace(SimConfig(),localization_sd=float(noise[i])),seed=1800000000+i)
        x[i]=observed_context(data,True)
        if (i+1)%4000==0:say(f"Noise bank: {i+1}/{n} ({time.perf_counter()-t0:.1f}s)")
    np.savez_compressed(path,theta=theta,summaries=x,noise=noise,
        config_json=np.asarray(json.dumps(asdict(SimConfig()))))
    write(out/"sbi_noise_bank_metadata.json",{"n":n,"features":83,"seconds":time.perf_counter()-t0,
        "distribution":"independent prior theta and loguniform localization_sd(0.05,0.7)",
        "config_except_noise":asdict(SimConfig())})


def train_models(out):
    import torch
    torch.set_num_threads(2)
    records={}
    path=out/"sbi_training.json"
    if path.exists(): records=json.loads(path.read_text())
    for family,seeds,bank in (("fixed",[117001,117002,117003],ROOT/"results/synthetic_v1/training_bank.npz"),
                              ("conditioned",[127001,127002,127003],out/"sbi_noise_bank.npz"),
                              ("marginalized",[127001,127002,127003],out/"sbi_noise_bank.npz")):
        with np.load(bank,allow_pickle=False) as z: theta=z["theta"].copy();x=z["summaries"].copy()
        if family=="marginalized": x=x[:,:82]
        for seed in seeds:
            name=f"{family}_{seed}";checkpoint=out/f"sbi_{name}.pt"
            if checkpoint.exists() and name in records:continue
            say(f"Training {name}, {len(x)} movies × {x.shape[1]} context variables")
            t0=time.perf_counter()
            model=train_posterior(theta,x,seed,max_epochs=300,components=3,hidden=128,patience=40,progress=True)
            model.save(checkpoint)
            records[name]={"family":family,"seed":seed,"n":len(x),"features":x.shape[1],
                "seconds":time.perf_counter()-t0,"history":model.history,"checkpoint":checkpoint.name}
            write(path,records)


def evaluation_bank(out,kind):
    path=out/f"sbi_{kind}_test_bank.npz";meta_path=out/f"sbi_{kind}_test_bank_metadata.json"
    if path.exists() and meta_path.exists():
        with np.load(path,allow_pickle=False) as z: arrays={k:z[k].copy() for k in z.files}
        return arrays,json.loads(meta_path.read_text())
    specs=plan()["matched_evaluation_counts" if kind=="matched" else "noise_evaluation_counts"]
    if kind=="paired_ssm":
        specs={"calibration":96,"fixed_null":96,"combined":96,"varied_baseline_null":48}
    all_theta=[];all_x=[];all_noise=[];all_seeds=[];all_position_hashes=[];slices={};start=0;t0=time.perf_counter()
    for j,(name,n) in enumerate(specs.items()):
        pseed=(1600000000 if kind=="matched" else 1600100000)+j
        if kind=="paired_ssm":pseed=740001 if name=="varied_baseline_null" else 710001
        if name.startswith("calibration") or name in ("varied_null","varied_baseline_null"): truth=sample_prior(n,pseed)
        else:
            scenario=name.split("_sd")[0]
            if scenario=="fixed_null":scenario="null"
            truth=np.tile(scenario_theta(scenario),(n,1))
        if name in ("varied_null","varied_baseline_null"):truth[:,3:]=0.
        if name=="calibration_noise": levels=np.exp(np.random.default_rng(pseed+500).uniform(np.log(.05),np.log(.7),n))
        else:levels=np.full(n,.6 if name.endswith("sd060") else .15)
        seeds=(1500000000 if kind=="matched" else 1500100000)+start+np.arange(n)
        if kind=="paired_ssm":
            seeds={"calibration":711000,"fixed_null":721000,"combined":731000,"varied_baseline_null":741000}[name]+np.arange(n)
        x=[]
        for i in range(n):
            data=simulate(truth[i],replace(SimConfig(),localization_sd=float(levels[i])),seed=int(seeds[i]))
            x.append(observed_context(data,True))
            all_position_hashes.append(hashlib.sha256(data["positions"].tobytes()).hexdigest())
        all_theta.extend(truth);all_x.extend(x);all_noise.extend(levels);all_seeds.extend(seeds)
        slices[name]={"start":start,"stop":start+n,"n":n};start+=n
        say(f"Generated {kind}/{name}: {n} test movies")
    arrays={"theta":np.asarray(all_theta),"summaries":np.asarray(all_x),"noise":np.asarray(all_noise),"seeds":np.asarray(all_seeds),"positions_sha256":np.asarray(all_position_hashes)}
    metadata={"kind":kind,"scenarios":slices,"seconds":time.perf_counter()-t0,"n":start,"independent_of_training":True}
    np.savez_compressed(path,**arrays);write(meta_path,metadata)
    return arrays,metadata


def evaluate(out,kind):
    import torch
    torch.set_num_threads(2)
    arrays,meta=evaluation_bank(out,kind)
    model_paths={"frozen_v1":ROOT/"results/synthetic_v1/neural_posterior.pt"}
    if kind in ("matched","paired_ssm"):
        model_paths.update({f"fixed_{s}":out/f"sbi_fixed_{s}.pt" for s in (117001,117002,117003)})
    else:
        model_paths.update({f"{family}_{s}":out/f"sbi_{family}_{s}.pt" for family in ("conditioned","marginalized") for s in (127001,127002,127003)})
    result_path=out/f"sbi_{kind}_evaluation.json"
    results=json.loads(result_path.read_text()) if result_path.exists() else {
        "metadata":meta,"models":{},"note":"Each model has the same independent test movies. Never treat network-movie pairs as independent replicates; compare models using paired movies."}
    for name,path in model_paths.items():
        if name in results["models"] and len(results["models"][name]["scenarios"])==len(meta["scenarios"]):continue
        model=NeuralPosterior.load(path);dimension=len(model.summary_mean)
        results["models"].setdefault(name,{"checkpoint":str(path.relative_to(ROOT)),"context_dimension":dimension,"scenarios":{}})
        for scenario,bounds in meta["scenarios"].items():
            if scenario in results["models"][name]["scenarios"]:continue
            records=[];t0=time.perf_counter()
            for i in range(bounds["start"],bounds["stop"]):
                x=arrays["summaries"][i,:dimension]
                draws=model.sample(x,1500,seed=int(arrays["seeds"][i])+5000000)
                rec=posterior_metrics(draws,arrays["theta"][i]);rec.update(seed=int(arrays["seeds"][i]),localization_sd=float(arrays["noise"][i]))
                if "positions_sha256" in arrays:rec["positions_sha256"]=str(arrays["positions_sha256"][i])
                records.append(rec)
            result={"records":records,"aggregate":aggregate_metrics(records),"seconds":time.perf_counter()-t0}
            if scenario=="calibration_noise":
                result["noise_strata"]={}
                for label,low,high in (("low",.05,.15),("middle",.15,.35),("high",.35,.700001)):
                    subset=[r for r in records if low<=r["localization_sd"]<high]
                    result["noise_strata"][label]=aggregate_metrics(subset)
            results["models"][name]["scenarios"][scenario]=result;write(result_path,results)
            a=result["aggregate"]
            say(f"{name}/{scenario}: coverage={np.round(a['coverage90'],3)}, any={a['any_detection_rate']:.3f}")


def compact_summary(out):
    summary={"note":"Ranges across training seeds are descriptive, conditional on a common bank; they are not confidence intervals or additional dataset replications."}
    for kind in ("matched","noise","paired_ssm"):
        path=out/f"sbi_{kind}_evaluation.json"
        if not path.exists():continue
        d=json.loads(path.read_text());summary[kind]={name:{k:v["aggregate"] for k,v in m["scenarios"].items()} for name,m in d["models"].items()}
    write(out/"sbi_summary.json",summary)
    scenario_rows=[];parameter_rows=[]
    for kind,models in summary.items():
        if kind=="note":continue
        for name,scenarios in models.items():
            for scenario,a in scenarios.items():
                scenario_rows.append({"evaluation":kind,"model":name,"scenario":scenario,"n_independent_movies":a["n"],
                    "any_effect_count":round(a["n"]*a["any_detection_rate"]),"any_effect_rate":a["any_detection_rate"],
                    "any_effect_wilson_low":a["any_detection_wilson"][0],"any_effect_wilson_high":a["any_detection_wilson"][1],
                    **{f"detect_{p}":a["detection_rate"][j] for j,p in enumerate(PARAMETER_NAMES[3:])}})
                for j,param in enumerate(PARAMETER_NAMES):
                    parameter_rows.append({"evaluation":kind,"model":name,"scenario":scenario,"parameter":param,
                        "n_independent_movies":a["n"],"coverage90":a["coverage90"][j],
                        "coverage90_wilson_low":a["coverage90_wilson"][j][0],"coverage90_wilson_high":a["coverage90_wilson"][j][1],
                        "rmse":a["rmse"][j],"bias":a["bias"][j],"mean_width90":a["mean_width90"][j]})
    for filename,rows in (("sbi_scenario_summary.csv",scenario_rows),("sbi_parameter_summary.csv",parameter_rows)):
        with (out/filename).open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default=str(ROOT/"results/manuscript_v2"))
    parser.add_argument("--stage",choices=["all","plan","bank","train","matched","noise","paired_ssm","summary"],default="all")
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if not (out/"sbi_plan.json").exists():write(out/"sbi_plan.json",plan())
    provenance(out);validate_context(out)
    (out/f"sbi_provenance_{args.stage}.json").write_text((out/"sbi_provenance.json").read_text())
    for stage,func in (("bank",noise_bank),("train",train_models),("matched",lambda p:evaluate(p,"matched")),("noise",lambda p:evaluate(p,"noise")),("paired_ssm",lambda p:evaluate(p,"paired_ssm")),("summary",compact_summary)):
        if args.stage in ("all",stage):func(out)
    say(f"Completed stage {args.stage}: {out}")


if __name__=="__main__":main()
