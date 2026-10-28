#!/usr/bin/env python3
"""Expanded synthetic SSM validation, prospectively specified after the v1 pilot.

The immutable plan is written before simulations. Every movie is an independent
replicate, all seeds differ from v1, and numerical retries never change the
scientific thresholds. Existing v1 artifacts are neither read as training data
nor modified. Restarting this script resumes completed movie-level records.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/cell_motility_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/cell_motility_cache")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from motility_inference.evaluation import aggregate_metrics, posterior_metrics, exposure_diagnostics, EFFECT_ROPE
from motility_inference.priors import PARAMETER_NAMES, PRIOR_LOW, PRIOR_HIGH, sample_prior, scenario_theta
from motility_inference.simulator import SimConfig, simulate

COUNTS = {"calibration": 96, "fixed_null": 96, "combined": 96, "varied_baseline_null": 48}
SIMULATION_BASES = {"calibration": 711000, "fixed_null": 721000, "combined": 731000, "varied_baseline_null": 741000}
SAMPLER_BASES = {"calibration": 811000, "fixed_null": 821000, "combined": 831000, "varied_baseline_null": 841000}
SCHEDULE = ((2000, 1500), (4000, 2000), (8000, 3000), (16000, 3000), (32000, 3000))


def safe(value):
    if isinstance(value, dict): return {str(k): safe(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [safe(v) for v in value]
    if isinstance(value, np.ndarray): return safe(value.tolist())
    if isinstance(value, np.bool_): return bool(value)
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, (np.floating,float)): return float(value) if np.isfinite(value) else None
    return value


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(safe(value), indent=2, allow_nan=False))
    temporary.replace(path)


def source_hashes():
    files = sorted((ROOT/"src"/"motility_inference").glob("*.py")) + [Path(__file__).resolve()]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def plan(config):
    versions = {}
    for package in ("numpy", "scipy", "numba", "arviz"):
        versions[package] = importlib.metadata.version(package)
    return {
        "study": "Expanded synthetic SSM validation for manuscript",
        "analysis_status": "Follow-up designed after inspecting the synthetic_v1 pilot; not preregistered externally.",
        "experimental_data": False,
        "counts": COUNTS,
        "independent_replication_unit": "Complete simulated movie; 12 killers and 201 recorded frames",
        "config": asdict(config),
        "parameter_names": PARAMETER_NAMES,
        "prior_low": PRIOR_LOW, "prior_high": PRIOR_HIGH,
        "theta_seed_calibration": 710001, "theta_seed_varied_baseline_null": 740001,
        "simulation_seed_bases": SIMULATION_BASES, "sampler_seed_bases": SAMPLER_BASES,
        "seed_rule": "Add the zero-based movie index to the relevant base; disjoint from v1 seed partitions.",
        "fixed_null_theta": scenario_theta("null"), "combined_theta": scenario_theta("combined"),
        "varied_null_design": "Draw all six parameters from the shared prior, then replace the three distance effects by zero; baseline parameters retain the prior's log-uniform marginals.",
        "known_nuisances": ["localization_sd", "heterogeneity_sd", "interaction_length", "contact_distance", "target_geometry"],
        "sampler_schedule_draws_warmup": SCHEDULE,
        "schedule_rule": "Rerun from fresh initialization with the same movie-specific sampler seed at the next prescribed length only if the numerical gate fails. Stop on first pass; retain all attempts.",
        "chains": 4,
        "numerical_gate": {"max_rank_rhat_strictly_less_than": 1.01, "minimum_bulk_ess": 400, "minimum_tail_ess": 400, "scope": "All global parameters and cell random effects"},
        "directional_decision": "For each effect independently, marginal posterior probability > 0.95 of delta_speed < -log(1.2), delta_tau < -log(1.2), or delta_jitter > log(1.2); any-effect is the OR of the three flags.",
        "effect_threshold_log_ratio": EFFECT_ROPE,
        "primary_outcomes": ["90% marginal interval coverage", "RMSE of posterior median", "bias", "90% interval width", "effect-specific and any-effect decision rates", "95% Wilson intervals across movies", "MCMC diagnostic failure rate"],
        "failure_handling": "Preserve every attempted movie. Primary posterior metrics use numerically passing fits; also retain all-fit exploratory metrics and worst/best bounds for unresolved outcomes. Never exclude based on exposure or scientific result.",
        "calibration_scope": "Marginal prior-predictive coverage, not a proof of joint calibration; MCMC rank fractions are descriptive, not independent exact SBC ranks.",
        "python": platform.python_version(), "platform": platform.platform(), "versions": versions,
        "source_sha256": source_hashes(),
    }


def checked_aggregate(records, planned):
    usable = [r for r in records if "covered90" in r]
    passed = [r for r in usable if r["diagnostics"]["converged"]]
    count = len(records); failed = count-len(passed)
    covered = np.sum([r["covered90"] for r in passed], axis=0) if passed else np.zeros(6)
    detected = sum(r["any_detected"] for r in passed)
    return {
        "n_planned": planned, "n_attempted": count, "n_passed": len(passed), "n_failed": failed,
        "complete": count == planned,
        "aggregate": aggregate_metrics(passed), "all_fits_exploratory": aggregate_metrics(usable),
        "coverage90_unresolved_bounds": np.stack([covered/max(count,1), (covered+failed)/max(count,1)],axis=1),
        "any_detection_unresolved_bounds": [detected/max(count,1), (detected+failed)/max(count,1)],
        "aggregation_note": "Primary metrics condition on passing numerical diagnostics; counts and unresolved bounds preserve failed attempts.",
        "total_sampler_seconds": sum(r.get("cumulative_sampler_seconds",0) for r in records),
        "records": sorted(records,key=lambda r:r["index"]),
    }


def run_one(task):
    from motility_inference.state_space import fit
    name, index, theta, config_dict = task
    config = SimConfig(**config_dict)
    seed = SIMULATION_BASES[name]+index
    sampler_seed = SAMPLER_BASES[name]+index
    began = time.perf_counter()
    history=[]
    try:
        data=simulate(theta,config,seed=seed)
        position_hash=hashlib.sha256(data["positions"].tobytes()).hexdigest()
        result=None
        for draws,warmup in SCHEDULE:
            result=fit(data,draws=draws,warmup=warmup,chains=4,seed=sampler_seed,
                       rhat_threshold=1.01,min_ess=400)
            history.append({"draws_per_chain":draws,"warmup_per_chain":warmup,
                            "elapsed_seconds":result["elapsed_seconds"],"diagnostics":result["diagnostics"]})
            if result["diagnostics"]["converged"]: break
        record=posterior_metrics(result["samples"],theta)
        record.update(index=index,seed=seed,sampler_seed=sampler_seed,
            positions_sha256=position_hash,diagnostics=result["diagnostics"],attempts=history,
            exposure=exposure_diagnostics(data),final_draws_per_chain=history[-1]["draws_per_chain"],
            cumulative_sampler_seconds=sum(a["elapsed_seconds"] for a in history),
            elapsed_seconds=time.perf_counter()-began)
    except Exception:
        record={"index":index,"seed":seed,"sampler_seed":sampler_seed,"truth":theta,
                "diagnostics":{"converged":False},"attempts":history,
                "exception":traceback.format_exc(),"elapsed_seconds":time.perf_counter()-began,
                "cumulative_sampler_seconds":sum(a["elapsed_seconds"] for a in history)}
    return name,safe(record)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default=str(ROOT/"results"/"manuscript_v2"))
    parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--plan-only",action="store_true")
    args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    config=SimConfig(box_size=180.)
    specification=safe(plan(config));plan_file=out/"ssm_plan.json"
    if plan_file.exists():
        previous=json.loads(plan_file.read_text())
        # Resumption may occur on another host; scientific design/source must agree.
        for key in ("counts","config","simulation_seed_bases","sampler_seed_bases","source_sha256","sampler_schedule_draws_warmup"):
            if previous[key] != specification[key]:
                raise ValueError(f"Existing immutable plan differs in {key}; choose a new output directory")
    else:
        specification["written_at_utc"]=datetime.now(timezone.utc).isoformat()
        write(plan_file,specification)
    print(f"Plan recorded: {plan_file}",flush=True)
    if args.plan_only:return
    theta={"calibration":sample_prior(COUNTS["calibration"],seed=710001),
           "fixed_null":np.tile(scenario_theta("null"),(COUNTS["fixed_null"],1)),
           "combined":np.tile(scenario_theta("combined"),(COUNTS["combined"],1)),
           "varied_baseline_null":sample_prior(COUNTS["varied_baseline_null"],seed=740001)}
    theta["varied_baseline_null"][:,3:]=0
    records={};tasks=[]
    for name,count in COUNTS.items():
        path=out/f"ssm_{name}.json"
        records[name]=json.loads(path.read_text())["records"] if path.exists() else []
        completed={r["index"] for r in records[name]}
        for i in range(count):
            if i not in completed: tasks.append((name,i,theta[name][i].tolist(),asdict(config)))
    began=time.perf_counter()
    context=multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=context) as pool:
        futures=[pool.submit(run_one,task) for task in tasks]
        for future in as_completed(futures):
            name,record=future.result()
            records[name].append(record)
            summary=checked_aggregate(records[name],COUNTS[name])
            write(out/f"ssm_{name}.json",summary)
            if len(records[name])%8==0 or not record["diagnostics"]["converged"]:
                print(f"{name}: {len(records[name])}/{COUNTS[name]}, passed {summary['n_passed']}; last draws {record.get('final_draws_per_chain','error')}",flush=True)
    runtime={"completed_at_utc":datetime.now(timezone.utc).isoformat(),"workers":args.workers,
             "wall_seconds_this_invocation":time.perf_counter()-began,
             "newly_processed_this_invocation":len(tasks),
             "counts":{name:{k:v for k,v in checked_aggregate(records[name],COUNTS[name]).items() if k in ("n_planned","n_attempted","n_passed","n_failed","complete","total_sampler_seconds")} for name in COUNTS}}
    write(out/"ssm_runtime.json",runtime)
    print(json.dumps(runtime,indent=2),flush=True)


if __name__=="__main__":main()
