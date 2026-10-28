#!/usr/bin/env python3
"""Reproducible synthetic-only comparison. Run from any working directory.

Stages are independently restartable; artifacts carry their generating design.
The neural estimator is a custom MDN, not the external `sbi` Python package.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
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
from motility_inference.simulator import SimConfig, simulate, save_dataset, load_dataset
from motility_inference.priors import PARAMETER_NAMES, PRIOR_LOW, PRIOR_HIGH, scenario_theta, sample_prior
from motility_inference.summaries import summarize
from motility_inference.sbi import generate_bank, train_posterior, NeuralPosterior, abc_samples
from motility_inference.evaluation import posterior_metrics, aggregate_metrics, exposure_diagnostics, EFFECT_ROPE


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False))


def say(message):
    print(message, flush=True)


def provenance(config, args):
    packages = {}
    for p in ("numpy", "scipy", "numba", "torch", "arviz", "pandas", "matplotlib", "nbformat", "nbclient"):
        try:
            packages[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            packages[p] = None
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted((ROOT / "src" / "motility_inference").glob("*.py"))}
    return {
        "study": "synthetic target-associated migration benchmark", "experimental_data": False,
        "config": asdict(config), "arguments": vars(args),
        "prior_low": PRIOR_LOW, "prior_high": PRIOR_HIGH, "parameter_names": PARAMETER_NAMES,
        "effect_rope": EFFECT_ROPE, "decision": "posterior P(prespecified directional effect beyond log(1.2)) > 0.95",
        "seed_partitions": {"training": 12001, "prior_test": 22001, "scenarios": 32001, "ssm_calibration": 42001, "ppc": 52001},
        "python": platform.python_version(), "platform": platform.system(),
        "versions": packages, "source_sha256": source_hashes,
    }


def speed_benchmark(out, config):
    theta = scenario_theta("null")
    start = time.perf_counter()
    compiled = simulate(theta, config, seed=10)
    first = time.perf_counter() - start
    reference = simulate(theta, config, seed=10, engine="numpy")
    error = float(np.max(np.abs(compiled["positions"] - reference["positions"])))
    timings = {}
    for engine, repeats in (("numpy", 5), ("numba", 100)):
        measured = []
        for r in range(repeats):
            t0 = time.perf_counter()
            simulate(theta, config, seed=100 + r, engine=engine)
            measured.append(time.perf_counter() - t0)
        timings[engine] = {"median_seconds": float(np.median(measured)), "repeats": repeats}
    record = {
        "first_call_seconds_in_this_process": first,
        "first_call_note": "May use an existing Numba disk cache; not a guaranteed clean compilation timing.",
        "timings": timings, "paired_engine_max_absolute_difference": error,
        "same_model_speedup": timings["numpy"]["median_seconds"] / timings["numba"]["median_seconds"],
    }
    # Broader pipeline timing is deliberately distinguished from identical-model speedup.
    from mantishrimp import SimulationConfig, DomainConfig, MotilityConfig, MechanicsConfig, CytotoxicityConfig, simulate as reference_simulate
    cfg = SimulationConfig(
        n_killers=config.n_killers, n_targets=config.n_targets,
        killer_radius=config.contact_distance / 2, target_radius=config.contact_distance / 2,
        duration=(config.n_frames - 1) * config.frame_dt, max_dt=config.frame_dt,
        max_drift_displacement=100000,
        domain=DomainConfig(0, config.box_size, 0, config.box_size, "periodic"),
        motility=MotilityConfig(killer_polarity_decay=np.exp(-theta[1]),
            target_polarity_decay=1., killer_polarity_noise=np.sqrt(2 * np.exp(-theta[1])) * np.exp(theta[0]),
            target_polarity_noise=0., killer_speed=1., target_speed=0.,
            killer_translation_noise=np.exp(theta[2]), target_translation_noise=0.),
        mechanics=MechanicsConfig(repulsion_kk=0,repulsion_tt=0,repulsion_kt=0,adhesion_kt=0,binding_rate=0),
        cytotoxicity=CytotoxicityConfig(killing_rate_max=0,exhaustion_rate_min=0,exhaustion_rate_max=0),
        seed=100,
    )
    t0 = time.perf_counter()
    result = reference_simulate(cfg)
    record["supplied_abm_seconds"] = time.perf_counter() - t0
    record["supplied_abm_rows"] = len(result.snapshots)
    record["supplied_abm_comparison_note"] = "Similar free-motion design, but different initialization/integrator/output; not a numerical-equivalence speedup."
    write_json(out / "speed.json", record)
    say(f"Equivalent reduced-model speedup: {record['same_model_speedup']:.1f}x; maximum difference {error:.3g}")


def build_bank(out, config, args):
    t0 = time.perf_counter()
    theta, features = generate_bank(args.n_train, config, seed=12001, progress=True)
    np.savez_compressed(out / "training_bank.npz", theta=theta, summaries=features,
                        config_json=np.asarray(json.dumps(asdict(config))))
    write_json(out / "bank_metadata.json", {"n":len(theta),"features":features.shape[1],
        "seconds":time.perf_counter()-t0,"seed":12001,"config":asdict(config)})


def read_bank(out, config):
    with np.load(out / "training_bank.npz", allow_pickle=False) as z:
        if json.loads(str(z["config_json"])) != asdict(config):
            raise ValueError("cached simulation bank uses a different design; choose another output directory")
        return z["theta"].copy(), z["summaries"].copy()


def fit_neural(out, config, args):
    import torch
    # This workflow uses small dense networks; cap CPU threads once before work.
    torch.set_num_threads(2)
    theta, features = read_bank(out, config)
    t0 = time.perf_counter()
    model = train_posterior(theta, features, seed=17001, max_epochs=args.epochs,
                            hidden=128, components=3, patience=40, progress=True)
    model.save(out / "neural_posterior.pt")
    write_json(out / "training.json", {"seconds":time.perf_counter()-t0,"history":model.history,
                                       "n":len(theta),"features":features.shape[1]})


def scenario_configs(config):
    # These are deliberate out-of-training-distribution checks, not evidence of robustness.
    return {
        "spontaneous_null": ("null", replace(config, spontaneous_sd=0.55)),
        "mechanical_null": ("null", replace(config, repulsion=5., binding_rate=2., adhesion=5., substeps=4)),
        "noisy_null": ("null", replace(config, localization_sd=0.6)),
        "fine_step_combined": ("combined", replace(config, substeps=4)),
        "moving_targets_combined": ("combined", replace(config, target_speed=0.7)),
    }


def aggregate_checked(records):
    """Keep failed MCMC runs unresolved rather than claiming valid intervals."""
    passed=[r for r in records if r["diagnostics"]["converged"]]
    n=len(records);failed=n-len(passed)
    covered=np.sum([r["covered90"] for r in passed],axis=0) if passed else np.zeros(6)
    detected=sum(r["any_detected"] for r in passed)
    return {"aggregate":aggregate_metrics(passed),"all_fits_exploratory":aggregate_metrics(records),
        "n_attempted":n,"n_passed":len(passed),"n_failed":failed,
        "coverage90_unresolved_bounds":np.stack([covered/max(n,1),(covered+failed)/max(n,1)],axis=1),
        "any_detection_unresolved_bounds":[detected/max(n,1),(detected+failed)/max(n,1)],
        "aggregation_note":"Primary metrics are conditional on passing MCMC diagnostics; failure counts and worst/best bounds remain visible.",
        "records":records}


def evaluate_sbi(out, config, args):
    model = NeuralPosterior.load(out / "neural_posterior.pt")
    bank_theta, bank_features = read_bank(out, config)
    prior_theta = sample_prior(args.n_test, seed=22001)
    records = {"neural": [], "abc": []}
    ranks = []
    for k, truth in enumerate(prior_theta):
        data = simulate(truth, config, seed=23001 + k)
        summary = summarize(data)
        posterior = model.sample(summary, args.posterior_draws, seed=24001 + k)
        abc = abc_samples(summary, bank_theta, bank_features, n_accept=min(256,len(bank_theta)))
        for method, samples in (("neural",posterior),("abc",abc["samples"])):
            rec = posterior_metrics(samples, truth)
            rec["seed"] = 23001+k
            rec["exposure"] = exposure_diagnostics(data)
            if method == "abc":
                rec.update(abc_epsilon=abc["epsilon"], abc_acceptance_fraction=abc["acceptance_fraction"])
            records[method].append(rec)
        ranks.append(records["neural"][-1]["rank_fraction"])
        if (k+1)%32 == 0:
            say(f"Independent prior-predictive datasets: {k+1}/{args.n_test}")
    write_json(out / "calibration.json", {"aggregate": {m:aggregate_metrics(r) for m,r in records.items()},"records":records})
    designs = {name:(name,config) for name in ("null","slowdown","decorrelation","jitter","combined","weak")}
    designs.update(scenario_configs(config))
    all_records = {}
    for idx,(label,(name,cfg)) in enumerate(designs.items()):
        case = {"neural": [], "abc": [], "design":asdict(cfg), "in_domain":cfg==config}
        truth = scenario_theta(name)
        for r in range(args.replicates):
            seed = 32001 + idx * 1000 + r
            data = simulate(truth,cfg,seed)
            summary = summarize(data)
            samples = model.sample(summary,args.posterior_draws,seed=seed+500000)
            abc = abc_samples(summary,bank_theta,bank_features,n_accept=min(256,len(bank_theta)))
            for method, draws in (("neural",samples),("abc",abc["samples"])):
                rec = posterior_metrics(draws,truth)
                rec.update(seed=seed,exposure=exposure_diagnostics(data))
                if method == "abc":
                    rec.update(abc_epsilon=abc["epsilon"], abc_acceptance_fraction=abc["acceptance_fraction"])
                case[method].append(rec)
            if r==0:
                save_dataset(data,out / "examples" / f"{label}.npz")
                np.savez_compressed(out / "examples" / f"{label}_sbi.npz",neural=samples,abc=abc["samples"])
                write_json(out / "examples" / f"{label}_support.json", model.input_diagnostics(summary))
        case["aggregate"] = {m:aggregate_metrics(case[m]) for m in ("neural","abc")}
        all_records[label] = case
        write_json(out / "scenarios.json",all_records)
        say(f"{label}: neural detection {case['aggregate']['neural']['detection_rate']}")


def state_space_suite(out, config, args):
    from motility_inference.state_space import fit, predict_parameter_paths
    def checked_fit(data, seed):
        draws, warmup = args.draws, args.warmup
        attempts = []
        while True:
            result = fit(data, draws=draws, warmup=warmup, chains=4, seed=seed)
            diag = result["diagnostics"]
            attempts.append({"draws": draws, "warmup":warmup,
                "max_rhat":diag["max_rhat"], "min_ess_bulk":diag["min_ess_bulk"],
                "min_ess_tail":diag["min_ess_tail"], "seconds":result["elapsed_seconds"]})
            # Fixed prespecified MCMC quality gate; simulation failures remain reported.
            passed = diag["max_rhat"] < 1.01 and min(diag["min_ess_bulk"],diag["min_ess_tail"]) >= 400
            result["diagnostics"]["converged"] = bool(passed)
            result["diagnostics"]["benchmark_attempts"] = attempts
            if passed or draws >= 8000:
                return result
            draws = min(8000, 2*draws)
            warmup = min(3000, 2*warmup)
            say(f"  Extending MCMC: Rhat={diag['max_rhat']:.3f}, minimum ESS={min(diag['min_ess_bulk'],diag['min_ess_tail']):.0f}")
    records = {}
    names = ["null","slowdown","decorrelation","jitter","combined","weak",
             "spontaneous_null","mechanical_null","noisy_null","fine_step_combined"]
    for k, name in enumerate(names):
        path = out / "examples" / f"{name}.npz"
        if not path.exists():
            raise ValueError("Run evaluate stage before state-space stage")
        data = load_dataset(path)
        say(f"State-space MCMC: {name}")
        result = checked_fit(data,seed=62001+k)
        np.savez_compressed(out / "examples" / f"{name}_ssm.npz", samples=result["samples"],
                            cell_samples=result["cell_samples"])
        rec = posterior_metrics(result["samples"], data["theta"])
        rec["diagnostics"] = result["diagnostics"]
        rec["elapsed_seconds"] = result["elapsed_seconds"]
        records[name] = rec
        paths = predict_parameter_paths(data,result)
        flat_paths = {}
        for key, value in paths.items():
            if isinstance(value, dict):
                flat_paths.update({f"{key}_{q}":array for q,array in value.items()})
            else:
                flat_paths[key] = value
        np.savez_compressed(out / "examples" / f"{name}_paths.npz", **flat_paths)
        write_json(out / "state_space.json",records)
        diag=result["diagnostics"]
        say(f"  Rhat={diag['max_rhat']:.3f}; bulk ESS={diag['min_ess_bulk']:.0f}; passed={diag['converged']}")
    calibration = []
    truths = sample_prior(args.ssm_test,seed=42001)
    for k,truth in enumerate(truths):
        data = simulate(truth,config,seed=43001+k)
        result = checked_fit(data,seed=44001+k)
        rec = posterior_metrics(result["samples"],truth)
        rec.update(seed=43001+k,diagnostics=result["diagnostics"],elapsed_seconds=result["elapsed_seconds"])
        calibration.append(rec)
        write_json(out / "state_space_calibration.json",aggregate_checked(calibration))
        say(f"State-space prior-predictive recovery: {k+1}/{args.ssm_test}")
    # Same seeds as the SBI cases; compare independent dataset-level decisions.
    repeated={}
    for name,index in (("null",0),("combined",4),("spontaneous_null",6),("mechanical_null",7)):
        theta=scenario_theta("combined" if name=="combined" else "null")
        cfg=scenario_configs(config)[name][1] if name.endswith("_null") else config
        count=args.replicates if name in ("null","combined") else min(8,args.replicates)
        cases=[]
        for r in range(count):
            seed=32001+index*1000+r
            if r==0:
                rec=records[name].copy()
            else:
                data=simulate(theta,cfg,seed)
                result=checked_fit(data,seed=72001+index*1000+r)
                rec=posterior_metrics(result["samples"],theta)
                rec.update(diagnostics=result["diagnostics"],elapsed_seconds=result["elapsed_seconds"])
            rec["seed"]=seed;cases.append(rec)
            repeated[name]=aggregate_checked(cases)
            write_json(out / "state_space_scenarios.json",repeated)
            if (r+1)%8==0:say(f"Repeated SSM {name}: {r+1}/{count}")


def reference_suite(out, config, args):
    from motility_inference.reference import simulate_reference
    from motility_inference.state_space import fit
    model=NeuralPosterior.load(out / "neural_posterior.pt")
    bank_theta,bank_features=read_bank(out,config)
    records={}
    designs={"reference_null":("null",config),"reference_combined":("combined",config),
             "reference_mechanics":("null",replace(config,repulsion=5.,binding_rate=2.,adhesion=5.))}
    for index,(label,(name,cfg)) in enumerate(designs.items()):
        case={"neural":[],"abc":[],"state_space":[],"design":asdict(cfg)}
        theta=scenario_theta(name)
        for r in range(min(16,args.replicates)):
            seed=82001+index*1000+r
            data=simulate_reference(theta,cfg,seed)
            summary=summarize(data)
            samples=model.sample(summary,args.posterior_draws,seed=seed+10000)
            abc=abc_samples(summary,bank_theta,bank_features,n_accept=min(256,len(bank_theta)))
            for method,draws in (("neural",samples),("abc",abc["samples"])):
                rec=posterior_metrics(draws,theta);rec["seed"]=seed
                case[method].append(rec)
            if r<4:
                result=fit(data,draws=8000,warmup=3000,chains=4,seed=seed+20000)
                rec=posterior_metrics(result["samples"],theta)
                rec.update(seed=seed,diagnostics=result["diagnostics"])
                case["state_space"].append(rec)
            if r==0:save_dataset(data,out / "examples" / f"{label}.npz")
        case["aggregate"]={m:aggregate_metrics(case[m]) for m in ("neural","abc")}
        case["state_space_checked"]=aggregate_checked(case["state_space"])
        records[label]=case;write_json(out / "reference_validation.json",records)
        say(f"{label}: neural detections {case['aggregate']['neural']['detection_rate']}")


def ppc(out, config, args):
    for label in ("null","combined","mechanical_null"):
        data=load_dataset(out / "examples" / f"{label}.npz")
        observed=summarize(data)
        collected={}
        for method,file,key in (("neural",f"{label}_sbi.npz","neural"),("state_space",f"{label}_ssm.npz","samples")):
            path=out / "examples" / file
            if not path.exists():
                continue
            with np.load(path,allow_pickle=False) as z:
                draws=z[key].reshape(-1,6)
            rng=np.random.default_rng(52001)
            summaries=[]
            for j,index in enumerate(rng.choice(len(draws),size=100,replace=False)):
                # Predict under the fitted baseline simulator, even for stress observations.
                pred=simulate(draws[index],config,seed=53001+j,
                              initial_positions=data["positions"][:,0],initial_targets=data["targets"][:,0])
                summaries.append(summarize(pred))
            summaries=np.asarray(summaries)
            low,high=np.quantile(summaries,[.05,.95],axis=0)
            collected[method]={"fraction_features_outside90":float(np.mean((observed<low)|(observed>high))),
                "observed":observed,"lower90":low,"upper90":high,"replicate_summaries":summaries,
                "note":"Geometry-conditioned posterior predictive; cell offsets redrawn from population. Feature discrepancies are correlated, not independent hypothesis tests."}
        write_json(out / "examples" / f"{label}_ppc.json",collected)


def refine_diagnostics(out, config, args):
    """Extend only failed chains, retaining their earlier diagnostics as history."""
    from motility_inference.state_space import fit,predict_parameter_paths
    from motility_inference.reference import simulate_reference
    def refine(rec,data,seed,example=None):
        if rec["diagnostics"]["converged"]:return rec
        history=[rec["diagnostics"]]
        result=None
        for draws in (16000,32000):
            say(f"Refining unresolved MCMC {example or seed}: {draws} draws/chain")
            result=fit(data,draws=draws,warmup=3000,chains=4,seed=seed)
            if result["diagnostics"]["converged"]:break
            history.append(result["diagnostics"])
        updated=posterior_metrics(result["samples"],data["theta"])
        updated.update(diagnostics=result["diagnostics"],elapsed_seconds=result["elapsed_seconds"],
                       refinement_history=history,seed=rec.get("seed",data["seed"]))
        if example:
            np.savez_compressed(out / "examples" / f"{example}_ssm.npz",samples=result["samples"],cell_samples=result["cell_samples"])
            paths=predict_parameter_paths(data,result)
            flat={}
            for k,v in paths.items():
                if isinstance(v,dict):flat.update({f"{k}_{q}":a for q,a in v.items()})
                else:flat[k]=v
            np.savez_compressed(out / "examples" / f"{example}_paths.npz",**flat)
        return updated
    path=out / "state_space.json"
    records=json.loads(path.read_text())
    for k,(name,rec) in enumerate(records.items()):
        records[name]=refine(rec,load_dataset(out / "examples" / f"{name}.npz"),62001+k,name)
        write_json(path,records)
    path=out / "state_space_calibration.json";cal=json.loads(path.read_text())
    for k,rec in enumerate(cal["records"]):
        if not rec["diagnostics"]["converged"]:
            data=simulate(rec["truth"],config,rec["seed"])
            cal["records"][k]=refine(rec,data,44001+k)
            write_json(path,aggregate_checked(cal["records"]))
    path=out / "state_space_scenarios.json";repeated=json.loads(path.read_text())
    for name,index in (("null",0),("combined",4),("spontaneous_null",6),("mechanical_null",7)):
        case=repeated[name]
        cfg=scenario_configs(config)[name][1] if name.endswith("_null") else config
        for r,rec in enumerate(case["records"]):
            if r==0:case["records"][r]=records[name]
            elif not rec["diagnostics"]["converged"]:
                data=simulate(rec["truth"],cfg,rec["seed"])
                case["records"][r]=refine(rec,data,72001+index*1000+r)
        repeated[name]=aggregate_checked(case["records"]);write_json(path,repeated)
    path=out / "reference_validation.json"
    if path.exists():
        reference=json.loads(path.read_text())
        for name,case in reference.items():
            for k,rec in enumerate(case["state_space"]):
                if not rec["diagnostics"]["converged"]:
                    data=simulate_reference(rec["truth"],SimConfig(**case["design"]),rec["seed"])
                    case["state_space"][k]=refine(rec,data,rec["seed"]+20000)
            case["state_space_checked"]=aggregate_checked(case["state_space"])
            write_json(path,reference)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default=str(ROOT / "results" / "synthetic_v1"))
    parser.add_argument("--stage",choices=["all","speed","bank","train","evaluate","state-space","reference","refine","ppc"],default="all")
    parser.add_argument("--n-train",type=int,default=16000)
    parser.add_argument("--n-test",type=int,default=256)
    parser.add_argument("--replicates",type=int,default=32)
    parser.add_argument("--epochs",type=int,default=300)
    parser.add_argument("--posterior-draws",type=int,default=1500)
    parser.add_argument("--draws",type=int,default=2000)
    parser.add_argument("--warmup",type=int,default=1500)
    parser.add_argument("--ssm-test",type=int,default=24)
    args=parser.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    config=SimConfig(box_size=180.)
    write_json(out / f"provenance_{args.stage}.json",provenance(config,args))
    stages={"speed":speed_benchmark,"bank":build_bank,"train":fit_neural,"evaluate":evaluate_sbi,"state-space":state_space_suite,"reference":reference_suite,"refine":refine_diagnostics,"ppc":ppc}
    for name,fn in stages.items():
        if args.stage in ("all",name):
            say(f"Starting {name}")
            fn(out,config) if name=="speed" else fn(out,config,args)
    say(f"Artifacts written to {out}")


if __name__=="__main__":
    main()
