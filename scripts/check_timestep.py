#!/usr/bin/env python3
"""Weak timestep-sensitivity experiment using independent simulated movies."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "src"))
from motility_inference.simulator import SimConfig,simulate,proximity
from motility_inference.priors import scenario_theta


def metrics(data):
    x=data["positions"];cfg=data["config"]
    v=np.diff(x,axis=1)
    g,gap,*_=proximity(x,data["targets"],cfg["box_size"],cfg["contact_distance"],cfg["interaction_length"])
    denominator=np.sqrt(np.sum(v[:,:-1]**2,axis=-1)*np.sum(v[:,1:]**2,axis=-1))
    cosine=np.divide(np.sum(v[:,:-1]*v[:,1:],axis=-1),denominator,out=np.zeros_like(denominator),where=denominator>1e-12)
    return np.array([np.mean(np.sum(v**2,axis=-1)),np.mean(cosine),np.mean(gap<0),np.mean(g),np.mean(np.sum((x[:,-1]-x[:,0])**2,axis=-1))])


def main():
    out=ROOT / "results" / "synthetic_v1";out.mkdir(parents=True,exist_ok=True)
    config=SimConfig();records={}
    for index,name in enumerate(("combined","mechanical_only")):
        theta=scenario_theta("null" if name=="mechanical_only" else name)
        base=replace(config,repulsion=5,binding_rate=2,adhesion=5) if name=="mechanical_only" else config
        levels={}
        for substeps in (1,2,4,8,16):
            values=np.array([metrics(simulate(theta,replace(base,substeps=substeps),seed=92001+index*10000+substeps*1000+r)) for r in range(128)])
            levels[str(substeps)]={"n":len(values),"mean":values.mean(0).tolist(),"se":(values.std(0,ddof=1)/np.sqrt(len(values))).tolist()}
        reference=levels["16"]
        for k,v in levels.items():
            if k=="16":continue
            difference=np.subtract(v["mean"],reference["mean"])
            se=np.hypot(v["se"],reference["se"])
            v["difference_vs_finest"]=difference.tolist()
            v["difference_in_mc_se_units"]=(difference/se).tolist()
        records[name]=levels
    record={"metric_names":["mean_step_squared","mean_turn_cosine","overlap_fraction","mean_gate","net_displacement_squared"],"records":records,
        "interpretation":"Weak ensemble timestep sensitivity, not a pathwise error estimate. Independent datasets at each timestep; finest available step is a reference, not analytic truth. Pointwise Monte Carlo SEs do not establish equivalence."}
    (out / "timestep_sensitivity.json").write_text(json.dumps(record,indent=2))
    print(json.dumps({name:{s:v.get('difference_in_mc_se_units') for s,v in levels.items()} for name,levels in records.items()},indent=2))


if __name__=="__main__":main()
