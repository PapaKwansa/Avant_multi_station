#!/usr/bin/env python3
"""Stage 5D-2 repeated synthetic hydraulic inversion with correct/misspecified time noise.

Immutable Stage 4A inputs are hash checked. A NEW random realization is generated
per experiment repeat from the SAME forward model, at the original Stage 1 truth.
This is a statistics/identifiability benchmark, NOT COMSOL or field validation.
Only training stations/gauges are fitted; S07/S08 and P03 remain held out.
The 'corr_correct' and 'corr_ignored' fits use the EXACT SAME simulated data.

Run from AVANT repository root; see README_STAGE5D2_REPEATED_RECOVERY.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from avant_model.inversion.coupled_dream import sha256_file
from avant_model.inversion.coupled_parameters import HydraulicBounds
from avant_model.inversion.joint_dream import (load_stage4a, build_joint_likelihood,
                                             joint_mle_record, validate_joint_mle)
from avant_model.inversion.repeated_recovery import (
    ErrorSpec, synthetic_observations, fit_replicate, wilson_interval,
)

PARAMETERS=("log10_K", "log10_Ss")
MODES=("strain_only", "pressure_only", "joint")
CASES=("iid_correct", "corr_correct", "corr_ignored")


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage4a-dir", required=True)
    p.add_argument("--hydraulic-config", required=True)
    p.add_argument("--mechanical-config", required=True)
    p.add_argument("--gauges-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--replicates", type=int, default=8,
                   help="8 for smoke; >=100 suggested for meaningful empirical coverage")
    p.add_argument("--seed", type=int, default=20261025)
    p.add_argument("--correlation-hours", type=float, default=6.0)
    p.add_argument("--max-nfev", type=int, default=60)
    return p


def summarize(rows, expected_repeats):
    """Parameterwise aggregation; failed runs MUST NOT silently disappear."""
    df=pd.DataFrame(rows)
    output=[]
    for scenario in CASES:
        for mode in MODES:
            sub=df[(df.scenario==scenario)&(df.method==mode)]
            if len(sub)!=expected_repeats*len(PARAMETERS):
                raise RuntimeError(f"Missing experiment rows for {scenario}/{mode}")
            ok=sub[sub.success]
            for param in PARAMETERS:
                x=ok[ok.parameter==param]
                if len(x):
                    bias=float(x.error_log10.mean())
                    rmse=float(np.sqrt(np.mean(x.error_log10.to_numpy()**2)))
                    mean_sd=float(x.local_sd_log10.mean())
                    k=int(x.covers_truth.sum())
                    low,high=wilson_interval(k,len(x))
                    coverage=k/len(x)
                else:
                    bias=rmse=mean_sd=coverage=low=high=None
                output.append({"scenario":scenario,"method":mode,"parameter":param,
                               "replicates_requested":expected_repeats,
                               "fits_successful":len(x),
                               "fits_failed":expected_repeats-len(x),
                               "bias_log10":bias,"rmse_log10":rmse,
                               "mean_reported_local_sd_log10":mean_sd,
                               "empirical_wald_95pct_coverage":coverage,
                               "wilson_95pct_lower":low,"wilson_95pct_upper":high})
    return pd.DataFrame(output)


def plot_results(summary, out):
    fig,axes=plt.subplots(1,2,figsize=(11.2,4.8),constrained_layout=True)
    for j,param in enumerate(PARAMETERS):
        ax=axes[j]
        group=summary[summary.parameter==param]
        x=np.arange(len(CASES))
        for m,method in enumerate(MODES):
            sub=group[group.method==method].set_index("scenario").loc[list(CASES)]
            ax.bar(x+(m-1)*.25,sub.rmse_log10.to_numpy(float),.24,label=method)
        ax.set_xticks(x,["IID correct","Corr. correct","Corr. ignored"],rotation=15,ha="right")
        ax.set(ylabel="Empirical parameter RMSE (log10 units)",title=param)
        ax.grid(axis="y",alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Repeated synthetic parameter recovery; fixed mechanical/source geometry")
    for ext in ("png","pdf"):
        fig.savefig(out/f"recovery_rmse.{ext}",dpi=220)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11.2,4.8),constrained_layout=True)
    for j,param in enumerate(PARAMETERS):
        ax=axes[j]
        group=summary[summary.parameter==param]
        x=np.arange(len(CASES))
        for m,method in enumerate(MODES):
            sub=group[group.method==method].set_index("scenario").loc[list(CASES)]
            xx=x+(m-1)*.25
            heights=sub.empirical_wald_95pct_coverage.to_numpy(float)
            lower=sub.wilson_95pct_lower.to_numpy(float)
            upper=sub.wilson_95pct_upper.to_numpy(float)
            ax.bar(xx,heights,.24,label=method)
            if np.isfinite(heights).all():
                ax.errorbar(xx,heights,
                            yerr=np.vstack((heights-lower,upper-heights)),
                            fmt="none",ecolor="black",capsize=2,lw=.8)
        ax.axhline(.95,ls="--",color="black",lw=1,label="Nominal 95%")
        ax.set_xticks(x,["IID correct","Corr. correct","Corr. ignored"],rotation=15,ha="right")
        ax.set(ylabel="Empirical coverage of local 95% Wald intervals",title=param,ylim=(0,1.05))
        ax.grid(axis="y",alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Coverage is empirical; error bars are 95% Wilson binomial intervals")
    for ext in ("png","pdf"):
        fig.savefig(out/f"wald_coverage.{ext}",dpi=220)
    plt.close(fig)


def main(argv=None):
    a=parser().parse_args(argv)
    if a.replicates<1 or a.max_nfev<5 or not np.isfinite(a.correlation_hours) or a.correlation_hours<=0:
        raise ValueError("Require replicates>=1, max-nfev>=5, correlation-hours>0")
    folder, hc, mc, gc, out = (Path(x).resolve() for x in
        (a.stage4a_dir,a.hydraulic_config,a.mechanical_config,a.gauges_csv,a.output_dir))
    if out.exists():
        raise FileExistsError(f"Output folder exists: {out}")
    report,stage1,strain,pressure=load_stage4a(folder,hc,mc,gc)
    bounds=HydraulicBounds()
    joint,hyd_cfg=build_joint_likelihood(hc,mc,stage1,strain,pressure,bounds)
    start,_=validate_joint_mle(joint,report,bounds)
    truth=np.asarray(stage1["truth_log10_K_log10_Ss"],float)
    if not bounds.contains(truth):
        raise ValueError("Truth outside prior support")
    true_s,true_p=joint.predict(truth)
    times=strain.times_h
    if not np.array_equal(times,pressure.times_h):
        raise ValueError("Observation timestamps disagree")
    independent=ErrorSpec(strain.sigma_ns,pressure.sigma_pa)
    correlated=ErrorSpec(strain.sigma_ns,pressure.sigma_pa,
                         a.correlation_hours,a.correlation_hours)
    rng=np.random.default_rng(a.seed)
    lower=np.array([bounds.log10_K[0],bounds.log10_Ss[0]])
    upper=np.array([bounds.log10_K[1],bounds.log10_Ss[1]])
    records=[]
    saved=[]
    print(f"Reusing immutable Stage 4A forward model: {a.replicates} replicates, "
          f"{len(CASES)} error cases, {len(MODES)} fits each.",flush=True)
    for i in range(a.replicates):
        iid=synthetic_observations(true_s,true_p,times,independent,rng)
        corr=synthetic_observations(true_s,true_p,times,correlated,rng)
        saved.append((iid,corr))
        data={"iid_correct":(iid,independent),
              "corr_correct":(corr,correlated),
              "corr_ignored":(corr,independent)}
        for scenario,(observations,assumed) in data.items():
            for mode in MODES:
                try:
                    fit=fit_replicate(joint.predict,times,*observations,
                                      strain.training_mask,pressure.training_mask,
                                      assumed,start,lower,upper,mode=mode,max_nfev=a.max_nfev)
                    for j,name in enumerate(PARAMETERS):
                        estimate=float(fit["estimate_log10"][j])
                        records.append({"replicate":i,"scenario":scenario,"method":mode,
                                        "parameter":name,"success":True,"failure_reason":"",
                                        "truth_log10":float(truth[j]),"estimate_log10":estimate,
                                        "error_log10":estimate-float(truth[j]),
                                        "local_sd_log10":float(fit["local_sd_log10"][j]),
                                        "covers_truth":bool(fit["wald_95pct_low_log10"][j]<=truth[j]<=fit["wald_95pct_high_log10"][j]),
                                        "heldout_strain_rmse_ns":fit["heldout_strain_rmse_ns"],
                                        "heldout_pressure_rmse_pa":fit["heldout_pressure_rmse_pa"],
                                        "weighted_sum_sq":fit["weighted_sum_sq"],
                                        "nfev":fit["function_evaluations"]})
                except (RuntimeError,ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
                    for j,name in enumerate(PARAMETERS):
                        records.append({"replicate":i,"scenario":scenario,"method":mode,
                                        "parameter":name,"success":False,
                                        "failure_reason":f"{type(exc).__name__}: {exc}",
                                        "truth_log10":float(truth[j]),"estimate_log10":np.nan,
                                        "error_log10":np.nan,"local_sd_log10":np.nan,
                                        "covers_truth":False,"heldout_strain_rmse_ns":np.nan,
                                        "heldout_pressure_rmse_pa":np.nan,"weighted_sum_sq":np.nan,"nfev":np.nan})
        print(f"Finished replicate {i+1}/{a.replicates}",flush=True)
    summary=summarize(records,a.replicates)
    out.mkdir(parents=True)
    pd.DataFrame(records).to_csv(out/"replicate_fits.csv",index=False)
    summary.to_csv(out/"recovery_summary.csv",index=False)
    # Save exactly the observed arrays for auditing paired-error assumptions.
    np.savez_compressed(out/"synthetic_replicates.npz",
        times_h=times,truth_log10=truth,truth_strain_ns=true_s,truth_pressure_pa=true_p,
        iid_strain_ns=np.stack([v[0][0] for v in saved]),
        iid_pressure_pa=np.stack([v[0][1] for v in saved]),
        correlated_strain_ns=np.stack([v[1][0] for v in saved]),
        correlated_pressure_pa=np.stack([v[1][1] for v in saved]),
        strain_train_mask=strain.training_mask,pressure_train_mask=pressure.training_mask)
    plot_results(summary,out)
    info={
        "scope":"Stage 5D-2 same-model repeated synthetic two-parameter hydraulic recovery; no COMSOL/field validation, no DREAM",
        "replicates":int(a.replicates),"seed":int(a.seed),"times_h":times.tolist(),
        "truth_log10":truth.tolist(),"initialization_log10":start.tolist(),
        "sigma_strain_ns":float(strain.sigma_ns),"sigma_pressure_pa":float(pressure.sigma_pa),
        "simulated_correlation_hours":float(a.correlation_hours),
        "scenarios":{
            "iid_correct":"independent Gaussian noise, independent Gaussian fit",
            "corr_correct":"correlated Gaussian noise, matched correlated generalized least squares fit",
            "corr_ignored":"SAME correlated observations as corr_correct, but fit ignores time correlation",
        },
        "methods":list(MODES),
        "coverage_definition":"Fraction of simulated generating truths inside fitted conditional local 95% Wald confidence intervals; NOT posterior credible-interval coverage",
        "provenance":{
            "stage4a_summary_sha256":sha256_file(folder/"joint_summary.json"),
            "stage4a_pressure_observations_sha256":sha256_file(folder/"synthetic_pressure_observations.npz"),
            "stage1_observations_sha256":sha256_file(Path(report["source_dir"])/"synthetic_observations.npz"),
            "hydraulic_config_sha256":sha256_file(hc),
            "mechanical_config_sha256":sha256_file(mc),"gauges_csv_sha256":sha256_file(gc),
        },
        "limits":["Same homogeneous FV and fixed analytical geometry generate and invert every replicate.",
                  "Independent observation channels; correlation only through time within channel.",
                  "Wald intervals from local information at MLE, not DREAM or profile likelihood.",
                  "Small replicate counts have wide binomial uncertainty; use >=100 for stable coverage assessment.",
                  "S07/S08 and P03 are held out of parameter fitting.",
                  "Correctly modeled correlation does not guarantee exact 95% coverage in nonlinear small-sample inversion.",
                  "Parameter truth is 2-D and mechanics/geometry remain fixed; no global uniqueness or field validity established."],
    }
    (out/"repeated_recovery_manifest.json").write_text(json.dumps(info,indent=2),encoding="utf-8")
    print(summary.to_string(index=False),flush=True)
    print("Saved Stage 5D-2 repeated-recovery diagnostics:",out,flush=True)
    return out

if __name__=="__main__":
    main()
