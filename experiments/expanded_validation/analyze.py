#!/usr/bin/env python3
"""Evaluate the frozen independent-seed test; never fit on new test outcomes."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
PREDICTORS = {
    "counterfactual_cost": "Paired recovery cost",
    "soc_ols": "Energy-deficit regression",
    "current_price": "Current-price estimate",
    "forecast_price": "Mean-price estimate",
}
COLORS = {"counterfactual_cost":"#0072B2", "soc_ols":"#777777", "current_price":"#D55E00", "forecast_price":"#009E73"}
REQUIRED = ["realized_extra_cost", "estimated_extra_cost", "soc_deficit_kwh", "current_price", "mean_forecast_price", "eta_charge"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def save_csv(data, name):
    pd.DataFrame(data).to_csv(HERE / name, index=False, float_format="%.17g")


def quantiles(values):
    return [float(x) for x in np.quantile(values, [.025, .975], method="linear")]


def summarize_predictions(frame, label, seed, draws=20000):
    assert len(frame) > 0
    indexes = np.random.default_rng(seed).integers(0, len(frame), size=(draws, len(frame)))
    metrics, paired = [], []
    proposed_abs = frame.absolute_error_counterfactual_cost.to_numpy(float)
    for predictor, name in PREDICTORS.items():
        error = frame[f"error_{predictor}"].to_numpy(float)
        absolute = abs(error)
        sampled = error[indexes]
        boot_mae = abs(sampled).mean(axis=1)
        boot_rmse = np.sqrt(np.square(sampled).mean(axis=1))
        mae_low, mae_high = quantiles(boot_mae)
        rmse_low, rmse_high = quantiles(boot_rmse)
        metrics.append({"group":label,"predictor":predictor,"label":name,"n_test":len(frame),
                        "mae":float(absolute.mean()), "mae_ci95_low":mae_low,"mae_ci95_high":mae_high,
                        "rmse":float(np.sqrt(np.square(error).mean())),"rmse_ci95_low":rmse_low,"rmse_ci95_high":rmse_high,
                        "median_absolute_error":float(np.median(absolute)),
                        "p90_absolute_error":float(np.quantile(absolute,.9)),
                        "p95_absolute_error":float(np.quantile(absolute,.95)),
                        "max_absolute_error":float(absolute.max()),"mean_signed_error":float(error.mean()),
                        "bootstrap_seed":seed,"bootstrap_draws":draws})
        if predictor != "counterfactual_cost":
            difference = absolute - proposed_abs
            low, high = quantiles(difference[indexes].mean(axis=1))
            paired.append({"group":label,"baseline":predictor,"reference":"counterfactual_cost","n_test":len(frame),
                           "mae_improvement_baseline_minus_reference":float(difference.mean()),
                           "ci95_low":low,"ci95_high":high,"bootstrap_seed":seed,"bootstrap_draws":draws})
    return metrics, paired


def save_figure(fig, name):
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", facecolor="white", metadata={"CreationDate":None})
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plots(predictions, metrics, strata, controllers):
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"serif", "font.serif":["DejaVu Serif"], "font.size":9,
                         "axes.titlesize":10,"axes.labelsize":9,"xtick.labelsize":8,"ytick.labelsize":8,
                         "legend.fontsize":8,"pdf.fonttype":42,"ps.fonttype":42})
    fig, axes = plt.subplots(1,2,figsize=(7.15,3.0),layout="constrained",gridspec_kw={"width_ratios":[1,1.17]})
    ax = axes[0]
    for noise, marker, color in [(0.,"o","#888888"),(.05,"s","#0072B2"),(.12,"^","#D55E00")]:
        part = predictions[predictions.noise_scale == noise]
        ax.scatter(part.realized_extra_cost, part.prediction_counterfactual_cost,s=17,marker=marker,c=color,alpha=.63,
                   linewidths=0,label=f"Noise {noise:g} (n={len(part)})")
    bounds = [min(0.,float(predictions[["realized_extra_cost","prediction_counterfactual_cost"]].min().min())),
              float(predictions[["realized_extra_cost","prediction_counterfactual_cost"]].max().max())*1.04]
    ax.plot(bounds,bounds,"--",color=".3",lw=.8,zorder=0)
    ax.set(xlim=bounds,ylim=bounds,xlabel="Realized extra cost (USD)",ylabel="Estimated extra cost (USD)",
           title=f"(a) New-seed test: {len(predictions)} eligible pairs")
    ax.legend(loc="upper left",frameon=False,handletextpad=.2,borderpad=.1)
    ax.grid(alpha=.15)
    ax = axes[1]
    selected = metrics[metrics.group == "all_eligible"].set_index("predictor").loc[list(PREDICTORS)]
    for y, (key,row) in enumerate(selected.iterrows()):
        ax.errorbar(row.mae,y,xerr=[[row.mae-row.mae_ci95_low],[row.mae_ci95_high-row.mae]],
                    fmt="o",color=COLORS[key],capsize=3,ms=4,lw=1.4)
        ax.annotate(f"{row.mae:.4f}",(row.mae_ci95_high,y),xytext=(5,0),textcoords="offset points",va="center",fontsize=8)
    ax.set_yticks(range(4),["Paired recovery cost","Energy-deficit\nregression","Current-price\nestimate","Mean-price\nestimate"])
    ax.invert_yaxis(); ax.set_ylim(3.55,-.55)
    ax.set_xlim(0,float(selected.mae_ci95_high.max())*1.28)
    ax.set(xlabel="MAE with 95% bootstrap interval (USD)",title="(b) Frozen predictors; common test cases")
    ax.grid(axis="x",alpha=.15)
    save_figure(fig,"independent_prediction")

    fig,axes = plt.subplots(1,2,figsize=(7.15,3.),layout="constrained",gridspec_kw={"width_ratios":[1.1,1.]})
    ax = axes[0]
    groups=["noise_scale_zero","noise_scale_0.05","noise_scale_0.12"]
    x=np.arange(3)
    for j,(key,name) in enumerate(PREDICTORS.items()):
        part=metrics[metrics.predictor==key].set_index("group").loc[groups]
        offset=(j-1.5)*.16
        ax.errorbar(x+offset,part.mae,yerr=[part.mae-part.mae_ci95_low,part.mae_ci95_high-part.mae],
                    fmt="o",ms=3,color=COLORS[key],capsize=2,label=name)
    ax.set(xticks=x,xticklabels=["0","0.05","0.12"],xlabel="Load/PV noise scale",ylabel="MAE (USD)",title="(a) Prediction error by noise level")
    ax.set_ylim(-.04, 1.12)
    ax.legend(frameon=False,loc="upper left",fontsize=6.8,handletextpad=.3)
    ax.grid(axis="y",alpha=.15)
    ax=axes[1]
    s=strata[strata.factor=="noise_scale"].sort_values("value")
    bottom=np.zeros(3)
    assert int(s.other_exclusions.sum()) == 0, "A new exclusion category requires an explicit additional bar."
    for key,label,color in [("eligible","Eligible pairs","#0072B2"),("paired_feasible_but_failed","Feasible, recovery failed","#E69F00"),("not_paired_feasible","Not paired feasible","#999999")]:
        values=s[key].to_numpy()
        ax.bar(x,values,bottom=bottom,color=color,width=.63,label=label)
        for xx,v,bb in zip(x,values,bottom):
            if v>0: ax.text(xx,bb+v/2,str(v),ha="center",va="center",fontsize=8,color="white" if key=="eligible" else "black")
        bottom+=values
    ax.set(xticks=x,xticklabels=["0","0.05","0.12"],xlabel="Load/PV noise scale",ylabel="Number of scenarios",title="(b) All 500 cases, including exclusions")
    ax.set_ylim(0,float(bottom.max())*1.36)
    ax.legend(frameon=False,loc="upper left",fontsize=6.8)
    save_figure(fig,"noise_and_cohort")

    fig,axes=plt.subplots(1,2,figsize=(7.15,2.6),layout="constrained")
    names=["Attribution + MPC","Measured state + MPC","Software replay","Measured-state reset","Maximum charging","Safe stop"]
    colors=["#0072B2","#555555","#D55E00","#009E73","#CC79A7","#E69F00"]
    y=np.arange(6)
    for ax,column,title,xlabel in [(axes[0],"success_percent","(a) Recovery across all 500 cases","Successful recoveries (%)"),
                                  (axes[1],"mean_unserved_kwh","(b) Service consequence across all 500 cases","Mean unserved energy (kWh)")]:
        ax.barh(y,controllers[column],color=colors,height=.65)
        ax.set_yticks(y,names); ax.invert_yaxis(); ax.set(title=title,xlabel=xlabel)
        ax.grid(axis="x",alpha=.15)
    axes[0].set_xlim(0,105)
    save_figure(fig,"independent_recovery")


def main():
    protocol=json.loads((HERE / "protocol.json").read_text())
    assert digest(HERE / "protocol.json")== (HERE / "protocol.sha256").read_text().split()[0]
    metadata=json.loads((RESULTS / "run_metadata.json").read_text())
    assert metadata["status"]=="complete"
    assert digest(RESULTS / "scenario_config.json")==protocol["scenario_config_sha256"]
    ep=pd.read_csv(RESULTS / "episodes.csv")
    old=pd.read_csv(EXPERIMENTS / "results/main/episodes.csv")
    assert len(ep)==500 and ep.scenario_id.is_unique and ep.seed.is_unique and (ep.strength>0).all()
    assert set(ep.seed)==set(range(10000,10500)) and not set(ep.seed)&set(old.seed)
    assert digest(EXPERIMENTS / "results/main/episodes.csv")==protocol["prediction_training"]["source_sha256"]
    coef=np.array([protocol["prediction_training"]["intercept"],protocol["prediction_training"]["slope"]])
    train=old[old.scenario_id.isin(protocol["prediction_training"]["scenario_ids"])]
    np.testing.assert_allclose(coef,np.linalg.lstsq(np.column_stack([np.ones(len(train)),train.soc_deficit_kwh]),train.realized_extra_cost,rcond=None)[0],atol=1e-14,rtol=0)
    paired=ep.actual_feasible & ep.reference_feasible
    finite=np.isfinite(ep[REQUIRED]).all(axis=1) & (ep.eta_charge>0)
    success=ep.realized_paired_success
    eligible=paired & finite & success
    audit=ep[["scenario_id","seed","strength","delay_steps","grid_cap_kw","deadline_steps","noise_scale","tariff_scale","feasibility_class"]].copy()
    audit["paired_feasible"]=paired; audit["successful_pair"]=success; audit["finite_inputs"]=finite; audit["included"]=eligible
    audit["reason"]=np.select([eligible,~paired,paired&~success],["included","not_paired_feasible","paired_feasible_but_recovery_failed"],default="nonfinite_inputs")
    save_csv(audit,"cohort_audit.csv")
    strata=[]
    for factor in ["noise_scale","strength","delay_steps","grid_cap_kw","deadline_steps","tariff_scale"]:
        for value,index in ep.groupby(factor).groups.items():
            strata.append({"factor":factor,"value":float(value),"all_cases":len(index),"paired_feasible":int(paired.loc[index].sum()),
                           "eligible":int(eligible.loc[index].sum()),"not_paired_feasible":int((~paired.loc[index]).sum()),
                           "paired_feasible_but_failed":int((paired.loc[index]&~success.loc[index]).sum()),
                           "other_exclusions":int((paired.loc[index]&success.loc[index]&~finite.loc[index]).sum())})
    strata=pd.DataFrame(strata); save_csv(strata,"cohort_by_factor.csv")
    frame=ep.loc[eligible].copy()
    frame["prediction_counterfactual_cost"]=frame.estimated_extra_cost
    frame["prediction_soc_ols"]=coef[0]+coef[1]*frame.soc_deficit_kwh
    frame["prediction_current_price"]=frame.soc_deficit_kwh/frame.eta_charge*frame.current_price
    frame["prediction_forecast_price"]=frame.soc_deficit_kwh/frame.eta_charge*frame.mean_forecast_price
    for key in PREDICTORS:
        frame[f"error_{key}"]=frame[f"prediction_{key}"]-frame.realized_extra_cost
        frame[f"absolute_error_{key}"]=abs(frame[f"error_{key}"])
    save_csv(frame,"test_predictions.csv")
    groups={"all_eligible":frame,"noise_scale_positive":frame[frame.noise_scale>0],"noise_scale_zero":frame[frame.noise_scale==0],
            "noise_scale_0.05":frame[frame.noise_scale==.05],"noise_scale_0.12":frame[frame.noise_scale==.12]}
    rows,pairs=[],[]
    for offset,(label,subset) in enumerate(groups.items()):
        m,p=summarize_predictions(subset,label,protocol["bootstrap"]["seed"]+offset,protocol["bootstrap"]["draws"])
        rows.extend(m);pairs.extend(p)
    metrics=pd.DataFrame(rows);paired_metrics=pd.DataFrame(pairs)
    save_csv(metrics,"prediction_metrics.csv");save_csv(paired_metrics,"paired_mae_improvements.csv")
    ctrl=pd.read_csv(RESULTS / "controllers.csv")
    method_order=["attribution_mpc","state_mpc","software_replay","reset_measured","max_charge","safe_stop"]
    controller_rows=[]
    for method in method_order:
        g=ctrl[ctrl.method==method]
        controller_rows.append({"method":method,"n_scenarios":len(g),"successful":int(g.physical_recovery_success.sum()),
                                "success_percent":float(g.physical_recovery_success.mean()*100),"mean_unserved_kwh":float(g.future_unserved_kwh.mean()),
                                "mean_target_shortfall_kwh":float(g.target_shortfall_kwh.mean()),
                                "mean_cumulative_solver_seconds_per_scenario":float(g.solver_seconds.mean())})
    controller_summary=pd.DataFrame(controller_rows);save_csv(controller_summary,"controller_summary.csv")
    traces=pd.read_csv(RESULTS / "traces.csv")
    base=traces[traces.method=="state_mpc"].sort_values(["scenario_id","t"]).reset_index(drop=True)
    equality={}
    for method in ["attribution_mpc","reset_measured"]:
        other=traces[traces.method==method].sort_values(["scenario_id","t"]).reset_index(drop=True)
        assert base[["scenario_id","t"]].equals(other[["scenario_id","t"]])
        columns=["requested_charge_kw","requested_discharge_kw","charge_kw","discharge_kw","energy_after_kwh","grid_import_kw","unserved_kw","step_cost"]
        diff=float(abs(base[columns].to_numpy()-other[columns].to_numpy()).max())
        assert diff < 1e-9
        equality[method]={"compared_actions":len(base),"max_absolute_difference_actions_states_costs":diff}
    original=json.loads((HERE / "original_assets_sha256.json").read_text())
    assert all(digest(EXPERIMENTS / path)==sha for path,sha in original.items())
    plots(frame,metrics,strata,controller_summary)
    summary={"status":"complete","analyzed_utc":datetime.now(timezone.utc).isoformat(),
             "study":"New profile seeds, fixed 500-case attacked test from the same synthetic generator",
             "protocol_sha256":digest(HERE / "protocol.json"),"cohort":{"all_new_attacked":len(ep),"paired_feasible":int(paired.sum()),
             "eligible":int(eligible.sum()),"not_paired_feasible":int((~paired).sum()),"paired_feasible_but_failed":int((paired&~success).sum()),
             "other_exclusions":int((paired&success&~finite).sum()),"eligible_noiseless":int((frame.noise_scale==0).sum()),
             "eligible_noisy":int((frame.noise_scale>0).sum()),"feasibility_counts":ep.feasibility_class.value_counts().to_dict()},
             "training":{"n":len(train),"intercept":float(coef[0]),"slope":float(coef[1]),"new_cases_used_for_fitting":0},
             "bootstrap":{**protocol["bootstrap"],"subgroup_seed_rule":"base seed + subgroup index in protocol order",
                "uncertainty_scope":"Conditional on fixed training coefficients and selected synthetic cohort; no refitting, no training/model selection/reference-state/model-mismatch/real-world uncertainty."},
             "prediction_metrics":rows,"paired_mae_improvements":pairs,"controller_summary":controller_rows,"action_equivalence":equality,
             "original_assets_unchanged":True,"new_AC_checks":0,
             "limitations":["New-seed same-generator synthetic test, not measured or out-of-distribution validation.",
                            "Prediction metrics are conditional on paired feasibility and successful actual/reference recovery; all 500 outcomes and exclusions are retained.",
                            "Matched physical models, known prices, exact physical observations and a clean same-controller reference favor the constrained model.",
                            "The energy agent is a deterministic MPC planning agent; no LLM, prompt interpretation or deployed attacker/detector is tested.",
                            "No new AC network validation; the 500 additional cases cannot inherit the earlier 100-case network claim.",
                            "Controller timing measures cumulative recovery solver time per scenario on this machine, not per-action latency."],
             "source_sha256":{str(p.relative_to(EXPERIMENTS)):digest(p) for p in [RESULTS/"episodes.csv",RESULTS/"controllers.csv",RESULTS/"traces.csv",HERE/"protocol.json",Path(__file__)]},
             "versions":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"matplotlib":matplotlib.__version__}}
    write_json(HERE / "summary.json",summary)
    print(json.dumps({"cohort":summary["cohort"],"metrics":rows[:4],"paired":pairs[:3]},indent=2))


if __name__=="__main__":
    main()
