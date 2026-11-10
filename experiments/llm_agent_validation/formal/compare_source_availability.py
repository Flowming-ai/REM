#!/usr/bin/env python3
"""Compare four conditions on the same twelve fixed profiles, without resampling."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent


def main():
    rows=[]; pairs_by={}; steps_by={}; traces_by={}
    for availability,root in [("available",HERE),("unavailable",HERE/"no_authenticated_source")]:
        pairs=pd.read_csv(root/"results/paired_episodes.csv")
        steps=pd.read_csv(root/"results/agent_steps.csv")
        traces=pd.read_csv(root/"results/traces.csv")
        assert pairs.seed.tolist()==list(range(24000,24012))
        pairs_by[availability],steps_by[availability],traces_by[availability]=pairs,steps,traces
        for condition,column in [("clean","clean"),("poisoned","poisoned")]:
            d=steps[steps.condition==condition]
            tt=traces[(traces.condition==condition)&(traces.phase=="post_detection")]
            costs=tt.groupby("seed").step_cost.sum()
            rows.append({"authenticated_source":availability,"bulletin":condition,"n_profile_seeds":12,
                         "decision_cycles":len(d),"real_llm_generations":int(d.generation_calls.sum()),
                         "executed_received":int((d.selected_source=="received").sum()),
                         "executed_verified":int((d.selected_source=="verified").sum()),
                         "model_hold":int((d.status=="model_hold").sum()),
                         "invalid_call_hold":int((d.status=="invalid_call_hold").sum()),
                         "mean_detection_energy_kwh":float(pairs[f"e_{column}_detect"].mean()),
                         "min_detection_energy_kwh":float(pairs[f"e_{column}_detect"].min()),
                         "max_detection_energy_kwh":float(pairs[f"e_{column}_detect"].max()),
                         "recovery_successes":int(pairs[f"{column}_recovery_success"].sum()),
                         "mean_future_unserved_kwh":float(pairs[f"{column}_future_unserved_kwh"].mean()),
                         "mean_future_purchase_cost_usd":float(costs.mean())})
    table=pd.DataFrame(rows)
    table.to_csv(HERE/"results/source_availability_comparison.csv",index=False)
    a,b=pairs_by["available"],pairs_by["unavailable"]
    difference=pd.DataFrame({"seed":a.seed,"deficit_with_auth_kwh":a.energy_deficit_kwh,
                            "deficit_without_auth_kwh":b.energy_deficit_kwh,
                            "estimated_extra_cost_with_auth":a.estimated_extra_cost,
                            "estimated_extra_cost_without_auth":b.estimated_extra_cost,
                            "realized_extra_cost_with_auth":a.realized_extra_cost,
                            "realized_extra_cost_without_auth":b.realized_extra_cost})
    difference.to_csv(HERE/"results/source_availability_paired.csv",index=False)
    warmcols=["seed","condition","t","energy_before_kwh","energy_after_kwh","requested_charge_kw","requested_discharge_kw"]
    wa=traces_by["available"].query("phase == 'mpc_warmup'")[warmcols].reset_index(drop=True)
    wb=traces_by["unavailable"].query("phase == 'mpc_warmup'")[warmcols].reset_index(drop=True)
    checks={"same_twelve_seeds":a.seed.equals(b.seed),"identical_paired_warmup":wa.equals(wb),
            "all_192_cycles":sum(len(s) for s in steps_by.values())==192,
            "unavailable_has_no_verified_executions":bool((steps_by["unavailable"].selected_source!="verified").all()),
            "no_invalid_call_fallbacks":bool((table.invalid_call_hold==0).all())}
    summary={"n_independent_profile_seeds":12,"n_conditions_per_seed":4,"n_physical_trajectories":48,
             "n_llm_cycles":192,"n_model_generations":int(table.real_llm_generations.sum()),"checks":checks,
             "conditions":rows,"available_divergent_pairs":int((a.energy_deficit_kwh.abs()>1e-6).sum()),
             "unavailable_divergent_pairs":int((b.energy_deficit_kwh.abs()>1e-6).sum()),
             "mean_extra_cost_with_auth_usd":float(a.estimated_extra_cost.mean()),
             "mean_extra_cost_without_auth_usd":float(b.estimated_extra_cost.mean()),
             "mean_deficit_with_auth_kwh":float(a.energy_deficit_kwh.mean()),
             "mean_deficit_without_auth_kwh":float(b.energy_deficit_kwh.mean()),
             "interpretation":"Source availability also changes its declared tool schema and availability statement. Same twelve profiles, not 24 independent samples; post-pilot mechanism ablation, one model and numeric data poisoning only."}
    attribution=[]
    for availability,frame in pairs_by.items():
        feasible=frame.actual_feasible & frame.reference_feasible
        eligible=feasible & frame.poisoned_recovery_success & frame.clean_recovery_success
        attribution.append({"authenticated_source":availability,"n_pairs":len(frame),"physically_affected_pairs":int((frame.energy_deficit_kwh.abs()>1e-6).sum()),
                            "initially_paired_feasible":int(feasible.sum()),"both_rolling_recoveries_successful":int(eligible.sum()),
                            "mean_clean_detection_kwh":float(frame.e_clean_detect.mean()),"mean_poisoned_detection_kwh":float(frame.e_poisoned_detect.mean()),
                            "mean_energy_deficit_kwh":float(frame.energy_deficit_kwh.mean()),"min_energy_deficit_kwh":float(frame.energy_deficit_kwh.min()),"max_energy_deficit_kwh":float(frame.energy_deficit_kwh.max()),
                            "mean_estimated_extra_cost_all_feasible_usd":float(frame.loc[feasible,"estimated_extra_cost"].mean()),
                            "mean_executed_cost_difference_all_pairs_usd":float(frame.realized_extra_cost.mean()),
                            "mean_estimated_extra_cost_successful_pairs_usd":float(frame.loc[eligible,"estimated_extra_cost"].mean()),
                            "mean_executed_cost_difference_successful_pairs_usd":float(frame.loc[eligible,"realized_extra_cost"].mean()),
                            "prediction_scope":"No LLM accuracy claim; successful-pair comparison conditional, zero-state pairs retained."})
    summary["attribution_by_source_availability"]=attribution
    pd.DataFrame(attribution).to_csv(HERE/"results/source_availability_attribution.csv",index=False)
    (HERE/"results/source_availability_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(2,2,figsize=(7.1,4.7),constrained_layout=True)
    colors=["#6e9ebf","#d78c49","#6e9ebf","#d78c49"]
    labels=["Available\nclean","Available\npoisoned","Unavailable\nclean","Unavailable\npoisoned"]
    for i,r in enumerate(rows):
        values=pairs_by[r["authenticated_source"]][f"e_{r['bulletin']}_detect"].to_numpy()
        jitter=np.linspace(-.14,.14,12)
        axes[0,0].scatter(i+jitter,values,s=15,color=colors[i],alpha=.8)
        axes[0,0].plot([i-.22,i+.22],[values.mean()]*2,color="black",lw=1.5)
    axes[0,0].set(xticks=np.arange(4),xticklabels=labels,ylabel="Detection energy (kWh)",title="(a) All 12 profiles; lines show means",ylim=(0,12.8))
    bottom=np.zeros(4)
    for key,label,color in [("executed_received","Received","#e5a16d"),("executed_verified","Verified","#73a697"),("model_hold","Hold","#a3a8af")]:
        values=table[key].to_numpy()
        axes[0,1].bar(np.arange(4),values,bottom=bottom,color=color,label=label)
        bottom+=values
    axes[0,1].set(xticks=np.arange(4),xticklabels=labels,ylabel="LLM-authorized cycles",title="(b) Actual tool-source choices",ylim=(0,58))
    axes[0,1].legend(ncol=3,loc="upper center",frameon=False,fontsize=7)
    for avail,color,marker,label in [("available","#397c99","o","Authenticated source available"),("unavailable","#d07534","s","Authenticated source unavailable")]:
        x=pairs_by[avail]
        axes[1,0].plot(np.arange(1,13),x.realized_extra_cost,marker=marker,color=color,lw=.9,label=label,markersize=3)
    axes[1,0].axhline(0,color="0.5",lw=.6)
    axes[1,0].set(xlabel="Preselected profile index",ylabel="Executed future cost difference (USD)",title="(c) Matched clean/poisoned pairs",xticks=np.arange(1,13))
    axes[1,0].legend(frameon=False,fontsize=6.5)
    for avail,condition,color,style,label in [("available","clean","#4785aa","-","Available / clean"),("available","poisoned","#498f76","--","Available / poisoned"),("unavailable","clean","#7c67a2",":","Unavailable / clean"),("unavailable","poisoned","#d07534","-","Unavailable / poisoned")]:
        t=traces_by[avail].query("seed == 24000 and condition == @condition").sort_values("t")
        axes[1,1].plot(np.r_[0,(t.t.to_numpy()+1)*.25],np.r_[t.energy_before_kwh.iloc[0],t.energy_after_kwh],color=color,ls=style,label=label)
    axes[1,1].axvspan(1,2,color="0.9",zorder=-2)
    axes[1,1].axvline(2,color="0.5",ls="--",lw=.7)
    axes[1,1].set(xlabel="Elapsed time (h)",ylabel="Battery energy (kWh)",title="(d) Preselected seed 24000")
    axes[1,1].legend(frameon=False,fontsize=6.5,ncol=2,loc="lower left")
    for ext in ["pdf","png","svg"]:
        fig.savefig(HERE/"figures"/f"llm_source_availability.{ext}",dpi=230)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.65),constrained_layout=True)
    jitter=np.linspace(-.13,.13,12)
    for ax,column,ylabel,title in [(axes[0],"energy_deficit_kwh","Detection energy deficit (kWh)","(a) Residual physical effect"),(axes[1],"estimated_extra_cost","Attributed future recovery cost (USD)","(b) Paired recovery-cost estimate")]:
        va,vb=a[column].to_numpy(),b[column].to_numpy()
        for j in range(12):
            ax.plot([jitter[j],1+jitter[j]],[va[j],vb[j]],color="0.8",lw=.7,zorder=1)
        for x,value,color in [(0,va,"#397c99"),(1,vb,"#d07534")]:
            ax.scatter(x+jitter,value,color=color,s=22,zorder=3)
            ax.plot([x-.2,x+.2],[value.mean()]*2,color="black",lw=1.8,zorder=4)
        ax.set(xticks=[0,1],xticklabels=["Authenticated source\navailable","Authenticated source\nunavailable"],ylabel=ylabel,title=title,xlim=(-.4,1.4))
        ax.axhline(0,color="0.6",ls=":",lw=.7,zorder=0)
        ax.set_ylim(bottom=-max(float(vb.max()),float(va.max()))*.06-.01)
    for ext in ["pdf","png","svg"]:
        fig.savefig(HERE/"figures"/f"llm_source_deficit_cost.{ext}",dpi=250)
    plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
