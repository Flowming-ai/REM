#!/usr/bin/env python3
"""Validate all retained LLM episodes and draw vector scientific figures."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_validation import digest, write_json


def plot(out, pairs, steps, traces):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    figures = HERE / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.6), constrained_layout=True)
    for condition, label, color in [("clean", "Clean bulletin", "#2870a8"), ("poisoned", "Poisoned bulletin", "#d47a32")]:
        frame = traces[(traces.seed == 24000) & (traces.condition == condition)].sort_values("t")
        axes[0, 0].plot(np.r_[0, (frame.t.to_numpy()+1)*.25], np.r_[frame.energy_before_kwh.iloc[0], frame.energy_after_kwh], label=label, color=color)
        agent = steps[(steps.seed == 24000) & (steps.condition == condition)].sort_values("t")
        axes[0, 1].step(agent.t*.25, agent.requested_charge_kw-agent.requested_discharge_kw, where="post", label=label, color=color)
    axes[0, 0].axvspan(1, 2, color="#ededed", zorder=-2)
    axes[0, 0].axvline(2, color="0.4", ls="--", lw=.8)
    axes[0, 0].set(xlabel="Elapsed time (h)", ylabel="Battery energy (kWh)", title="(a) Preselected seed 24000")
    axes[0, 0].legend(fontsize=7, frameon=False)
    axes[0, 1].set(xlabel="Elapsed time (h)", ylabel="Requested net charging (kW)", title="(b) LLM-authorized actions", xticks=[1, 1.25, 1.5, 1.75], ylim=(-4.4, 4.4))
    x = np.arange(len(pairs))
    axes[1, 0].bar(x-.18, pairs.estimated_extra_cost, width=.36, label="Estimated", color="#6f99b7")
    axes[1, 0].bar(x+.18, pairs.realized_extra_cost, width=.36, label="Executed", color="#dcaa81")
    axes[1, 0].set(xlabel="Preselected pair (seed = 23999 + index)", ylabel="Future cost difference (USD)", title="(c) All 12 pairs", xticks=x, xticklabels=np.arange(1, len(pairs)+1))
    axes[1, 0].axhline(0, color="0.4", lw=.7)
    if np.nanmax(abs(pairs.realized_extra_cost.to_numpy())) < 1e-5:
        axes[1, 0].set_ylim(-.05, .05)
        axes[1, 0].text(.5, .65, "No cost divergence in these pairs", transform=axes[1, 0].transAxes, ha="center", fontsize=7)
    axes[1, 0].legend(frameon=False, fontsize=7)
    cats = ["received", "verified", "hold"]
    for off, condition, color in [(-.18, "clean", "#2870a8"), (.18, "poisoned", "#d47a32")]:
        z = steps[steps.condition == condition]
        values = [int((z.selected_source == c).sum()) if c != "hold" else int((z.status != "executed_plan").sum()) for c in cats]
        axes[1, 1].bar(np.arange(3)+off, values, width=.36, label=condition.capitalize(), color=color)
    axes[1, 1].set(xticks=np.arange(3), xticklabels=["Received plan", "Verified plan", "Hold/fallback"], ylabel="Decision cycles", title="(d) Actual model choices", ylim=(0, 53))
    axes[1, 1].legend(frameon=False, fontsize=7)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(figures / f"llm_agent_validation.{ext}", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.4, 3.1))
    ax.set_xlim(0, 11.4); ax.set_ylim(0, 3.1); ax.axis("off")
    blocks = [(0.1, .65, 1.8, 1.6, "Observations\nMeasured battery state\nLoad / PV\nExternal price bulletin", "#e8f1f8"),
              (2.3, .65, 1.9, 1.6, "Qwen2.5-3B-Instruct\nFresh context each step\nPlan or hold\nAuthorize / hold", "#e0eaf6"),
              (4.6, .65, 1.8, 1.6, "Trusted tools\nReceived bulletin only\nMPC plan_dispatch\nReturn plan ID + action", "#f8dfc9"),
              (6.8, .65, 1.8, 1.6, "Physical simulator\nExecute authorized plan\nClip to physical limits\nRecord actual transition", "#f7e7d9"),
              (9., .65, 2.15, 1.6, "At prescribed detection\nPaired measured states\nRecovery-cost attribution\nCommon MPC recovery", "#e6efdf")]
    for x, y, w, h, txt, color in blocks:
        ax.add_patch(Rectangle((x,y),w,h,facecolor=color,edgecolor="#56626e",lw=.8))
        ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=8.2,linespacing=1.5)
    for x1,x2 in [(1.9,2.3),(4.2,4.6),(6.4,6.8),(8.6,9.)]:
        ax.add_patch(FancyArrowPatch((x1,1.45),(x2,1.45),arrowstyle="-|>",mutation_scale=14,color="#506578"))
    ax.plot([5.5,5.5,3.2],[2.25,2.62,2.62],color="#506578",lw=1.0)
    ax.add_patch(FancyArrowPatch((3.2,2.62),(3.2,2.25),arrowstyle="-|>",mutation_scale=12,color="#506578"))
    ax.text(4.35,2.8,"Plan result returned to the LLM",ha="center",fontsize=8)
    ax.text(5.7,.22,"Native tool template and grammar-constrained greedy decoding; full calls and physical transitions are retained.",ha="center",fontsize=8.5)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(figures / f"llm_workflow.{ext}",dpi=220,bbox_inches="tight")
    plt.close(fig)


def main():
    out = HERE / "results"
    pairs = pd.read_csv(out / "paired_episodes.csv")
    steps = pd.read_csv(out / "agent_steps.csv")
    traces = pd.read_csv(out / "traces.csv")
    transcripts = [json.loads(x) for x in (out / "model_transcripts.jsonl").read_text().splitlines()]
    ledger = [json.loads(x) for x in (out / "effect_ledger.jsonl").read_text().splitlines()]
    check = {}
    check["all_12_preregistered_seeds"] = pairs.seed.tolist() == list(range(24000,24012))
    check["all_96_llm_cycles_retained"] = len(steps)==96 and bool((steps.groupby(["seed","condition"]).size()==4).all())
    check["all_960_state_transitions"] = len(traces)==960 and bool((traces.groupby(["seed","condition"]).size()==40).all())
    check["transcript_count_matches_calls"] = len(transcripts)==int(steps.generation_calls.sum())
    check["generation_budget"] = all(1<=x<=4 for x in steps.generation_calls)
    check["physical_balance"] = bool((abs(traces.balance_error_kw)<1e-7).all())
    check["energy_hard_bounds"] = bool(((traces.energy_after_kwh>=1.2-1e-7)&(traces.energy_after_kwh<=12+1e-7)).all())
    check["nonexecuted_cycles_hold"] = bool((steps.loc[steps.status!="executed_plan",["requested_charge_kw","requested_discharge_kw"]].abs().to_numpy()<1e-10).all())
    chains={}; chain_ok=True
    for original in ledger:
        e=dict(original); h=e.pop("event_hash")
        key=(e["scenario_id"],e["condition"])
        chain_ok &= h==digest(e) and e["previous_event_hash"]==chains.get(key,"0"*64)
        chains[key]=h
    check["all_effect_chains_valid"] = chain_ok and len(chains)==24 and len(ledger)==96
    grouped={}
    for r in transcripts:
        grouped.setdefault((r["scenario_id"],r["condition"],r["t"]),[]).append(r)
    check["fresh_context_each_cycle"] = all(len(v[0]["messages"])==2 and v[0]["turn"]==0 for v in grouped.values())
    check["shared_system_prompt"] = len({v[0]["messages"][0]["content"] for v in grouped.values()})==1
    check["no_attack_label_in_initial_observation"] = all(not any(w in v[0]["messages"][1]["content"] for w in ['"condition"','"poisoned"','"strength"','"seed"']) for v in grouped.values())
    check["all_outputs_in_logged_legal_grammar"] = all(x["raw_response"] in x["legal_serializations"] for x in transcripts)
    initial_options = [x["legal_calls"] for x in transcripts if x["turn"]==0]
    check["all_initial_cycles_offer_received_or_hold"] = all(
        {c["arguments"].get("forecast_source") for c in calls if c["name"]=="plan_dispatch"}=={"received"}
        and {c["name"] for c in calls}=={"plan_dispatch","hold"} for calls in initial_options)
    check["native_prompt_sha256"] = all(hashlib.sha256(x["rendered_prompt"].encode()).hexdigest()==x["rendered_prompt_sha256"] for x in transcripts)
    frozen=json.loads((HERE/"protocol.json").read_text())
    check["frozen_implementation_hash"] = frozen["implementation_sha256"]==hashlib.sha256((HERE/"run_validation.py").read_bytes()).hexdigest()
    check["model_revision_same_as_pilot"] = json.loads((out/"environment_manifest.json").read_text())["model_revision"]=="aa8e72537993ba99e69dfaafa59ed015b17504d1"
    executed_ok=True
    for r in steps.to_dict("records"):
        if r["status"]!="executed_plan": continue
        records=grouped[(r["scenario_id"],r["condition"],r["t"])]
        execute=records[-1]
        matching=[e for e in records if e["tool_result"].get("plan_id")==r["executed_plan_id"]]
        executed_ok &= len(matching)==1 and execute.get("parsed_call",{}).get("tool")=="execute_plan"
        if matching:
            plan=matching[0]["tool_result"]
            executed_ok &= abs(plan["first_charge_kw"]-r["requested_charge_kw"])<1e-9 and abs(plan["first_discharge_kw"]-r["requested_discharge_kw"])<1e-9
    check["all_executions_follow_real_model_authorization"] = bool(executed_ok)
    warm=traces[traces.phase=="mpc_warmup"]
    cols=["energy_before_kwh","energy_after_kwh","requested_charge_kw","requested_discharge_kw"]
    check["paired_warmup_identical"] = bool(np.allclose(warm[warm.condition=="clean"][cols],warm[warm.condition=="poisoned"][cols],atol=0,rtol=0))
    info=json.loads((out/"environment_manifest.json").read_text())
    check["source_files_unchanged"] = all(hashlib.sha256((HERE.parents[2]/p).read_bytes()).hexdigest()==v for p,v in info["source_hashes"].items())
    check["script_matches_executed_manifest"] = hashlib.sha256((HERE/"run_validation.py").read_bytes()).hexdigest()==info["script_sha256"]
    write_json(out/"validation.json", {"checks":check,"passed":sum(check.values()),"total":len(check),"all_passed":all(check.values())})
    if not all(check.values()):
        raise RuntimeError(str({k:v for k,v in check.items() if not v}))
    eligible=pairs.actual_feasible & pairs.reference_feasible & pairs.poisoned_recovery_success & pairs.clean_recovery_success
    error=(pairs.loc[eligible,"estimated_extra_cost"]-pairs.loc[eligible,"realized_extra_cost"]).to_numpy()
    divergent=abs(pairs.energy_deficit_kwh)>1e-6
    grouped_counts=[]
    for condition in ["clean","poisoned"]:
        z=steps[steps.condition==condition]
        grouped_counts.append({"condition":condition,"decision_cycles":len(z),"executed_received":int((z.selected_source=="received").sum()),
                               "executed_verified":int((z.selected_source=="verified").sum()),"model_hold":int((z.status=="model_hold").sum()),
                               "invalid_call_hold":int((z.status=="invalid_call_hold").sum()),"tool_budget_hold":int((z.status=="tool_budget_hold").sum()),
                               "generation_calls":int(z.generation_calls.sum()),"mean_inference_seconds_per_cycle":float(z.inference_seconds.mean())})
    pd.DataFrame(grouped_counts).to_csv(out/"tool_choice_summary.csv",index=False)
    summary={"n_pairs":len(pairs),"n_llm_cycles":len(steps),"n_generation_calls":len(transcripts),"n_physically_divergent_pairs":int(divergent.sum()),
             "n_initially_paired_feasible":int((pairs.actual_feasible & pairs.reference_feasible).sum()),"n_successful_paired_predictions":int(eligible.sum()),
             "energy_deficit_mean_kwh":float(pairs.energy_deficit_kwh.mean()),"energy_deficit_range_kwh":[float(pairs.energy_deficit_kwh.min()),float(pairs.energy_deficit_kwh.max())],
             "estimated_cost_mean_usd":float(pairs.estimated_extra_cost.mean()),"executed_cost_mean_usd":float(pairs.realized_extra_cost.mean()),
             "eligible_prediction_mae_usd":float(abs(error).mean()) if len(error) else None,
             "eligible_prediction_rmse_usd":float(np.sqrt((error**2).mean())) if len(error) else None,
             "tool_choices":grouped_counts,"raw_tool_calls":dict(Counter(x.get("parsed_call",{}).get("tool","invalid") for x in transcripts)),
             "decoding":"grammar-constrained greedy; plan(received)/hold and execute/hold choices come from model output",
             "design_origin":"Authenticated-source availability ablation on the same twelve formal seeds; not additional independent samples.",
             "interpretation":"All fixed pairs retained. If no physical divergence occurs, zero attribution error is algebraically trivial and is not predictive validation.",
             "figure_case_selection":"seed 24000, fixed first case; no outcome-based selection"}
    write_json(out/"summary.json",summary)
    plot(out,pairs,steps,traces)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
