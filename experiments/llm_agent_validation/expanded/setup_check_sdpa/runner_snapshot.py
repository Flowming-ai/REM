#!/usr/bin/env python3
"""Predeclared, paired 300-profile local LLM experiment; no paid API.

The old 12-profile scripts and main physics are imported read-only. Numerical
planning remains MPC; only an actual model execute_plan call permits execution.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parents[1]
sys.path.insert(0, str(EXPERIMENTS / "src"))
from recovery.model import Scenario, apply_action, issue_forecast, make_profile, simulate_recovery
from recovery.optimization import BatterySpec, solve_battery


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


FORMAL = module(HERE.parent / "formal/run_validation.py", "preserved_formal")
NOAUTH = module(HERE.parent / "formal/no_authenticated_source/run_validation.py", "preserved_noauth")
digest, write_json = FORMAL.digest, FORMAL.write_json
BATCH_SIZE = 16
ARMS = ("llm_available", "llm_unavailable", "always_received", "always_verified")
CONDITIONS = ("clean", "poisoned")


def cases():
    cells = list(itertools.product([.55, .80, .95], [1, 2, 3, 4, 6], [5.5, 7.]))
    assignments = np.repeat(np.arange(len(cells)), 10)
    np.random.default_rng(20260928).shuffle(assignments)
    result = []
    for i, cell in enumerate(assignments):
        strength, delay, grid = cells[int(cell)]
        s = Scenario(f"llm300_{i:03d}", 30000+i, strength=strength, delay_steps=delay,
                     grid_cap_kw=grid, noise_scale=.07, deadline_steps=8)
        result.append((s, int(cell)))
    return result


def source_hashes():
    files = [Path(__file__), Path(FORMAL.__file__), Path(NOAUTH.__file__),
             *sorted((EXPERIMENTS / "src/recovery").glob("*.py"))]
    return {str(p.relative_to(EXPERIMENTS)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def protocol():
    return {
        "model_id": FORMAL.MODEL_ID, "model_revision": FORMAL.REVISION,
        "cases": [{**asdict(s), "cell": cell} for s, cell in cases()],
        "factor_assignment_seed": 20260928,
        "arms": list(ARMS), "conditions": list(CONDITIONS),
        "independent_unit": "One new profile seed; eight paired trajectories share its exogenous profile.",
        "design": "30 balanced strength x detection-delay x source-capacity cells, ten unique seeds per cell; all 300 retained.",
        "source_availability": "Available and unavailable use the preserved native-tool prompts and schemas below; availability and its instructions change together.",
        "available_system_prompt": FORMAL.SYSTEM, "available_tool_schema": FORMAL.TOOLS,
        "unavailable_system_prompt": NOAUTH.SYSTEM, "unavailable_tool_schema": NOAUTH.TOOLS,
        "generation": {"do_sample": False, "num_beams": 1, "max_new_tokens": 192,
                       "dtype": "bfloat16", "attention": "sdpa", "seed": 20260924,
                       "batch_size_max": BATCH_SIZE, "padding_side": "left",
                       "decoding": "Native tools, grammar-constrained token-level greedy generation; legal syntax does not measure unconstrained model capability."},
        "batch_order": "Consecutive blocks of ten ascending scenario IDs; t ascending; scenario then arm then condition; active calls in this order, batches of at most sixteen. No caching or reuse of model outputs.",
        "context": "Fresh each interval; no attack, branch, scenario ID, seed, strength or arm label enters the prompt. Max four tool calls per cycle.",
        "warmup": "Indices 0..3 use the common clean MPC, physically copied to all eight branches.",
        "attack": "Only future received prices multiplied by 1-strength; current price and state measurements correct. Numerical data poisoning, no injected instruction.",
        "post_detection": "All branches use the unchanged measured-state MPC; same forecasts and realized disturbances.",
        "baselines": "Always-received and always-verified deterministically plan and execute. Both are labelled source-available; verified requires the authenticated source. Baselines are not model generations.",
        "failure_action": "Invalid call, unsuccessful plan or exhausted budget holds; never silently execute a plan for the LLM.",
        "primary_outcomes": ["signed clean-minus-poisoned detection energy", "positive energy-deficit rate >1e-6 kWh", "paired availability effect on energy deficit", "physical recovery success and unserved critical energy"],
        "cost_eligibility": "Both detection-time problems feasible and both subsequent rolling recoveries successful; additionally report finite/failed counts, zero-state pairs and nonzero-state pairs. Availability cost differences use the common eligible intersection.",
        "statistics": {"bootstrap_replicates": 20000, "seed": 20260929,
                       "scheme": "Resample ten whole profile seeds within each of the 30 design cells; retain all paired conditions. Percentile 95% intervals conditional on fixed design/model.",
                       "tool_rates": "Compute within-profile cycle proportions then average profiles; do not weight long delay profiles more heavily. Raw cycle counts separately reported.",
                       "factor_analysis": "Predeclared marginal groups and 30 cell descriptions; ten per cell is descriptive, not strong cell-wise inference."},
        "trajectory_figure_selection": "Lowest seed in each of the two cells strength=.95, delay=4 and grid=5.5 or 7; selected from design before outcomes.",
        "scope": "One simulated battery system and one local language model; fresh context, no physical robot or hardware, no prompt-injection or autonomous recovery claim.",
        "source_sha256": source_hashes(),
    }


def freeze():
    p = HERE / "protocol.json"
    obj = protocol()
    if p.exists():
        saved = json.loads(p.read_text())
        if saved["protocol"] != obj:
            raise RuntimeError("Frozen protocol or source changed. Preserve the run; do not relabel changed code as its reproduction.")
    else:
        saved = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                 "protocol_sha256": digest(obj), "protocol": obj}
        write_json(p, saved)
        (HERE / "protocol.sha256").write_text(saved["protocol_sha256"]+"\n")
        pd.DataFrame(obj["cases"]).to_csv(HERE / "scenarios.csv", index=False)
    return saved


class BatchModel(FORMAL.LocalModel):
    def __init__(self):
        super().__init__()
        self.model.set_attn_implementation("sdpa")
        if self.model.config._attn_implementation != "sdpa":
            raise RuntimeError("Requested SDPA backend was not activated.")
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.batch_number = 0

    def prepare(self, messages, available):
        native, last_result = [], None
        for message in messages:
            if message["role"] == "assistant":
                name, args = FORMAL.parse_call(message["content"])
                native.append({"role": "assistant", "content": "", "tool_calls":
                               [{"type": "function", "function": {"name": name, "arguments": args}}]})
            elif message["role"] == "user" and message["content"].startswith("TRUSTED TOOL RESULT: "):
                last_result = json.loads(message["content"].split(": ", 1)[1])
                native.append({"role": "tool", "name": last_result.get("tool", "unknown"),
                               "content": json.dumps(last_result, separators=(",", ":"))})
            else:
                native.append(dict(message))
        if last_result and last_result.get("tool") == "plan_dispatch":
            calls = ([{"name": "execute_plan", "arguments": {"plan_id": last_result["plan_id"]}}]
                     if last_result["success"] else []) + [{"name": "hold", "arguments": {}}]
        else:
            calls = [{"name": "plan_dispatch", "arguments": {"forecast_source": "received"}}]
            if available:
                calls.append({"name": "plan_dispatch", "arguments": {"forecast_source": "verified"}})
                if not last_result:
                    calls.append({"name": "verify_tariff", "arguments": {}})
            calls.append({"name": "hold", "arguments": {}})
        strings = ["<tool_call>\n"+json.dumps(call, separators=(",", ":"))+"\n</tool_call>" for call in calls]
        sequences = [self.tokenizer.encode(s, add_special_tokens=False) for s in strings]
        rendered = self.tokenizer.apply_chat_template(native, tools=FORMAL.TOOLS if available else NOAUTH.TOOLS,
                                                      tokenize=False, add_generation_prompt=True)
        prefixes = {}
        for seq in sequences:
            for pos, token in enumerate(seq):
                prefixes.setdefault(tuple(seq[:pos]), set()).add(token)
            prefixes.setdefault(tuple(seq), set()).add(self.tokenizer.eos_token_id)
        return {"native_messages": native, "rendered_prompt": rendered,
                "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                "legal_calls": calls, "legal_serializations": strings,
                "legal_token_sequences": sequences, "prefixes": prefixes}

    def generate_batch(self, items):
        prepared = [self.prepare(c["messages"], c["available"]) for c in items]
        inputs = self.tokenizer([p["rendered_prompt"] for p in prepared], padding=True, return_tensors="pt").to("cuda")
        length = inputs["input_ids"].shape[1]
        eos = self.tokenizer.eos_token_id
        def allowed(batch_id, input_ids):
            prefix = tuple(input_ids[length:].tolist())
            if eos in prefix:
                return [eos]
            if prefix not in prepared[batch_id]["prefixes"]:
                raise RuntimeError("Generated sequence left declared grammar.")
            return sorted(prepared[batch_id]["prefixes"][prefix])
        self.torch.cuda.synchronize()
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=192,
                                         pad_token_id=eos, eos_token_id=eos, prefix_allowed_tokens_fn=allowed)
        self.torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        results = []
        for i, p in enumerate(prepared):
            ids = output[i, length:].tolist()
            if eos in ids:
                ids = ids[:ids.index(eos)+1]
            p.pop("prefixes")
            results.append({**p, "raw_response": self.tokenizer.decode(ids, skip_special_tokens=True),
                            "output_token_ids": ids, "output_token_count": len(ids),
                            "input_token_count": int(inputs["attention_mask"][i].sum().item()),
                            "padded_input_length": int(length), "batch_id": self.batch_number,
                            "batch_size": len(items), "batch_inference_seconds": elapsed,
                            "decoding": "grammar-constrained greedy"})
        self.batch_number += 1
        return results


def meta(s, cell, arm, condition):
    return {"scenario_id": s.scenario_id, "seed": s.seed, "cell": cell, "arm": arm,
            "condition": condition, "strength": s.strength, "delay_steps": s.delay_steps,
            "grid_cap_kw": s.grid_cap_kw, "deadline_steps": s.deadline_steps,
            "detection_t": s.detection_t, "deadline_t": s.deadline_t}


def warmup(s, p, spec):
    energy, rows = s.initial_energy_kwh, []
    for t in range(s.attack_t):
        plan = solve_battery(energy, issue_forecast(s, p, t), spec,
                             target_energy_kwh=s.target_energy_kwh, allow_rescue=True)
        if not plan.success:
            raise RuntimeError("Warmup failed: "+plan.status)
        result = apply_action(energy, plan.charge_kw[0], plan.discharge_kw[0], p.actual_load[t],
                              p.actual_pv[t], p.grid_limit[t], p.tariff[t], spec)
        rows.append({"t": t, "phase": "mpc_warmup", **result})
        energy = result["energy_after_kwh"]
    return energy, rows


def begin_cycle(branch, t, spec):
    s, p = branch["scenario"], branch["profile"]
    received = issue_forecast(s, p, t, poisoned=branch["meta"]["condition"] == "poisoned")
    verified = issue_forecast(s, p, t)
    obs = FORMAL.observation(s, p, t, branch["energy"], received, spec)
    available = branch["meta"]["arm"] != "llm_unavailable"
    return {"branch": branch, "t": t, "received": received, "verified": verified,
            "obs": obs, "available": available, "plans": {}, "events": [],
            "messages": [{"role": "system", "content": FORMAL.SYSTEM if available else NOAUTH.SYSTEM},
                         {"role": "user", "content": json.dumps(obs, separators=(",", ":"))}],
            "c": 0., "d": 0., "done": False, "status": "tool_budget_hold",
            "source": "none", "plan_id": "none", "parse_failure": False}


def accept_generation(cycle, generation, turn, spec, transcript):
    b = cycle["branch"]; s = b["scenario"]
    event = {**b["meta"], "t": cycle["t"], "turn": turn,
             "messages": [dict(m) for m in cycle["messages"]], **generation}
    try:
        name, args = FORMAL.parse_call(generation["raw_response"])
        if {"name": name, "arguments": args} not in generation["legal_calls"]:
            raise ValueError("Call outside declared legal set.")
        event["parsed_call"] = {"tool": name, "arguments": args}
        if name == "plan_dispatch":
            source = args["forecast_source"]
            if source == "verified" and not cycle["available"]:
                raise ValueError("Authenticated source unavailable.")
            plan = solve_battery(b["energy"], cycle[source], spec,
                                 target_energy_kwh=s.target_energy_kwh, allow_rescue=True)
            pid = digest({"observation": cycle["obs"], "source": source, "turn": turn})[:16]
            cycle["plans"][pid] = (source, plan)
            result = {"tool": name, "plan_id": pid, "forecast_source": source,
                      "success": bool(plan.success), "strict_feasible": bool(plan.strict_feasible),
                      "first_charge_kw": float(plan.charge_kw[0]) if plan.success else None,
                      "first_discharge_kw": float(plan.discharge_kw[0]) if plan.success else None,
                      "predicted_terminal_energy_kwh": float(plan.energy_kwh[-1]) if plan.success else None,
                      "predicted_purchase_cost_usd": float(plan.cost) if plan.success else None,
                      "instruction": "Execute this exact plan_id or hold; planning has not moved energy."}
        elif name == "verify_tariff":
            if not cycle["available"]:
                raise ValueError("No verification tool.")
            result = {"tool": name, "first_interval_index": cycle["t"],
                      "verified_price_usd_per_kwh": cycle["verified"].price_per_kwh.tolist()}
        elif name == "execute_plan":
            source, plan = cycle["plans"][args["plan_id"]]
            if not plan.success:
                raise ValueError("Unsuccessful plan.")
            cycle.update(c=float(plan.charge_kw[0]), d=float(plan.discharge_kw[0]), source=source,
                         plan_id=args["plan_id"], status="executed_plan", done=True)
            result = {"tool": name, "accepted": True, "requested_charge_kw": cycle["c"],
                      "requested_discharge_kw": cycle["d"]}
        elif name == "hold":
            cycle.update(status="model_hold", done=True)
            result = {"tool": name, "requested_charge_kw": 0., "requested_discharge_kw": 0.}
        else:
            raise ValueError("Unknown tool.")
    except (ValueError, TypeError, KeyError) as exc:
        cycle.update(parse_failure=True, done=True, status="invalid_call_hold")
        result = {"error": str(exc), "fallback": "hold"}
    event["tool_result"] = result
    transcript.write(json.dumps(event, separators=(",", ":"), allow_nan=False)+"\n")
    cycle["events"].append(event)
    if not cycle["done"]:
        cycle["messages"] += [{"role": "assistant", "content": generation["raw_response"]},
                              {"role": "user", "content": "TRUSTED TOOL RESULT: "+json.dumps(result, separators=(",", ":"))}]


def finish_cycle(cycle, spec):
    b = cycle["branch"]; p = b["profile"]; t = cycle["t"]
    result = apply_action(b["energy"], cycle["c"], cycle["d"], p.actual_load[t], p.actual_pv[t],
                          p.grid_limit[t], p.tariff[t], spec)
    row = {**b["meta"], "t": t, "phase": "llm_dispatch" if b["meta"]["arm"].startswith("llm_") else "rule_dispatch",
           "status": cycle["status"], "selected_source": cycle["source"],
           "executed_plan_id": cycle["plan_id"], "parse_failure": cycle["parse_failure"],
           "generation_calls": len(cycle["events"]),
           "verification_calls": sum(e.get("parsed_call", {}).get("tool") == "verify_tariff" for e in cycle["events"]),
           **result}
    event = {**b["meta"], "t": t, "observation": cycle["obs"],
             "received_forecast_prices": cycle["received"].price_per_kwh.tolist(),
             "tool_transcript_hashes": [digest(e) for e in cycle["events"]],
             "transition": row, "previous_event_hash": b["last_hash"]}
    event["event_hash"] = digest(event)
    b["last_hash"] = event["event_hash"]
    b["energy"] = row["energy_after_kwh"]
    b["history"].append(row)
    return row, event


def rule_cycle(cycle, spec):
    source = "verified" if cycle["branch"]["meta"]["arm"] == "always_verified" else "received"
    b = cycle["branch"]
    plan = solve_battery(b["energy"], cycle[source], spec,
                         target_energy_kwh=b["scenario"].target_energy_kwh, allow_rescue=True)
    if plan.success:
        cycle.update(c=float(plan.charge_kw[0]), d=float(plan.discharge_kw[0]), source=source,
                     status="rule_executed", plan_id="deterministic_rule", done=True)
    else:
        cycle.update(status="rule_infeasible_hold", done=True)


def run_part(model, selected, output):
    output.mkdir(parents=True, exist_ok=False)
    spec = BatterySpec(); branches = []; profiles = []
    for s, cell in selected:
        p = make_profile(s); energy, history = warmup(s, p, spec)
        for t in range(s.horizon):
            profiles.append({"scenario_id": s.scenario_id, "seed": s.seed, "cell": cell, "t": t,
                             **{key: float(getattr(p, key)[t]) for key in asdict(p)}})
        for arm in ARMS:
            for condition in CONDITIONS:
                m = meta(s, cell, arm, condition)
                branches.append({"scenario": s, "profile": p, "meta": m, "energy": energy,
                                 "history": [{**m, **r} for r in history], "last_hash": "0"*64})
    steps, ledgers, traces, pairs, outcomes = [], [], [], [], []
    with gzip.open(output / "model_transcripts.jsonl.gz", "wt") as transcript:
        for t in range(4, max(s.detection_t for s, _ in selected)):
            cycles = [begin_cycle(b, t, spec) for b in branches if t < b["scenario"].detection_t]
            for c in cycles:
                if not c["branch"]["meta"]["arm"].startswith("llm_"):
                    rule_cycle(c, spec)
            for turn in range(4):
                active = [c for c in cycles if not c["done"] and c["branch"]["meta"]["arm"].startswith("llm_")]
                for start in range(0, len(active), BATCH_SIZE):
                    batch = active[start:start+BATCH_SIZE]
                    for c, generation in zip(batch, model.generate_batch(batch)):
                        accept_generation(c, generation, turn, spec, transcript)
            for c in cycles:
                row, event = finish_cycle(c, spec); steps.append(row); ledgers.append(event)
            print(json.dumps({"part": output.name, "interval_completed": t, "llm_cycles_so_far": sum(r["phase"] == "llm_dispatch" for r in steps)}), flush=True)
    by_pair = {}
    for b in branches:
        s, p, m = b["scenario"], b["profile"], b["meta"]
        metric, rtrace, _, _ = simulate_recovery(s, p, spec, b["energy"], "state_mpc")
        outcomes.append({**metric, **m})
        traces += b["history"] + [{**r, **m} for r in rtrace]
        by_pair.setdefault((s.scenario_id, m["arm"]), {})[m["condition"]] = (b, metric)
    for (sid, arm), pair in by_pair.items():
        clean, mc = pair["clean"]; actual, ma = pair["poisoned"]
        s, p = actual["scenario"], actual["profile"]
        f = issue_forecast(s, p, s.detection_t)
        pa = solve_battery(actual["energy"], f, spec, target_energy_kwh=s.target_energy_kwh, deadline_index=s.deadline_steps)
        pr = solve_battery(clean["energy"], f, spec, target_energy_kwh=s.target_energy_kwh, deadline_index=s.deadline_steps)
        finite = bool(pa.success and pr.success)
        both_success = bool(ma["physical_recovery_success"] and mc["physical_recovery_success"])
        row = {k: v for k, v in actual["meta"].items() if k != "condition"}
        row.update(e_poisoned_detect=actual["energy"], e_clean_detect=clean["energy"],
                   energy_deficit_kwh=clean["energy"]-actual["energy"],
                   actual_feasible=bool(pa.success), reference_feasible=bool(pr.success),
                   estimated_extra_cost=float(pa.cost-pr.cost) if finite else np.nan,
                   realized_extra_cost=ma["future_cost"]-mc["future_cost"],
                   both_rolling_success=both_success, prediction_eligible=finite and both_success,
                   pre_cost_difference=sum(r["step_cost"] for r in actual["history"])-sum(r["step_cost"] for r in clean["history"]))
        for prefix, metric in (("poisoned", ma), ("clean", mc)):
            for key in ("physical_recovery_success", "future_cost", "future_unserved_kwh", "deadline_success",
                        "target_shortfall_kwh", "final_energy_kwh", "infeasible_plan_steps"):
                row[prefix+"_"+key] = metric[key]
        pairs.append(row)
    for name, rows in (("paired_episodes", pairs), ("agent_steps", steps), ("traces", traces),
                       ("profiles", profiles), ("recovery_outcomes", outcomes)):
        pd.DataFrame(rows).to_csv(output / (name+".csv"), index=False)
    with gzip.open(output / "effect_ledger.jsonl.gz", "wt") as stream:
        for event in ledgers:
            stream.write(json.dumps(event, separators=(",", ":"), allow_nan=False)+"\n")
    complete = {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "scenario_ids": [s.scenario_id for s, _ in selected],
                "files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir()) if p.is_file()}}
    write_json(output / "complete.json", complete)
    return {"profiles": len(selected), "trajectories": len(branches), "cycles": len(steps),
            "llm_generations": sum(r["generation_calls"] for r in steps)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--setup-check", action="store_true", help="Separate four-profile GPU/interface check; no main-test seeds.")
    parser.add_argument("--resume", action="store_true", help="Skip only completed hash-verified parts; refuse a partial part.")
    args = parser.parse_args()
    if args.setup_check:
        selected = [(Scenario("setup_29990", 29990, strength=.8, delay_steps=6, grid_cap_kw=5.5), -1),
                    (Scenario("setup_29991", 29991, strength=.8, delay_steps=6, grid_cap_kw=7.), -1),
                    (Scenario("setup_29992", 29992, strength=.8, delay_steps=6, grid_cap_kw=5.5), -1),
                    (Scenario("setup_29993", 29993, strength=.8, delay_steps=6, grid_cap_kw=7.), -1)]
        output = HERE / "setup_check_sdpa"
        if output.exists():
            raise RuntimeError("Preserve existing setup check.")
        model = BatchModel()
        started = time.perf_counter()
        result = run_part(model, selected, output)
        write_json(output / "setup_summary.json", {**result, "seconds": time.perf_counter()-started,
                   "scope": "Technical execution check only, excluded from main 300 and not used to select cases or prompts.",
                   "source_sha256": source_hashes()})
        print(json.dumps(result), flush=True)
        return
    frozen = freeze()
    if args.freeze_only:
        print(json.dumps({"frozen": frozen["frozen_at_utc"], "hash": frozen["protocol_sha256"]})); return
    output = HERE / "results"
    if output.exists() and not args.resume:
        raise RuntimeError("Results exist; only explicit hash-checked resume is allowed.")
    output.mkdir(exist_ok=True); parts = output / "parts"; parts.mkdir(exist_ok=True)
    manifest_path = output / "environment_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["protocol_sha256"] != frozen["protocol_sha256"]:
            raise RuntimeError("Resume protocol mismatch.")
    else:
        manifest = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": frozen["protocol_sha256"],
                    "python_executable": sys.executable, "python": platform.python_version(), "platform": platform.platform(),
                    "packages": {k: importlib.metadata.version(k) for k in ["torch", "transformers", "huggingface-hub", "numpy", "scipy", "pandas", "matplotlib", "tokenizers", "safetensors", "jinja2"]},
                    "source_sha256": source_hashes(), "model_id": FORMAL.MODEL_ID, "model_revision": FORMAL.REVISION,
                    "batch_timing_note": "Batch elapsed time is recorded once per batch and repeated on its members; it is not per-call or per-action latency."}
        write_json(manifest_path, manifest)
    selected = cases(); model = None; start = time.perf_counter()
    for part, offset in enumerate(range(0, len(selected), 10)):
        dest = parts / f"part_{part:03d}"
        if dest.exists():
            complete = dest / "complete.json"
            if not complete.exists():
                raise RuntimeError(f"Partial part retained at {dest}; archive and explicitly repair before resuming.")
            saved = json.loads(complete.read_text())
            for name, expected in saved["files_sha256"].items():
                if hashlib.sha256((dest/name).read_bytes()).hexdigest() != expected:
                    raise RuntimeError("Completed part hash mismatch.")
            continue
        if model is None:
            model = BatchModel()
            manifest.update(gpu=model.torch.cuda.get_device_name(), cuda=model.torch.version.cuda,
                            model_parameter_count=sum(p.numel() for p in model.model.parameters()))
            write_json(manifest_path, manifest)
        result = run_part(model, selected[offset:offset+10], dest)
        print(json.dumps({"part": part, "completed_profiles": offset+10, "wall_seconds_this_launch": time.perf_counter()-start, **result}), flush=True)
    for name in ("paired_episodes", "agent_steps", "traces", "profiles", "recovery_outcomes"):
        pd.concat([pd.read_csv(p/(name+".csv")) for p in sorted(parts.iterdir())], ignore_index=True).to_csv(output/(name+".csv"), index=False)
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(manifest_path, manifest)
    write_json(output / "completion.json", {"status": "complete", "independent_profiles": 300,
               "parts": 30, "protocol_sha256": frozen["protocol_sha256"],
               "result_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob("*.csv")}})


if __name__ == "__main__":
    main()
