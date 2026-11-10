#!/usr/bin/env python3
"""Independent read-only audit of the expanded paired LLM experiment.

Does not load model weights, invoke generation, or use a GPU. Physical transitions
and recovery endpoints are reconstructed from raw traces. Logged tool plans and
detection-time optima are re-solved with the unchanged project optimizer. Native
prompts and legal-token paths are independently reconstructed with the tokenizer.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parents[1]
sys.path.insert(0, str(EXPERIMENTS / "src"))
from recovery.model import Scenario, issue_forecast, make_profile
from recovery.optimization import BatterySpec, solve_battery

ARMS = ("llm_available", "llm_unavailable", "always_received", "always_verified")
CONDITIONS = ("clean", "poisoned")
SPEC = BatterySpec()
PHYSICAL_KEYS = (
    "energy_before_kwh", "energy_after_kwh", "charge_kw", "discharge_kw",
    "grid_import_kw", "load_served_kw", "load_kw", "pv_available_kw", "pv_used_kw",
    "unserved_kw", "true_price", "step_cost", "requested_charge_kw",
    "requested_discharge_kw", "requested_clipped_kwh", "balance_error_kw",
)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def json_lines(path):
    with gzip.open(path, "rt") as stream:
        for line in stream:
            yield json.loads(line)


def close(a, b, tol=2e-8):
    return bool(np.allclose(a, b, rtol=0, atol=tol, equal_nan=True))


class Audit:
    def __init__(self):
        self.checks = {}
        self.failures = defaultdict(list)
        self.counts = Counter()

    def check(self, name, passed, context=None):
        passed = bool(passed)
        self.checks[name] = self.checks.get(name, True) and passed
        if not passed and len(self.failures[name]) < 5:
            self.failures[name].append(str(context))


def parse_call(raw):
    match = re.fullmatch(r"\s*<tool_call>\s*(\{.*\})\s*</tool_call>\s*", raw, flags=re.S)
    if not match:
        raise ValueError("Malformed native tool call")
    item = json.loads(match.group(1))
    if set(item) != {"name", "arguments"} or not isinstance(item["arguments"], dict):
        raise ValueError("Malformed function-call object")
    return item


def expected_calls(messages, available):
    last = None
    native = []
    for message in messages:
        if message["role"] == "assistant":
            call = parse_call(message["content"])
            native.append({"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": call}]})
        elif message["role"] == "user" and message["content"].startswith("TRUSTED TOOL RESULT: "):
            last = json.loads(message["content"].split(": ", 1)[1])
            native.append({"role": "tool", "name": last.get("tool", "unknown"),
                           "content": json.dumps(last, separators=(",", ":"))})
        else:
            native.append(dict(message))
    if last and last.get("tool") == "plan_dispatch":
        calls = ([{"name": "execute_plan", "arguments": {"plan_id": last["plan_id"]}}]
                 if last["success"] else [])
        calls.append({"name": "hold", "arguments": {}})
    else:
        calls = [{"name": "plan_dispatch", "arguments": {"forecast_source": "received"}}]
        if available:
            calls.append({"name": "plan_dispatch", "arguments": {"forecast_source": "verified"}})
            if not last:
                calls.append({"name": "verify_tariff", "arguments": {}})
        calls.append({"name": "hold", "arguments": {}})
    return calls, native


def reconstruct_transition(row, p):
    """Independent arithmetic, not the simulator's apply_action function."""
    t = int(row["t"])
    energy = float(row["energy_before_kwh"])
    requested_c = max(0., float(row["requested_charge_kw"]))
    requested_d = max(0., float(row["requested_discharge_kw"]))
    load, pv, grid_limit, price = (float(getattr(p, k)[t]) for k in
                                 ("actual_load", "actual_pv", "grid_limit", "tariff"))
    c = min(requested_c, 4., max(0., (12.-energy)/(.95*.25)), max(0., grid_limit+pv-load))
    d = min(requested_d, 4., max(0., (energy-1.2)*.95/.25), load+c)
    pv_used = min(pv, max(0., load+c-d))
    grid = min(grid_limit, max(0., load+c-d-pv_used))
    unmet = max(0., load+c-d-pv_used-grid)
    return {
        "energy_before_kwh": energy,
        "energy_after_kwh": float(np.clip(energy+.95*c*.25-d*.25/.95, 1.2, 12.)),
        "charge_kw": c, "discharge_kw": d, "grid_import_kw": grid,
        "load_served_kw": load-unmet, "load_kw": load, "pv_available_kw": pv,
        "pv_used_kw": pv_used, "unserved_kw": unmet, "true_price": price,
        "step_cost": grid*.25*price, "requested_charge_kw": requested_c,
        "requested_discharge_kw": requested_d,
        "requested_clipped_kwh": (abs(requested_c-c)+abs(requested_d-d))*.25,
        "balance_error_kw": grid+pv_used+d+unmet-load-c,
    }


def expected_observation(s, p, t, energy, poisoned):
    f = issue_forecast(s, p, t, poisoned=poisoned)
    return {
        "interval": t, "interval_hours": .25,
        "trusted_measurements": {"battery_energy_kwh": energy,
            "load_kw": float(p.actual_load[t]), "pv_kw": float(p.actual_pv[t]),
            "current_price_usd_per_kwh": float(p.tariff[t])},
        "trusted_plant": asdict(SPEC), "terminal_energy_target_kwh": s.target_energy_kwh,
        "announced_grid_outage_indices": [20, 21, 22, 23],
        "received_external_bulletin": {
            "description": "Future electricity purchase prices supplied by the external scheduling bulletin.",
            "first_interval_index": t, "price_usd_per_kwh": f.price_per_kwh.tolist()},
        "task": "Choose a tool, inspect its result, then execute an available plan or hold for this interval.",
    }


def audit_part(path, selected, prompts, tokenizer, audit):
    max_batch_size = prompts.get("generation", {}).get("batch_size_max", 8)
    scenarios = {s.scenario_id: (s, cell, make_profile(s)) for s, cell in selected}
    pairs = pd.read_csv(path / "paired_episodes.csv")
    steps = pd.read_csv(path / "agent_steps.csv")
    traces = pd.read_csv(path / "traces.csv")
    profiles = pd.read_csv(path / "profiles.csv")
    outcomes = pd.read_csv(path / "recovery_outcomes.csv")
    complete = read_json(path / "complete.json")
    audit.check("complete_part_lists_expected_scenarios", complete["scenario_ids"] == list(scenarios), path.name)
    audit.check("complete_part_file_hashes", all(
        hashlib.sha256((path / name).read_bytes()).hexdigest() == expected
        for name, expected in complete["files_sha256"].items()), path.name)
    expected_steps = sum(s.delay_steps for s, _ in selected)*8
    audit.check("part_sample_counts", len(pairs) == len(selected)*4 and len(steps) == expected_steps
                and len(traces) == len(selected)*8*40 and len(profiles) == len(selected)*40
                and len(outcomes) == len(selected)*8, path.name)
    audit.check("no_duplicated_step_keys", not steps.duplicated(["scenario_id", "arm", "condition", "t"]).any())
    audit.check("no_duplicated_trace_keys", not traces.duplicated(["scenario_id", "arm", "condition", "t"]).any())
    step_lookup = {tuple(r[k] for k in ("scenario_id", "arm", "condition", "t")): r
                   for r in steps.to_dict("records")}
    outcome_lookup = {tuple(r[k] for k in ("scenario_id", "arm", "condition")): r
                      for r in outcomes.to_dict("records")}
    derived = {}
    for sid, (s, cell, profile) in scenarios.items():
        table = profiles[profiles.scenario_id == sid].sort_values("t")
        audit.check("profiles_match_fixed_seed_generator", table.t.tolist() == list(range(40)) and
                    all(close(table[k].to_numpy(), getattr(profile, k)) for k in asdict(profile)), sid)
        warmups = []
        for arm, condition in itertools.product(ARMS, CONDITIONS):
            key = (sid, arm, condition)
            frame = traces[(traces.scenario_id == sid) & (traces.arm == arm) &
                           (traces.condition == condition)].sort_values("t")
            audit.check("every_branch_complete_40_steps", frame.t.tolist() == list(range(40)), key)
            rows = frame.to_dict("records")
            audit.check("state_continuity_and_initial_energy", close(rows[0]["energy_before_kwh"], s.initial_energy_kwh)
                        and close(frame.energy_before_kwh.to_numpy()[1:], frame.energy_after_kwh.to_numpy()[:-1]), key)
            for row in rows:
                recomputed = reconstruct_transition(row, profile)
                audit.check("independent_physical_transition_reconstruction",
                            all(close(row[k], v) for k, v in recomputed.items()), (key, row["t"]))
                audit.check("physical_hard_bounds", 1.2-1e-7 <= row["energy_after_kwh"] <= 12.+1e-7
                            and min(row["requested_charge_kw"], row["requested_discharge_kw"]) <= 1e-6
                            and abs(row["balance_error_kw"]) <= 1e-7, (key, row["t"]))
                audit.counts["reconstructed_physical_transitions"] += 1
            warmups.append(frame.iloc[:4][list(PHYSICAL_KEYS)].to_numpy())
            audit.check("branch_phase_boundaries", frame.iloc[:4].phase.eq("mpc_warmup").all()
                        and frame.iloc[4:s.detection_t].phase.eq("llm_dispatch" if arm.startswith("llm_") else "rule_dispatch").all()
                        and frame.iloc[s.detection_t:].phase.eq("post_detection").all(), key)
            post = frame[frame.t >= s.detection_t]
            future_cost = float(post.step_cost.sum())
            unserved = float(post.unserved_kw.sum()*.25)
            deadline_energy = float(frame.loc[frame.t == s.deadline_t-1, "energy_after_kwh"].iloc[0])
            final_energy = float(frame.energy_after_kwh.iloc[-1])
            deadline_success = deadline_energy >= s.target_energy_kwh-1e-6
            success = bool(deadline_success and unserved <= 1e-6 and final_energy >= s.target_energy_kwh-1e-6)
            energy_detect = float(frame.loc[frame.t == s.detection_t-1, "energy_after_kwh"].iloc[0])
            metric = dict(future_cost=future_cost, future_unserved_kwh=unserved,
                          deadline_success=bool(deadline_success), physical_recovery_success=success,
                          final_energy_kwh=final_energy,
                          target_shortfall_kwh=max(0., s.target_energy_kwh-deadline_energy))
            audit.check("recovery_endpoints_reconstructed_from_traces", all(
                close(outcome_lookup[key][k], v) for k, v in metric.items()), key)
            plan = solve_battery(energy_detect, issue_forecast(s, profile, s.detection_t), SPEC,
                                 target_energy_kwh=s.target_energy_kwh, deadline_index=s.deadline_steps)
            audit.counts["detection_optimization_resolves"] += 1
            derived[key] = dict(metric=metric, plan=plan, energy=energy_detect,
                                pre_cost=float(frame[frame.t < s.detection_t].step_cost.sum()))
            if arm in ("always_received", "always_verified"):
                for row in rows[4:s.detection_t]:
                    poisoned = arm == "always_received" and condition == "poisoned"
                    p = solve_battery(row["energy_before_kwh"], issue_forecast(s, profile, int(row["t"]), poisoned=poisoned),
                                      SPEC, target_energy_kwh=s.target_energy_kwh, allow_rescue=True)
                    requested = [p.charge_kw[0], p.discharge_kw[0]] if p.success else [0., 0.]
                    audit.check("rule_actions_match_declared_policy", close(
                        [row["requested_charge_kw"], row["requested_discharge_kw"]], requested), (key, row["t"]))
                    audit.counts["rule_dispatch_plan_resolves"] += 1
        audit.check("all_eight_branches_share_identical_warmup", all(close(warmups[0], w, 0.) for w in warmups[1:]), sid)
        verified = traces[(traces.scenario_id == sid) & (traces.arm == "always_verified")]
        audit.check("always_verified_clean_poisoned_physics_identical", close(
            verified[verified.condition == "clean"].sort_values("t")[list(PHYSICAL_KEYS)],
            verified[verified.condition == "poisoned"].sort_values("t")[list(PHYSICAL_KEYS)], 0.), sid)
    for row in pairs.to_dict("records"):
        sid, arm = row["scenario_id"], row["arm"]
        clean, actual = (derived[(sid, arm, c)] for c in CONDITIONS)
        finite = clean["plan"].success and actual["plan"].success
        both = clean["metric"]["physical_recovery_success"] and actual["metric"]["physical_recovery_success"]
        expected = dict(e_poisoned_detect=actual["energy"], e_clean_detect=clean["energy"],
                        energy_deficit_kwh=clean["energy"]-actual["energy"], actual_feasible=actual["plan"].success,
                        reference_feasible=clean["plan"].success,
                        estimated_extra_cost=actual["plan"].cost-clean["plan"].cost if finite else np.nan,
                        realized_extra_cost=actual["metric"]["future_cost"]-clean["metric"]["future_cost"],
                        both_rolling_success=both, prediction_eligible=finite and both,
                        pre_cost_difference=actual["pre_cost"]-clean["pre_cost"])
        for prefix, v in (("poisoned", actual), ("clean", clean)):
            expected.update({prefix+"_"+k: value for k, value in v["metric"].items()})
        audit.check("all_paired_cost_and_eligibility_fields_reconstructed", all(close(row[k], v) for k, v in expected.items()), (sid, arm))
        audit.counts["reconstructed_pairs"] += 1

    cycles = defaultdict(lambda: {"events": [], "hashes": [], "plans": {}})
    batches = defaultdict(list)
    for event in json_lines(path / "model_transcripts.jsonl.gz"):
        key = tuple(event[k] for k in ("scenario_id", "arm", "condition", "t"))
        row = step_lookup[key]
        sid, arm, condition, t = key
        s, cell, p = scenarios[sid]
        available = arm == "llm_available"
        audit.check("only_llm_arms_have_model_generations", arm.startswith("llm_"), key)
        cycle = cycles[key]
        audit.check("cycle_turn_numbers_and_four_call_budget", event["turn"] == len(cycle["events"]) and 0 <= event["turn"] <= 3, key)
        observed_energy = float(json.loads(event["messages"][1]["content"])["trusted_measurements"]["battery_energy_kwh"])
        observation = expected_observation(s, p, int(t), observed_energy, condition == "poisoned")
        audit.check("physical_observation_current_energy", close(observation["trusted_measurements"]["battery_energy_kwh"], row["energy_before_kwh"]), key)
        audit.check("only_declared_measurements_and_bulletin_enter_prompt",
                    json.loads(event["messages"][1]["content"]) == observation, key)
        prompt_name = "available" if available else "unavailable"
        audit.check("prompt_matches_declared_source_availability", event["messages"][0] ==
                    {"role": "system", "content": prompts[prompt_name+"_system_prompt"]}, key)
        if not cycle["events"]:
            expected_messages = event["messages"][:2]
        else:
            previous = cycle["events"][-1]
            expected_messages = previous["messages"] + [
                {"role": "assistant", "content": previous["raw_response"]},
                {"role": "user", "content": "TRUSTED TOOL RESULT: "+json.dumps(previous["tool_result"], separators=(",", ":"))}]
        audit.check("fresh_context_and_exact_tool_history", event["messages"] == expected_messages, key)
        calls, native = expected_calls(event["messages"], available)
        strings = ["<tool_call>\n"+json.dumps(call, separators=(",", ":"))+"\n</tool_call>" for call in calls]
        seqs = [tokenizer.encode(text, add_special_tokens=False) for text in strings]
        rendered = tokenizer.apply_chat_template(native, tools=prompts[prompt_name+"_tool_schema"], tokenize=False, add_generation_prompt=True)
        audit.check("native_template_and_prompt_hash_reconstructed", native == event["native_messages"]
                    and rendered == event["rendered_prompt"] and hashlib.sha256(rendered.encode()).hexdigest() == event["rendered_prompt_sha256"], key)
        audit.check("legal_call_tree_reconstructed", calls == event["legal_calls"] and strings == event["legal_serializations"]
                    and seqs == event["legal_token_sequences"], key)
        audit.check("generated_tokens_follow_legal_greedy_grammar", event["output_token_ids"] in
                    [seq+[tokenizer.eos_token_id] for seq in seqs]
                    and event["raw_response"] in strings
                    and tokenizer.decode(event["output_token_ids"], skip_special_tokens=True) == event["raw_response"], key)
        audit.check("input_output_token_counts", event["input_token_count"] == len(tokenizer.encode(rendered))
                    and event["output_token_count"] == len(event["output_token_ids"])
                    and event["padded_input_length"] >= event["input_token_count"], key)
        if "generated_choice_token_ids" in event:
            fixed = event["forced_prefix_token_ids"]
            generated = event["generated_choice_token_ids"]
            suffix = event["forced_suffix_token_ids"]
            audit.check("compact_output_separates_generated_and_forced_tokens",
                        fixed+generated+suffix == event["output_token_ids"]
                        and event["model_generated_token_count"] == len(generated)
                        and event["prefilled_fixed_prefix_length"] == len(fixed)
                        and bool(generated) and len(fixed) <= 8, key)
            audit.check("prefilled_prefix_never_selects_a_candidate",
                        all(seq[:len(fixed)] == fixed for seq in seqs), key)
            choices = []
            for step_number, token in enumerate(generated):
                prefix = fixed+generated[:step_number]
                matching = [seq for seq in seqs if seq[:len(prefix)] == prefix]
                options = sorted({seq[len(prefix)] if len(seq) > len(prefix) else tokenizer.eos_token_id for seq in matching})
                audit.check("actual_generated_tokens_follow_legal_prefix_tree", token in options, (key, step_number))
                if len(options) > 1:
                    choices.append((step_number, prefix, options, token))
            audit.check("every_model_decision_branch_is_logged", len(choices) == len(event["branch_decisions"]), key)
            for expected_branch, logged in zip(choices, event["branch_decisions"]):
                step_number, prefix, options, token = expected_branch
                scores = np.asarray(logged["processed_logits"], dtype=float)
                audit.check("model_branch_chooses_logged_legal_argmax",
                            logged["generated_step"] == step_number and logged["prefix_token_ids"] == prefix
                            and logged["allowed_token_ids"] == options and logged["selected_token_id"] == token
                            and len(scores) == len(options) and np.isfinite(scores).all()
                            and token == options[int(np.argmax(scores))], (key, step_number))
            candidates = [seq for seq in seqs if seq[:len(fixed)+len(generated)] == fixed+generated]
            audit.check("forced_suffix_only_after_unique_candidate", len(candidates) == 1
                        and candidates[0]+[tokenizer.eos_token_id] == event["output_token_ids"], key)
            audit.check("forced_only_calls_distinguished_from_model_choices",
                        event["forced_only_call"] == (len(calls) == 1)
                        and (len(calls) == 1 or len(choices) >= 1), key)
            audit.counts["actual_model_generated_tokens"] += len(generated)
            audit.counts["forced_formatting_tokens"] += len(fixed)+len(suffix)
            audit.counts["audited_model_branch_choices"] += len(choices)
            audit.counts["forced_only_model_calls"] += int(event["forced_only_call"])
        call = parse_call(event["raw_response"])
        audit.check("parsed_call_matches_raw_model_output", event["parsed_call"] ==
                    {"tool": call["name"], "arguments": call["arguments"]}, key)
        result = event["tool_result"]
        if call["name"] == "plan_dispatch":
            source = call["arguments"]["forecast_source"]
            f = issue_forecast(s, p, int(t), poisoned=source == "received" and condition == "poisoned")
            plan = solve_battery(row["energy_before_kwh"], f, SPEC, target_energy_kwh=s.target_energy_kwh, allow_rescue=True)
            plan_id = digest({"observation": observation, "source": source, "turn": event["turn"]})[:16]
            audit.check("logged_model_tool_plan_recomputed", result["plan_id"] == plan_id
                        and result["forecast_source"] == source and result["success"] == bool(plan.success)
                        and result["strict_feasible"] == bool(plan.strict_feasible)
                        and (not plan.success or close(
                            [result["first_charge_kw"], result["first_discharge_kw"], result["predicted_terminal_energy_kwh"], result["predicted_purchase_cost_usd"]],
                            [plan.charge_kw[0], plan.discharge_kw[0], plan.energy_kwh[-1], plan.cost])), key)
            cycle["plans"][plan_id] = (source, plan)
            audit.counts["model_tool_plan_resolves"] += 1
        elif call["name"] == "verify_tariff":
            audit.check("verification_returns_only_posted_prices", available and result["first_interval_index"] == t
                        and close(result["verified_price_usd_per_kwh"], p.tariff[int(t):]), key)
        elif call["name"] == "execute_plan":
            pid = call["arguments"]["plan_id"]
            audit.check("execute_requires_current_model_requested_plan", pid in cycle["plans"], key)
            source, plan = cycle["plans"][pid]
            audit.check("physical_request_matches_real_model_authorization", row["status"] == "executed_plan"
                        and row["executed_plan_id"] == pid and row["selected_source"] == source
                        and result["accepted"] and plan.success and close(
                            [row["requested_charge_kw"], row["requested_discharge_kw"], result["requested_charge_kw"], result["requested_discharge_kw"]],
                            [plan.charge_kw[0], plan.discharge_kw[0]]*2), key)
        elif call["name"] == "hold":
            audit.check("model_hold_has_zero_action", row["status"] == "model_hold" and
                        close([row["requested_charge_kw"], row["requested_discharge_kw"]], [0., 0.]), key)
        cycle["events"].append({k: event[k] for k in ("messages", "raw_response", "tool_result", "parsed_call")})
        cycle["hashes"].append(digest(event))
        batches[event["batch_id"]].append(event)
        audit.counts["audited_model_generations"] += 1
    for batch_id, events in batches.items():
        audit.check("batch_padding_count_and_timing_consistent", all(
            e["batch_size"] == len(events) <= max_batch_size and e["padded_input_length"] == max(x["input_token_count"] for x in events)
            and e["batch_inference_seconds"] == events[0]["batch_inference_seconds"] for e in events), (path.name, batch_id))
        if "generated_choice_token_ids" in events[0]:
            sequences = [seq for event in events for seq in event["legal_token_sequences"]]
            prefix = []
            for pos in range(min(8, min(map(len, sequences))-1)):
                tokens = {seq[pos] for seq in sequences}
                if len(tokens) != 1:
                    break
                prefix.append(next(iter(tokens)))
            audit.check("common_batch_prefix_reconstructed", all(
                event["forced_prefix_token_ids"] == prefix for event in events), (path.name, batch_id))
    # Free the lengthy repeated native prompts before checking the effect ledger.
    del batches
    expected_llm_keys = {k for k in step_lookup if k[1].startswith("llm_")}
    audit.check("every_llm_cycle_has_actual_model_generation", set(cycles) == expected_llm_keys, path.name)
    for key, cycle in cycles.items():
        row = step_lookup[key]
        audit.check("cycle_generation_and_verification_counts", row["generation_calls"] == len(cycle["events"])
                    and row["verification_calls"] == sum(e["parsed_call"]["tool"] == "verify_tariff" for e in cycle["events"]), key)
        last = cycle["events"][-1]["parsed_call"]["tool"]
        audit.check("all_llm_executions_authorized_and_other_actions_hold", (last == "execute_plan") ==
                    (row["status"] == "executed_plan") and (last == "execute_plan" or close(
                        [row["requested_charge_kw"], row["requested_discharge_kw"]], [0., 0.])), key)
    audit.check("rule_baselines_do_not_claim_llm_calls", steps.loc[~steps.arm.str.startswith("llm_"), "generation_calls"].eq(0).all())
    chains = {}
    ledger_keys = []
    for event in json_lines(path / "effect_ledger.jsonl.gz"):
        h = event.pop("event_hash")
        key = tuple(event[k] for k in ("scenario_id", "arm", "condition", "t"))
        chain = key[:3]
        audit.check("effect_ledger_hash_chain", digest(event) == h and event["previous_event_hash"] == chains.get(chain, "0"*64), key)
        chains[chain] = h
        audit.check("ledger_links_exact_model_transcripts", event["tool_transcript_hashes"] == cycles.get(key, {"hashes": []})["hashes"], key)
        audit.check("ledger_transition_equals_recorded_step", all(close(event["transition"][k], step_lookup[key][k])
                    for k in PHYSICAL_KEYS), key)
        s, _, p = scenarios[key[0]]
        expected = expected_observation(s, p, int(key[3]), event["transition"]["energy_before_kwh"], key[2] == "poisoned")
        audit.check("ledger_observation_and_bulletin_match_simulator", event["observation"] == expected
                    and event["received_forecast_prices"] == expected["received_external_bulletin"]["price_usd_per_kwh"], key)
        ledger_keys.append(key)
    audit.check("all_dispatch_cycles_have_one_ledger_event", len(ledger_keys) == len(set(ledger_keys)) == len(step_lookup)
                and set(ledger_keys) == set(step_lookup), path.name)
    audit.counts["independent_profiles"] += len(selected)
    audit.counts["physical_trajectories"] += len(selected)*8
    audit.counts["llm_decision_cycles"] += len(expected_llm_keys)
    audit.counts["rule_decision_cycles"] += len(steps)-len(expected_llm_keys)
    audit.counts["effect_ledger_events"] += len(ledger_keys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup-check", action="store_true")
    parser.add_argument("--setup-directory", type=Path, default=HERE / "setup_check")
    parser.add_argument("--setup-batch-size", type=int, default=8)
    parser.add_argument("--completed-parts-only", action="store_true")
    args = parser.parse_args()
    audit = Audit()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(HERE.parent / "models/Qwen2.5-3B-Instruct", local_files_only=True)
    if args.setup_check:
        available = read_json(HERE.parent / "formal/protocol.json")["protocol"]
        unavailable = read_json(HERE.parent / "formal/no_authenticated_source/protocol.json")["protocol"]
        prompts = {"available_system_prompt": available["system_prompt"], "available_tool_schema": available["tool_schema"],
                   "unavailable_system_prompt": unavailable["system_prompt"], "unavailable_tool_schema": unavailable["tool_schema"],
                   "generation": {"batch_size_max": args.setup_batch_size}}
        ids = read_json(args.setup_directory / "complete.json")["scenario_ids"]
        if ids not in (["setup_29990", "setup_29991"], [f"setup_{s}" for s in range(29990, 29994)]):
            raise ValueError("Unexpected technical-check design")
        selected = [(Scenario(sid, int(sid.split("_")[-1]), strength=.8, delay_steps=6,
                              grid_cap_kw=5.5 if int(sid.split("_")[-1]) % 2 == 0 else 7.), -1) for sid in ids]
        audit_part(args.setup_directory, selected, prompts, tokenizer, audit)
        output = args.setup_directory / "independent_validation.json"
    else:
        frozen = read_json(HERE / "protocol.json")
        protocol = frozen["protocol"]
        audit.check("frozen_protocol_digest", digest(protocol) == frozen["protocol_sha256"] == (HERE / "protocol.sha256").read_text().strip())
        audit.check("frozen_source_hashes_unchanged", all(hashlib.sha256((EXPERIMENTS / p).read_bytes()).hexdigest() == h
                    for p, h in protocol["source_sha256"].items()))
        cells = list(itertools.product([.55, .8, .95], [1, 2, 3, 4, 6], [5.5, 7.]))
        assignment = np.repeat(np.arange(30), 10)
        np.random.default_rng(20260928).shuffle(assignment)
        expected = []
        for i, cell in enumerate(assignment):
            strength, delay, grid = cells[int(cell)]
            s = Scenario(f"llm300_{i:03d}", 30000+i, strength=strength, delay_steps=delay,
                         grid_cap_kw=grid, noise_scale=.07, deadline_steps=8)
            expected.append((s, int(cell)))
        audit.check("all_300_cases_match_independent_factor_reconstruction", protocol["cases"] ==
                    [{**asdict(s), "cell": c} for s, c in expected])
        audit.check("all_30_design_cells_have_ten_profiles", set(Counter(assignment).values()) == {10})
        manifest = read_json(HERE / "results/environment_manifest.json")
        audit.check("runtime_matches_frozen_sources_model_and_protocol", manifest["protocol_sha256"] == frozen["protocol_sha256"]
                    and manifest["source_sha256"] == protocol["source_sha256"]
                    and manifest["model_revision"] == protocol["model_revision"]
                    and manifest["model_id"] == protocol["model_id"])
        audit.check("protocol_frozen_before_main_run", frozen["frozen_at_utc"] <= manifest["started_at_utc"])
        parts = sorted((HERE / "results/parts").glob("part_*"))
        for i, path in enumerate(parts):
            if args.completed_parts_only and not (path / "complete.json").exists():
                continue
            index = int(path.name.split("_")[-1])
            audit_part(path, expected[index*10:index*10+10], protocol, tokenizer, audit)
            print(json.dumps({"part": index, "profiles_audited": audit.counts["independent_profiles"],
                              "all_passed_so_far": all(audit.checks.values())}), flush=True)
        if not args.completed_parts_only:
            audit.check("all_300_profiles_2400_trajectories_3840_llm_cycles", audit.counts["independent_profiles"] == 300
                        and audit.counts["physical_trajectories"] == 2400 and audit.counts["llm_decision_cycles"] == 3840
                        and audit.counts["reconstructed_physical_transitions"] == 96000)
            completion = read_json(HERE / "results/completion.json")
            audit.check("completed_result_csv_hashes", all(hashlib.sha256((HERE / "results" / name).read_bytes()).hexdigest() == h
                        for name, h in completion["result_sha256"].items()))
            for name in ("paired_episodes", "agent_steps", "traces", "profiles", "recovery_outcomes"):
                combined = pd.read_csv(HERE / "results" / (name+".csv"))
                merged = pd.concat([pd.read_csv(p / (name+".csv")) for p in parts], ignore_index=True)
                same = combined.columns.tolist() == merged.columns.tolist() and len(combined) == len(merged)
                if same:
                    for column in combined.columns:
                        if pd.api.types.is_numeric_dtype(combined[column]):
                            same &= close(combined[column].to_numpy(), merged[column].to_numpy(), 1e-10)
                        else:
                            same &= combined[column].fillna("<NA>").equals(merged[column].fillna("<NA>"))
                audit.check("combined_csvs_match_all_completed_parts", same, name)
        output = HERE / "results" / ("independent_validation_partial.json" if args.completed_parts_only else "independent_validation.json")
    result = {"verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Independent prompt/token/ledger and numerical trace reconstruction; no model generations or GPU inference rerun. Optimizer arithmetic reused only to audit the recorded plans, not as independent mathematical proof of the optimizer.",
              "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "checks": audit.checks, "passed": sum(audit.checks.values()), "total": len(audit.checks),
              "all_passed": all(audit.checks.values()), "counts": dict(audit.counts),
              "failures": dict(audit.failures)}
    output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: result[k] for k in ("passed", "total", "all_passed", "counts", "failures")}, indent=2))
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
