#!/usr/bin/env python3
"""Retrospective OOF sensitivity analysis using saved scenarios only.

The original holdout analysis is reproduced before the supplementary analysis.
No plant trajectories, recovery optimizations, or original outputs are changed.
Bootstrap intervals are conditional on the fixed OOF predictions: models are
not refitted in bootstrap draws, and development/deployment uncertainty is not
covered. This is not a new independent test set.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parents[1]
SOURCE = EXPERIMENTS / "results" / "main"
CV_SEED = 20260924
BOOTSTRAP_SEED = 20260925
N_FOLDS = 5
BOOTSTRAP_DRAWS = 20_000
PREDICTORS = {
    "counterfactual_cost": "Paired constrained cost",
    "soc_ols": "Energy-deficit affine regression",
    "current_price": "Deficit / efficiency × current price",
    "forecast_price": "Deficit / efficiency × mean forecast price",
}
REQUIRED = [
    "realized_extra_cost", "estimated_extra_cost", "soc_deficit_kwh",
    "current_price", "mean_forecast_price", "eta_charge",
]
INPUTS = [
    SOURCE / "episodes.csv", SOURCE / "prediction_metrics_attacked.csv",
    SOURCE / "heldout_predictions_attacked.csv",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def truth(series: pd.Series) -> np.ndarray:
    normalized = series.astype(str).str.strip().str.lower()
    assert normalized.isin(["true", "false"]).all(), series.name
    return normalized.eq("true").to_numpy()


def fit_affine(train: pd.DataFrame) -> np.ndarray:
    x = np.column_stack([np.ones(len(train)), train.soc_deficit_kwh.to_numpy(float)])
    return np.linalg.lstsq(x, train.realized_extra_cost.to_numpy(float), rcond=None)[0]


def predict(frame: pd.DataFrame, coefficients: np.ndarray) -> pd.DataFrame:
    result = frame.copy()
    result["prediction_counterfactual_cost"] = result.estimated_extra_cost
    result["prediction_soc_ols"] = coefficients[0] + coefficients[1] * result.soc_deficit_kwh
    result["prediction_current_price"] = result.soc_deficit_kwh / result.eta_charge * result.current_price
    result["prediction_forecast_price"] = result.soc_deficit_kwh / result.eta_charge * result.mean_forecast_price
    for key in PREDICTORS:
        result[f"error_{key}"] = result[f"prediction_{key}"] - result.realized_extra_cost
        result[f"absolute_error_{key}"] = result[f"error_{key}"].abs()
    return result


def metrics(frame: pd.DataFrame, fold: int | None = None) -> list[dict]:
    rows = []
    for key, label in PREDICTORS.items():
        error = frame[f"error_{key}"].to_numpy(float)
        row = {
            "predictor": key, "label": label, "n_cases": len(frame),
            "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
        }
        if fold is not None:
            row["fold"] = fold
        rows.append(row)
    return rows


def save_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(HERE / name, index=False, float_format="%.17g")


def main() -> None:
    before = {str(p.relative_to(EXPERIMENTS)): digest(p) for p in INPUTS}
    episodes = pd.read_csv(INPUTS[0])
    assert episodes.scenario_id.is_unique and episodes.seed.is_unique
    attacked = episodes.loc[episodes.strength > 0].copy()
    paired_feasible = truth(attacked.actual_feasible) & truth(attacked.reference_feasible)
    finite = np.isfinite(attacked[REQUIRED].to_numpy(float)).all(axis=1)
    successful = truth(attacked.realized_paired_success)
    valid_efficiency = attacked.eta_charge.to_numpy(float) > 0
    eligible = successful & finite & valid_efficiency
    # The original rule is successful + finite; verify it implies paired
    # feasibility in this dataset rather than silently introducing a new rule.
    assert np.all(~eligible | paired_feasible)
    frame = attacked.loc[eligible].sort_values("scenario_id").reset_index(drop=True)
    assert len(episodes) == 100 and len(attacked) == 75
    assert int(paired_feasible.sum()) == 62 and len(frame) == 55
    assert len(frame) % N_FOLDS == 0

    audit = attacked[["scenario_id", "seed", "strength", "feasibility_class"]].copy()
    audit["paired_feasible"] = paired_feasible
    audit["finite_predictor_inputs_and_target"] = finite
    audit["successful_actual_and_reference_recovery"] = successful
    audit["positive_charge_efficiency"] = valid_efficiency
    audit["included"] = eligible
    audit["exclusion_reason"] = np.where(
        eligible, "included", np.where(~paired_feasible, "not_paired_feasible",
                                        "paired_feasible_but_recovery_failed")
    )
    save_csv(audit, "cohort_audit.csv")

    # Reproduce the published seed % 5 split before examining CV results.
    old_test = frame.seed.astype(int) % 5 == 0
    original_coefficients = fit_affine(frame.loc[~old_test])
    reproduced = predict(frame.loc[old_test], original_coefficients)
    reproduced_metrics = pd.DataFrame(metrics(reproduced))
    recorded_metrics = pd.read_csv(INPUTS[1]).set_index("predictor")
    assert len(reproduced) == 13 and int((~old_test).sum()) == 42
    for row in reproduced_metrics.itertuples():
        for metric in ["mae", "rmse"]:
            np.testing.assert_allclose(getattr(row, metric), recorded_metrics.loc[row.predictor, metric], atol=1e-12, rtol=0)
    recorded_predictions = pd.read_csv(INPUTS[2]).set_index("scenario_id").sort_index()
    reproduced_indexed = reproduced.set_index("scenario_id").sort_index()
    assert recorded_predictions.index.equals(reproduced_indexed.index)
    prediction_columns = [f"prediction_{key}" for key in PREDICTORS]
    max_prediction_error = float(np.max(np.abs(
        reproduced_indexed[prediction_columns].to_numpy(float)
        - recorded_predictions[prediction_columns].to_numpy(float)
    )))
    np.testing.assert_allclose(reproduced_indexed[prediction_columns], recorded_predictions[prediction_columns], atol=1e-12, rtol=0)
    save_csv(reproduced_metrics, "original_holdout_metrics_reproduced.csv")

    # Sorted scenario identifiers make assignment independent of CSV row order.
    # The seed is fixed once, with no seed selection or tuning from outcomes.
    permutation = np.random.default_rng(CV_SEED).permutation(len(frame))
    fold_indices = np.array_split(permutation, N_FOLDS)
    fold_records, fold_metrics, predicted_parts = [], [], []
    seen_test_ids: list[str] = []
    for fold, test_indices in enumerate(fold_indices, start=1):
        train_indices = np.setdiff1d(np.arange(len(frame)), test_indices)
        train, test = frame.iloc[train_indices], frame.iloc[test_indices]
        assert len(train) == 44 and len(test) == 11
        assert not set(train.scenario_id) & set(test.scenario_id)
        coefficients = fit_affine(train)
        part = predict(test, coefficients)
        part["fold"] = fold
        part["original_holdout"] = part.seed.astype(int) % 5 == 0
        predicted_parts.append(part)
        fold_metrics.extend(metrics(part, fold=fold))
        fold_records.append({
            "fold": fold, "n_train": len(train), "n_test": len(test),
            "soc_ols_intercept": float(coefficients[0]),
            "soc_ols_slope": float(coefficients[1]),
            "train_ids": sorted(train.scenario_id.tolist()),
            "test_ids": sorted(test.scenario_id.tolist()),
        })
        seen_test_ids.extend(test.scenario_id.tolist())
    assert len(seen_test_ids) == len(set(seen_test_ids)) == len(frame)
    assert set(seen_test_ids) == set(frame.scenario_id)
    oof = pd.concat(predicted_parts).sort_values("scenario_id").reset_index(drop=True)
    assert np.isfinite(oof[prediction_columns].to_numpy(float)).all()
    save_csv(oof, "oof_predictions.csv")
    save_csv(pd.DataFrame(fold_metrics), "fold_metrics.csv")

    # All predictors share each draw of whole-scenario rows. No within-scenario
    # time steps are treated as independent samples, and no models are refitted.
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(oof), size=(BOOTSTRAP_DRAWS, len(oof)))
    result_rows = metrics(oof)
    intervals = {}
    for row in result_rows:
        error = oof[f"error_{row['predictor']}"].to_numpy(float)
        samples = error[draws]
        boot_mae = np.abs(samples).mean(axis=1)
        boot_rmse = np.sqrt(np.mean(samples ** 2, axis=1))
        for metric, values in [("mae", boot_mae), ("rmse", boot_rmse)]:
            low, high = np.quantile(values, [0.025, 0.975], method="linear")
            row[f"{metric}_ci95_low"], row[f"{metric}_ci95_high"] = float(low), float(high)
        intervals[row["predictor"]] = boot_mae
    comparisons = []
    proposed_error = oof.absolute_error_counterfactual_cost.to_numpy(float)
    for key in PREDICTORS:
        if key == "counterfactual_cost":
            continue
        difference = oof[f"absolute_error_{key}"].to_numpy(float) - proposed_error
        paired_samples = difference[draws].mean(axis=1)
        low, high = np.quantile(paired_samples, [0.025, 0.975], method="linear")
        # This also checks identical resampling rows across predictors.
        np.testing.assert_allclose(paired_samples, intervals[key] - intervals["counterfactual_cost"], atol=1e-14, rtol=0)
        comparisons.append({
            "baseline": key, "reference": "counterfactual_cost", "n_cases": len(oof),
            "mae_improvement_baseline_minus_reference": float(difference.mean()),
            "ci95_low": float(low), "ci95_high": float(high),
            "bootstrap_unit": "scenario", "bootstrap_draws": BOOTSTRAP_DRAWS,
        })
    save_csv(pd.DataFrame(result_rows), "oof_metrics.csv")
    save_csv(pd.DataFrame(comparisons), "paired_mae_improvements.csv")
    after = {str(p.relative_to(EXPERIMENTS)): digest(p) for p in INPUTS}
    assert before == after, "An original source file changed during analysis."
    summary = {
        "analysis": "Retrospective supplementary five-fold OOF sensitivity analysis; not a new independent test",
        "cohort": {"all_scenarios": len(episodes), "attacked": len(attacked),
                   "attacked_paired_feasible": int(paired_feasible.sum()),
                   "finite_and_successful_pairs": len(frame), "old_holdout": len(reproduced),
                   "old_training": int((~old_test).sum()),
                   "paired_feasible_but_not_successful": int((paired_feasible & ~successful).sum()),
                   "original_holdout_noiseless": int((reproduced.noise_scale == 0).sum()),
                   "oof_noiseless": int((oof.noise_scale == 0).sum())},
        "fold_assignment": {"method": "Sort scenario_id; NumPy default_rng(seed).permutation(n); array_split into five folds",
                            "seed": CV_SEED, "n_folds": N_FOLDS,
                            "fit": "OLS intercept + energy-deficit slope, only on 44 training cases per fold",
                            "fixed_predictors": [key for key in PREDICTORS if key != "soc_ols"]},
        "bootstrap": {"unit": "whole scenario row shared across all predictors", "draws": BOOTSTRAP_DRAWS,
                      "seed": BOOTSTRAP_SEED, "interval": "95% percentile, NumPy linear quantiles",
                      "scope": "Conditional on fixed OOF predictions and the selected synthetic cohort; no refitting in bootstrap draws. Does not include training, model-selection, reference-state, model-mismatch, or deployment uncertainty."},
        "verification": {"original_holdout_mae_rmse_reproduced_atol": 1e-12,
                         "original_holdout_max_prediction_difference": max_prediction_error,
                         "original_source_files_unchanged": before == after,
                         "every_eligible_scenario_tested_exactly_once": True,
                         "each_training_and_test_fold_disjoint": True,
                         "seed_selected_before_examining_cv_results": True},
        "source_sha256": before, "analysis_sha256": digest(Path(__file__)),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "folds": fold_records, "oof_metrics": result_rows, "paired_mae_improvements": comparisons,
        "limitations": [
            "The cohort is selected on paired feasibility and realized recovery success; infeasible or failed pairs are excluded.",
            "The 55 cases reuse the original 42 training and 13 held-out cases; the retrospective CV is not independent confirmation.",
            "Only the affine energy-deficit predictor is fitted; fixed predictors are evaluated on the full eligible cohort.",
            "OOF errors share overlapping training folds; fixed-prediction bootstrap intervals are descriptive, not comprehensive CV risk uncertainty.",
            "Matched synthetic physics, known prices, common exogenous disturbances, and exact reference states favor the paired model.",
            "No hypothesis test, multiplicity correction, deployment claim, or claim of general statistical significance is made.",
        ],
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"cohort": summary["cohort"], "oof_metrics": result_rows,
                      "paired_mae_improvements": comparisons, "verification": summary["verification"]}, indent=2))


if __name__ == "__main__":
    main()
