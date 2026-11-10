# Repaying Physical Consequence Debt after Poisoned Energy-Agent Decisions

Code, analysis plans and saved results of a simulation study on the cost of recovering a battery system after its
controller has acted on a falsified price forecast.

A battery controller that plans from forecasts can be misled by a falsified price forecast into discharging energy
that it later needs. Correcting the forecast after detection does not undo the executed actions. The study defines the
**physical consequence debt**

    ΔJ = V(e_a) − V(e_r),

the difference between the lowest costs of completing the same recovery task from the attacked battery level `e_a` and
from the level `e_r` of a paired reference run without the attack. It tests whether ΔJ predicts the extra cost that is
realised after detection, and whether giving ΔJ to the controller changes its decisions.

## Main results

| Data set | Scenarios used | ΔJ | Price at detection | Cheapest hours | Target level |
| --- | ---: | ---: | ---: | ---: | ---: |
| Synthetic test set (USD) | 329 | 0.014 | 0.257 | 0.213 | 0.810 |
| Synthetic test set, meter error 0.60 kWh (USD) | 112 | 0.267 | 0.377 | 0.329 | 0.979 |
| Measured household profiles (GBP) | 352 | 0.0067 | 0.064 | 0.063 | 0.375 |

Mean absolute error of each estimate of the realised extra cost. The cheapest-hours and target-level estimates were
added after the tests; the other estimates and the decision rules were fixed before them.

- ΔJ predicts the realised extra cost with a mean absolute error of about 1% of its mean on the synthetic test set.
- Adding ΔJ to the objective of the controller cannot change its decisions: the actions are identical in all 15,732
  recovery steps of the test set and in all 12,584 steps of the measured-profile set.
- Under battery-meter error, recovery improves with a planning reserve: at an error of 0.60 kWh, full recoveries rise
  from 123 to 393 of 500 scenarios.

## Repository layout

```
experiments/
  src/recovery/                 battery model, attack, recovery methods, optimisation (linear programme, HiGHS)
  tests/                        16 automated tests
  run_experiments.py            development set (100 scenarios) and controlled sweeps
  validate_results.py           independent check of saved results
  results/main/                 results of the development set
  expanded_validation/          test set of 500 new scenarios
  noise_validation/             battery-meter error and planning reserve
  real_data_validation/         measured household profiles with a dynamic tariff
  llm_agent_validation/         language-model agent (Qwen2.5-3B-Instruct)
  supplementary/prediction_cv/  cross-validation of the development-set predictions
assignment 3/
  code/                         scripts that recompute every reported number and draw the figures
  build/                        recomputed numbers (JSON)
  figures/                      Figs. 1-6
```

Each experiment folder holds the script that produced its results, the saved results, and its analysis plan
(`protocol*.json`) with a SHA-256 checksum (`protocol*.sha256`) that was stored before the experiment was run.

## Installation

The code needs Python 3.12 and [uv](https://docs.astral.sh/uv/). All commands are run in the repository folder.

```bash
uv sync --project experiments --frozen --python 3.12
uv run --project experiments pytest experiments/tests -q
```

## Reproducing the reported numbers and figures

The numbers and figures are recomputed from the saved results.

```bash
uv run --project experiments python "assignment 3/code/a3_numbers.py"
uv run --project experiments python "assignment 3/code/stronger_comparators.py"
uv run --project experiments python "assignment 3/code/check_measured_actions.py"

cd "assignment 3/code"
for name in fig2_example_scenario fig3_prediction fig4_detection_delay fig5_recovery_boundary fig6_sensor_error; do
    uv run --project ../../experiments python $name.py
done
cd ../..
uv run --project experiments --with cairosvg python "assignment 3/code/fig1_model_framework.py"
```

| Script | Output | Time |
| --- | --- | --- |
| `a3_numbers.py` | `assignment 3/build/a3_numbers.json`: every reported number; stops if a recomputed value differs from the summaries saved by the experiments | seconds |
| `stronger_comparators.py` | `assignment 3/build/stronger_comparators.json`: the cheapest-hours and target-level estimates; stops if a rebuilt scenario does not reproduce the saved ΔJ | about 10 s |
| `check_measured_actions.py` | `assignment 3/build/measured_actions.json`: step-by-step comparison of the two controllers on the 400 measured-profile scenarios | about 2 min |
| `fig1_*.py` to `fig6_*.py` | `assignment 3/figures/` | seconds |

## Re-running the experiments

The development set and the controlled sweeps can be repeated into a new folder:

```bash
uv run --project experiments python experiments/run_experiments.py --episodes 100 --output experiments/results/reproduced
uv run --project experiments python experiments/validate_results.py --results experiments/results/reproduced
```

The scripts of the other experiments (`run.py`, and `reserve.py` for the reserve controllers) store their plan before
they run and refuse to run while that plan exists, so that saved results cannot be overwritten. To repeat one of them,
remove the `protocol*.json`, `protocol*.sha256` and `results/` of that experiment in a copy of the repository and run
its script. The language-model experiment also needs a GPU, the packages in
`experiments/llm_agent_validation/requirements_llm.txt` and the model weights, which
`experiments/llm_agent_validation/download_model.py` downloads at a fixed revision.

## Data

- **Synthetic scenarios** are generated from fixed random seeds by `experiments/src/recovery/model.py`.
- **Measured profiles** (`experiments/real_data_validation/data/real_profiles.csv`) combine the grid import of six
  homes and the output of four solar systems in southern Germany with half-hourly electricity prices.
  `prepare_data.py` builds the file from the two sources, and `data_metadata.json` records the series, the period, the
  plant sizes and the checksums of the source files.
  - Open Power System Data, *Data Package Household Data*, version 2020-04-15, primary data from the project CoSSMic,
    licence CC BY 4.0: <https://data.open-power-system-data.org/household_data/2020-04-15/>
  - Octopus Energy, *Agile Octopus* tariff AGILE-24-10-01, half-hourly unit rates of region C from 5 April to
    9 October 2026, from the supplier's public interface: <https://api.octopus.energy/v1/products/AGILE-24-10-01/>
- **Language model**: Qwen2.5-3B-Instruct, used unmodified. The weights are not part of this repository.

## Citation

```
F. Guo, "Repaying Physical Consequence Debt after Poisoned Energy-Agent Decisions," 2026.
https://github.com/Flowming-ai/REM
```
