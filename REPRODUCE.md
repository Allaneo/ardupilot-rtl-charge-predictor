# Reproduce the project

Verified on macOS 15.8 (Intel) with Python 3.11.16 and ArduPilot commit `d12a8f8997634cee977581cbaa5548c04facc1f5`. The simulator patch and fixed parameters are under `patches/` and `config/`. Run every command from the project directory.

## Environment

```sh
bash scripts/bootstrap_environment.sh
```

This creates `.venv`, clones ArduPilot if needed, checks the patch state, builds ArduCopter SITL and runs the unit tests. It expects `/usr/local/bin/python3.11` on this macOS setup.

## One example flight

Each flight needs a new, empty run directory:

```sh
.venv/bin/python scripts/run_sitl_flight.py \
  --run-dir data/raw/example-001 \
  --altitude-m 40 --distance-m 300 --bearing-deg 90 \
  --observation-window-s 10 \
  --payload-mass-kg 0.5 --wind-speed-m-s 2 --wind-direction-deg 90 \
  --scenario-id example-001 --run-id example-001 --speedup 2

.venv/bin/python scripts/process_flight.py \
  --run-dir data/raw/example-001 \
  --output reports/m8/example_row.json

.venv/bin/python scripts/reproduce_baseline.py \
  --flight-row reports/m8/example_row.json \
  --output reports/m8/example_prediction.json
```

The baseline script fits a Ridge model to the 40-flight training sample in `data/examples/train_f0_sample.csv` and predicts the new flight. SITL runs vary slightly, so the charge need not match the saved example exactly.

## Data

The processed datasets are included; the raw DataFlash logs (several GB) stay local.

| File | Flights | Manifest |
| --- | ---: | --- |
| `data/processed/pipeline_dataset.csv` | 250 development (172 train, 39 validation, 39 original test) | `data/manifests/pipeline_scenarios.csv` |
| `data/processed/headwind_training_dataset.csv` | 39 strong-headwind training | `data/manifests/headwind_training_scenarios.csv` |
| `data/processed/confirmation_dataset.csv` | 60 confirmation | `data/manifests/confirmation_scenarios.csv` |

To regenerate a dataset from scratch, generate its manifest, run its flights (restartable; `--jobs 2` runs two SITL instances), then process the logs. For the two new sets:

```sh
.venv/bin/python scripts/generate_confirmation_scenarios.py
.venv/bin/python scripts/generate_confirmation_scenarios.py --prefix train --seed 20261001 \
  --headwind-count 40 --general-count 0 --split train \
  --output data/manifests/headwind_training_scenarios.csv

.venv/bin/python scripts/run_manifest.py --manifest data/manifests/confirmation_scenarios.csv --jobs 2 --speedup 2
.venv/bin/python scripts/build_dataset.py --manifest data/manifests/confirmation_scenarios.csv \
  --output data/processed/confirmation_dataset.csv \
  --report data/manifests/confirmation_dataset_quality.json
```

The headwind training set is built the same way with its own manifest and output paths. A failed flight is archived and retried once with `scripts/requeue_failed_runs.py <run_id> --manifest <manifest>`. Keep the laptop awake during flights; `caffeinate -is` works on macOS.

Cruise-speed labels, used to train the speed models, are measured from the logs:

```sh
.venv/bin/python scripts/analyze_rtl_speed.py --split train --split validation --split test \
  --expected-count 250 --output reports/m19_training_speed_labels/rtl_speed_250.csv
```

## Results and the scripts that produce them

| Result | Script | Output |
| --- | --- | --- |
| Development models on F0/F1/F2 | `scripts/train_models.py`, `scripts/refit_frozen_candidates_trainval.py` | `reports/m6/`, `reports/m16_equal_data_all_candidates/` |
| Learned travel time on each input set | `scripts/compare_speed_features_f0_f1.py` | `reports/m12_speed_f0_f1/` |
| 450-network architecture sweep | `scripts/sweep_input_architectures.py`, `scripts/summarize_input_architectures.py` | `reports/m13_input_architecture_sweep/` |
| Speed diagnostics and measured-speed oracle | `scripts/analyze_rtl_speed.py`, `scripts/speed_oracle_check.py` | `reports/m9/`, `reports/m17_speed_oracle/` |
| Tilt-limit speed model | `scripts/tilt_limit_speed_check.py` | `reports/m18_tilt_limit_speed/` |
| Final confirmation evaluation | `scripts/evaluate_confirmation.py`, `scripts/confirmation_speed_accuracy.py` | `reports/m19_confirmation/`, `reports/m19_confirmation_speed/` |
| Cross-validated architecture sweep | `scripts/cv_architecture_sweep.py` | `reports/m20_cv_architecture_sweep/` |
| Final model, nested cross-validation | `scripts/nested_cv_final_model.py` | `reports/m21_nested_cv_final_model/`, `models/m21_final/final_model.joblib` (local) |
| Inputs × models on 349 flights | `scripts/cv_input_comparison.py` | `reports/m22_cv_input_comparison/` |
| Depth and epoch-cap check | `scripts/cv_depth_epoch_check.py` | `reports/m23_depth_epoch_check/` |

Scripts that score held-out flights refuse to overwrite their output directory. The confirmation evaluation was run once; to repeat it, work in a separate checkout rather than deleting the saved report. The protocol it follows is recorded in `DECISIONS.md` (D029 and D031) before any confirmation flight was scored.

To retrain the development models from the included CSV without touching the saved reports:

```sh
.venv/bin/python scripts/train_models.py \
  --output-dir .cache/reproduced-m6 \
  --models-dir .cache/reproduced-models
```

The cross-validation scripts use four processes; set `OMP_NUM_THREADS=1` (and the equivalent BLAS variables) so they do not oversubscribe the CPU. The final model file holds the tilt-limit speed model, the three seed networks and usage notes.

The full unit test suite runs with `.venv/bin/python -m pytest -q`.
