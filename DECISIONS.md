# Decisions

## D001: Predict charge, not the operational decision

- Date: 2026-09-21
- Status: accepted

The ML target is charge required to complete RTL in mAh. Remaining-battery comparison and continue/return/land logic are deterministic downstream calculations and are out of scope.

## D002: Use one sample per complete simulated flight

- Date: 2026-09-21
- Status: accepted

Inputs are frozen at RTL initiation using the current state and a pre-RTL observation window. Telemetry after RTL initiation is used only to create labels and diagnostics.

## D003: Keep centred payload as the only hidden vehicle condition

- Date: 2026-09-21
- Status: accepted

Motor degradation and CG offset were removed. Centred payload is retained because it changes required thrust and can be inferred indirectly from recent onboard symptoms.

## D004: Do not use frame reference mass as payload

- Date: 2026-09-21
- Status: accepted

Changing the frame model's reference mass also affects propulsion-model calibration. The implementation will instead add a simulation-only external payload mass, then validate that zero payload reproduces the unmodified simulator.

## D005: Preserve a deployable/oracle wind distinction

- Date: 2026-09-21
- Status: accepted

Deployable models use EKF-estimated wind. True simulated wind is privileged information used only for estimator validation and explicitly labelled oracle models.

## D006: Use topology-independent motor summaries

- Date: 2026-09-21
- Status: accepted

Raw motor outputs may be retained for diagnostics, but model inputs use fixed-length summaries so a later quad-to-hexa experiment does not require changing the input dimension.

## D007: Establish an unmodified SITL baseline first

- Date: 2026-09-21
- Status: accepted

The environment, flight automation and baseline RTL must work with unmodified ArduPilot before any payload patch is introduced.

## D008: Do not run the broad macOS prerequisite installer by default

- Date: 2026-09-21
- Status: accepted

ArduPilot's macOS helper updates Homebrew, installs global packages and can edit shell startup files. The project will first build with the existing compiler and a project-local Python environment. Any missing system dependency will be identified and installed separately only if required.

## D009: Add centred mass through the existing external-payload accumulator

- Date: 2026-09-22
- Status: accepted

The ArduPilot patch adds `SIM_PAYLOAD_MASS` and initializes the existing `external_payload_mass` accumulator from it. Existing sprayer and gripper payloads remain additive. The value affects `gross_mass()` in the force-to-acceleration calculation but does not alter frame reference mass, thrust calibration, aerodynamic drag or rotational inertia.

Matched flights validated the implementation at 0.0, 0.5 and 1.0 kg. At zero payload, patched and unmodified SITL differed by 1 mAh of RTL charge, 0.0009 A of hover current and 0.00002 of normalized hover throttle. Hover current, hover throttle and RTL charge then increased monotonically with payload. The provisional engineering range is therefore 0.0 to 1.0 kg for the 3 kg base frame, subject to the later full operating-envelope tests.

## D010: Use a normal wheel install in the project environment

- Date: 2026-09-22
- Status: accepted

Python 3.11.16 skips filesystem-hidden `.pth` files. macOS repeatedly marked the editable-install `.pth` inside the dot-named virtual environment as hidden, making imports unreliable. Bootstrap therefore installs the project as a normal wheel and pins Python 3.11 or newer.

## D011: Freeze the first wind envelope at 0 to 6 m/s

- Date: 2026-09-22
- Status: accepted

The pinned quad model's suggested drag coefficients, `EK3_DRAG_BCOEF_X/Y=17.209` and `EK3_DRAG_MCOEF=0.209`, are fixed across payload values. Six successful engineering flights spanning calm conditions, multiple wind directions, 0 to 1 kg payload and speeds through 6 m/s produced a mean EKF wind-vector error of 0.47 m/s and a maximum of 0.70 m/s at the RTL decision window.

An 8 m/s, 0.5 kg engineering case returned home and touched down but did not automatically disarm within 240 seconds. It is retained as a failed run and not used as a regression sample. The first dataset therefore uses 6 m/s as its wind maximum. This is an evidence-based operating-envelope limit, not an estimator-accuracy limit; the 8 m/s estimate itself was available, but the full-flight completion criterion failed.

## D012: Use DataFlash events and two independent electrical calculations

- Date: 2026-09-22
- Status: accepted

The parser defines `t0` as the first DataFlash `MODE` event with `ModeNum=6` and defines completion as the first later `ARM` event with `ArmState=0`. Deployable inputs come only from the half-open 10-second window `[t0-10 s, t0)`. Post-`t0` telemetry is restricted to labels and diagnostics.

The primary `rtl_charge_mah` label is the interpolated difference in `BAT.CurrTot` between those two events. An independent trapezoidal integral of `BAT.Curr` must agree within the larger of 2 mAh or 1% of the logged label. The flight runner's integer MAVLink charge difference is retained as a third diagnostic, not as the highest-resolution target.

Across 14 completed acceptance flights, the largest difference between the primary label and independent integral was 0.481 mAh. The retained incomplete 8 m/s flight is rejected before feature construction because automatic disarm was not observed.

## D013: Freeze the pipeline request manifest before bulk flight generation

- Date: 2026-09-22
- Status: accepted

The 250 pipeline scenarios are generated once from seed `20260921` using independently shuffled Latin-hypercube columns. Their requested values, family-level splits, fingerprints and ArduPilot/patch/parameter provenance are stored before flight execution. The manifest is overwrite-protected by default and updated atomically after each run.

The frozen allocation is 172 train, 39 validation and 39 test flights. Bearing quadrants contain 62, 63, 63 and 62 flights, and the maximum absolute pairwise correlation among sampled nonconstant variables is 0.180. Generation and fingerprint reproduction are automated quality gates.

## D014: Reduce runtime without changing the frozen flight scenarios

- Date: 2026-09-22
- Status: accepted

The manifest runner now starts the next pending flight when any instance becomes free, while preserving per-run directories, ports, atomic status writes and failure records. Dataset construction caches each successfully parsed row against hashes of its raw log, result JSON and processing source, and parses cache misses in separate processes. An 83-row unchanged rebuild used 83 cache hits and took 6.21 seconds; all eight data-quality gates passed.

The SITL launcher accepts a configurable speedup, records it in each run result, and measures the observation window using vehicle boot time. A matched pipeline scenario was repeated at 1x and 2x, in addition to its original 1x run. All three completed automatic disarm; the repeat and 2x runs produced 1422.305 and 1422.748 mAh, their DataFlash RTL durations were 143.837 and 143.885 s, and each had 100 decision-window battery samples. Their logged-charge cross-check errors were 0.358 and 0.360 mAh. The measured wall RTL phase fell from 144.1 s at 1x to 71.9 s at 2x. Use 2x for the remaining pipeline flights, subject to the existing completion and dataset quality gates.

## D015: Keep M6 selection strictly within train and validation

- Date: 2026-09-23
- Status: accepted

Fit scalers and estimators on the 172 training flights. Use only the 39 validation flights for bounded hyperparameter selection and early stopping. The 39 test-flight labels remain unopened until M7. Deployable candidates use the explicit F0/F1/F2 allowlists; payload truth, simulated wind truth and post-decision diagnostics are excluded. Report the true-wind oracle separately, and recompute every wind-derived physics feature from true wind for that comparison.

## D016: Implement the small neural network transparently

- Date: 2026-09-23
- Status: accepted

Use the planned 64/32 ReLU multilayer perceptron with a linear output, train-only feature and target standardization, Adam updates, MAE/Huber candidate losses, L2 regularization and validation-MAE early stopping. Implement it in NumPy so the course-level forward pass, backpropagation and optimizer remain inspectable without adding a heavyweight framework dependency. Compare it with ridge and XGBoost rather than presuming it will win.

## D017: Freeze and evaluate the held-out test set once

- Date: 2026-09-23
- Status: accepted

After all candidate configurations and model artifacts were frozen using training/validation only, perform one M7 evaluation on the fixed 39-flight test split. Score the validation-selected baselines, F0/F1/F2 models and the true-wind oracle diagnostics, but do not select or tune based on test results. Quantify uncertainty with paired bootstrap resampling over complete flights. Subsequent plots and condition breakdowns must use saved predictions and may join only non-target condition fields; do not rerun the scoring step.

## D018: Keep the 250-flight analysis labelled as a pilot

- Date: 2026-09-23
- Status: superseded by D019 after user review

Section 11.3 of `PLAN.md` distinguishes an approximately 250-flight pipeline set from an approximately 3,000-flight main set. The 250-flight data and its 39-flight held-out evaluation are a complete pilot analysis, not completion of the original main-study milestone. The pilot test scores have now been seen, so they cannot remain a blind final test for subsequent development. If the main study proceeds, preassign a new independent test split, freeze its manifest and analyze that split only after its model-selection procedure is fixed. Package the pilot honestly while keeping the unrun main study visible in the progress log and course report.

## D019: Stop at 250 flights for the Course 2 project

- Date: 2026-09-23
- Status: accepted at user's direction

The approximately 3,000-flight stage was introduced in planning without an adequate discussion of its simulation cost. The user does not consider that larger campaign achievable or educationally useful for this course project. The completed 250-flight dataset is therefore the final Course 2 study dataset, subject to explicit limitations from its 39-flight test split. No further bulk simulation is planned. Preserve the original `main_flights: 3000` field in the frozen experiment file as a historical planning value, but do not treat it as an active target. More flights could narrow statistical uncertainty, but that alone does not advance the chosen course-learning objective enough to justify the work.

The F0/F1/F2 feature sets and Ridge baseline were also implemented without first reviewing their educational framing with the user. They are valid recorded experiments, but their role and presentation in the final course narrative remain for user review. Do not redesign the experiments or reinterpret the held-out results silently after seeing test scores. Future educational design choices should be surfaced before implementation.

## D020: Treat controller-aware feature polish as exploratory

- Date: 2026-09-23
- Status: accepted at user's direction

After reviewing the inconsistency between the saved F2 cruise-time and required-airspeed assumptions, the user approved a bounded follow-up using the existing training/validation logs only. Measure middle-route RTL ground speed from those logs; supervise a small speed model using decision-time telemetry; cross-fit its training-row predictions before feeding them to a charge model; and compare physics features with and without a cubic airspeed proxy on validation flights. Do not run new SITL flights, change the frozen F0/F1/F2 artifacts, or use the already-viewed test set to claim an unbiased gain from this new design. Keep the course-report draft paused pending discussion of educational framing.

## D021: Version the corrected nominal F2 inputs

- Date: 2026-09-23
- Status: accepted at user's direction for feature preparation; model-training design remains open

The user requested removal of the zero climb-time input and correction of the inconsistent F2 travel-time/required-airspeed assumptions. Preserve the original F2 configuration, processed dataset and M6/M7 artifacts as a reproducible historical experiment. An opt-in revised F2 builder removes both constant climb inputs and computes horizontal time and required airspeed from the same 10 m/s target-ground-speed assumption. This is internally consistent but is still a nominal approximation: strong headwinds can prevent the controller from achieving 10 m/s. It is not a substitute for the separately evaluated two-stage F3 speed predictor. Do not train or compare revised model scores until the user agrees on the complete input sets and architecture protocol.

## D022: Retain Ridge as an explicitly additional baseline

- Date: 2026-09-23
- Status: accepted at user's direction

Ridge remains in the project because it provides a useful regularized linear reference, especially with correlated voltage/current inputs. It is not presented as a Course 2 technique. The Course 2 comparison is the neural network versus decision-tree ensembles; engineering formulas remain separate baselines. Any new training matrix must label Ridge explicitly as an additional out-of-course baseline and must be shown to the user before execution.

## D023: Include F3 in the input comparison and check Ridge speed first

- Date: 2026-09-23
- Status: accepted at user's direction for the bounded speed and charge check

The user requested that F3 be included in the upcoming input comparison, with a Ridge cruise-speed predictor evaluated before using its predicted speed in the charge model. This supersedes the earlier suggestion to leave F3 outside that comparison. On the same 172 training and 39 validation flights, using the five M9 decision-time speed inputs, Ridge alpha 1.0 was selected by five-fold training cross-validation. Its validation speed MAE was 0.454 m/s versus 0.837 for a fixed 10 m/s speed and 0.251 for the saved random-forest speed model. In 14 flights with estimated headwind at least 2 m/s, the errors were 0.894, 2.256 and 0.608 m/s respectively. A bounded downstream F3 Ridge charge check with cross-fitted training speed predictions scored 40.7 mAh validation MAE and a 416.3 mAh worst underprediction; the saved random-forest-speed F3 Ridge result scored 37.6 and 414.5, and frozen F2 Ridge scored 41.9 and 277.6. F3 belongs in the upcoming comparison, but its speed-model choice and strong-headwind failure must be explicit. The complete neural architecture and charge-model matrix is still to be agreed before running it.

## D024: Prepare a configurable neural architecture study

- Date: 2026-09-23
- Status: candidate catalog prepared; full sweep not run

The user asked to include 64–32–16 and 128–64–32–16 in addition to the previously proposed widths and depths. The implementation now accepts an arbitrary positive-width hidden-layer tuple. The prepared catalog has 15 shapes: single-layer widths 16, 32, 64, 128, 256, 512 and 1024; tapered 32–16, 64–32, 128–64, 128–64–32, 64–32–16 and 128–64–32–16; and five or ten 16-neuron layers. This catalog enables a width/depth comparison; it does not imply that any size is known in advance to be too large. Use the same train/validation split and report the pattern across seeds rather than selecting a winner from one small validation difference. The original 64–32 default and saved M6/M7 artifacts remain reproducible.

## D025: Use the better validated speed predictor for F3 and recalculate dependent inputs

- Date: 2026-09-23
- Status: accepted at user's direction for validation-only feature comparison

The user chose whichever cruise-speed predictor performs better and asked whether its estimate could improve all inputs derived from speed or travel time. The M9 random forest has lower validation speed MAE than Ridge (0.251 versus 0.454 m/s), so it is the current F3 speed source. A controlled train/validation comparison rebuilt seven F2 fields from either a fixed 10 m/s ground speed or the random forest's predicted speed: horizontal and total time, north/east air velocity, airspeed and its square, and current-times-time charge. The other 38 inputs were checked equal across variants. With fixed Ridge charge settings, validation MAE changed from 72.9 to 47.5 mAh; with fixed 64–32 neural settings, from 50.7 to 48.8 mAh. In the 14 high-headwind validation flights, Ridge MAE changed from 170.8 to 87.0 mAh. Large worst underpredictions remain, and the already-viewed test split was not used. These are exploratory validation results, not a new final model selection.

## D026: Check learned travel time in F0 and F1 separately

- Date: 2026-09-23
- Status: exploratory validation check at user's direction

The user clarified that learned cruise speed should also be tested as an addition to historical F0 and F1, not only in F2/F3. Use a separate speed forest for each set so F0 does not silently receive F1's current/throttle information: F0's speed inputs are route/wind-derived quantities plus mean voltage; F1's also use current and throttle. Derive F0 speed inputs only from fields already available in F0, and cross-fit training-row speed predictions before charge fitting. Compare the unchanged input set, speed alone, fixed 10 m/s travel time, predicted-speed travel time, and speed plus predicted travel time with fixed Ridge and 64–32 neural settings. Use only the 172 training and 39 validation flights.

Validation results: the F0/F1 speed forests had 0.252/0.251 m/s speed MAE. Adding speed alone worsened charge MAE for both charge models and both input sets. Adding `distance_home_m / predicted_speed` improved Ridge charge MAE from 77.5 to 55.5 mAh on F0 and from 71.2 to 55.2 on F1, while `distance_home_m / 10` barely changed Ridge MAE (76.9 and 70.7). With both speed and predicted time, Ridge scored 59.4 on F0 and 54.2 on F1; the fixed 64–32 neural scored 52.1 and 53.0. These are validation-only exploratory comparisons, not a new frozen winner. The worst charge underprediction remained above 450 mAh in each improved neural variant and above 580 mAh for improved Ridge variants. No test labels or new SITL flights were used.

## D027: Complete the agreed input and neural-architecture comparison

- Date: 2026-09-23
- Status: completed as exploratory validation analysis

The user approved comparing fixed 10 m/s with learned speed wherever speed or travel time enters an input set. The completed matrix has ten variants: original F0/F1, each with fixed-time and learned-time additions; corrected F2 with fixed/learned speed; and F3 with the cubic proxy under fixed/learned speed. F1's learned variant includes both predicted speed and travel time; its fixed variant includes only distance/10 because a constant 10 m/s column contains no information. F0's learned variant uses only F0-available information in its speed predictor. The F2/F3 learned variants use the F1-available speed predictor. Training-row speed estimates are five-fold out-of-fold.

Each variant received all 15 previously prepared neural shapes with three seeds (450 neural fits total), plus one fixed Ridge alpha 1.0 and one fixed XGBoost reference (20 reference fits). The neural optimizer, Huber loss, L2 penalty, maximum epochs and early-stopping rule were held constant. The 172 training and 39 validation flights were unchanged; no SITL run or held-out test scoring occurred. All 470 saved validation metrics were independently recomputed from saved predictions.

Matched fixed-versus-learned comparisons favored learned speed in 42/45 F0 neural runs, 29/45 F1 runs, 34/45 F2 runs and 27/45 F3 runs. These are counts over architectures and seeds on the *same* 39 validation flights, not independent generalization trials. Ridge validation MAE changed from 76.9 to 55.5 mAh (F0), 70.7 to 54.2 (F1), 72.9 to 47.5 (F2), and 41.1 to 37.6 (F3). The lowest three-seed mean validation MAE among the 15 neural shapes was 30.3 mAh for F3 learned speed with one 1,024-unit layer; its mean training MAE was 7.0 mAh and mean worst validation underprediction was 297.2 mAh. The large train/validation gap and already-reused 39-flight validation set preclude calling it a confirmed winner or safe predictor. The original M7 test result remains frozen. The F2 and F3 feature constructions differ beyond the cubic term, so their cross-family difference must not be attributed solely to that term.
## D028: Post-hoc held-out check for the revised candidates

- Date: 2026-09-23
- Status: completed; post-hoc evidence recorded

The user requested a final test of the revised candidate after the architecture sweep. Score a predeclared small set: F3 learned-speed 1,024-unit neural networks with all three sweep seeds, F3 learned-speed Ridge, corrected F2 learned-speed 256-unit neural networks with all three sweep seeds, corrected F2 learned-speed Ridge, and the already frozen M7 F2 Ridge reference. The candidate architectures and charge hyperparameters come from train/validation only; no candidate will be selected from the new test numbers.

For the revised inputs, fit the auxiliary speed forest on the 211 train/validation speed labels and apply it to the test decision-window inputs. Generate cross-fitted speed predictions for the 211 charge-training rows. Do not use test speed labels. Fit the charge models on all 211 train/validation flights using each seed's validation-selected epoch as a fixed epoch count, then score the 39 test charge labels. Preserve the original M7 report and label this as a post-hoc revised-candidate check: the test flights were already used once for M7, so it is informative but not a fresh untouched-study claim.

Results: frozen M7 F2 Ridge scored 31.0 mAh test MAE; F3 learned-speed Ridge scored 25.4 mAh; its three 1,024-unit neural seeds scored 26.3, 17.7 and 21.4 mAh. F2 learned-speed Ridge scored 32.5 mAh; its three 256-unit neural seeds scored 25.6, 35.4 and 27.6 mAh. Relative to frozen F2 Ridge, the paired flight-bootstrap intervals were -12.0 to +0.25 mAh for F3 Ridge, -14.8 to +5.0 for F3 neural seed 20260924, -21.3 to -6.0 for seed 20260925, and -17.8 to -1.8 for seed 20260926. F3 neural seed 20260925 had the lowest observed test MAE, but seed sensitivity means it is not a universal winner. The observed test results support learned-speed features, especially F3, while remaining descriptive because this test set was already used once for M7.

## D029: Fresh confirmation set with a predeclared protocol

- Date: 2026-09-23
- Status: accepted at user's direction; protocol fixed before any confirmation flight was generated or run

This amends D019. By M18, the 39 test flights had been scored several times and the 39 validation flights had guided M9–M18. The tilt-limit speed model and learned-speed inputs therefore need new, untouched flights. The user approved 60 new flights.

Confirmation flights. Prefix `confirm`, seed 20260930, same pinned simulator, patch, parameter file, 10-second decision window, 2x speedup and completion rules as the pipeline. Group H (30 flights) oversamples strong headwind on the return. Wind speed is 4–6 m/s, and wind direction is the return heading plus an offset of −30° to +30° (wind coming from the direction of travel home), giving at least 3.46 m/s true headwind. Group G (30 flights) is a Latin hypercube over the full frozen envelope, drawn like the pipeline. Other variables are Latin hypercube samples over the frozen envelope in both groups. All 60 rows are labelled `test`. A flight that does not complete RTL with automatic disarm is reported, retried once with the same scenario and excluded only if it fails again; exclusions are reported.

Training data. All 250 existing flights (former train, validation and test). Cruise-speed labels are measured for all 250 with the M9 method. No confirmation label or log is read before the single final scoring run.

Speed models, fitted on the 250 flights. Random forest: M9 settings and inputs. Tilt-limit: maximum airspeed linear in decision-window throttle, fitted on flights with more than half their middle route at ≥29.5° lean; ground speed = min(10, sqrt(V_max² − crosswind²) − headwind), clipped to at least 1 m/s. Training-row speeds for the charge models are five-fold out-of-fold (fold seed 20260923).

Charge candidates, fixed now:
- A: original F2, Ridge alpha 1.0.
- B: original F2, 64–32 network (Huber, L2 0.001, Adam learning rate 0.001), seeds 20260924–20260926, reported as the three-seed mean prediction.
- C: F3 with tilt-limit speed, Ridge alpha 1.0.
- D: F3 with tilt-limit speed, 1,024-unit network, same training settings and seeds as B, reported as the three-seed mean prediction.
- E: F3 with random-forest speed, Ridge alpha 1.0.
- Reference: nominal time × current baseline.

Epoch counts for B and D are chosen without the confirmation set. For each seed, run five-fold cross-validation on the 250 training flights with early stopping (patience 45, maximum 600). The final model is then trained on all 250 flights for the median best epoch. Individual seeds are reported descriptively.

Primary comparisons (paired flight bootstrap, 10,000 resamples, negative favours the first model): C − A, D − B and C − E. Metrics: MAE with bootstrap interval, RMSE, mean, 95th-percentile and maximum underprediction, MAE by group (H, G) and by estimated headwind ≥ 3 m/s. Speed-model accuracy (fixed 10 m/s, forest, tilt-limit) is also reported against measured confirmation cruise speed, extracted after scoring. The scoring script refuses to overwrite its output and is run once. No candidate, hyperparameter or feature is changed after confirmation results are seen.

## D030: Calibrated reserve margin (dropped)

- Date: 2026-09-23
- Status: dropped on 2026-09-25 at the user's direction

A reserve-margin rule was briefly added to the confirmation protocol. It has been removed from the project: the reported results are the mean predictions only, and sizing a reserve is left as a recommendation in the course report.

## D031: Add strong-headwind training flights before confirmation scoring

- Date: 2026-09-23
- Status: accepted at user's direction; added before any confirmation result was computed

The 250 development flights contain only 17 with at least 4 m/s return headwind, too few to train the charge models where they fail. Add 40 new training flights with prefix `train-h` and seed 20261001, using the same design as confirmation group H: wind 4–6 m/s from the return heading ±30°, other variables Latin hypercube over the frozen envelope, same simulator, patch, parameters, decision window, speedup and completion rules. All 40 are labelled `train`; failures are retried once and reported.

This amends D029 only in the training set. All D029 training steps (speed models, out-of-fold speeds, cross-validated epochs) use the 290 flights formed by the 250 development flights plus these 40. Cruise-speed labels are measured for the 40 new flights with the M9 method. The 60 confirmation flights stay untouched test data, and the candidates, comparisons, metrics and single scoring run are unchanged.

## D032: Cross-validated architecture sweep, then a nested cross-validated final model

- Date: 2026-09-24
- Status: accepted at user's direction; recorded before either analysis was run

The confirmation set has been used once. The architecture used there (one 1,024-unit layer) was chosen from a sweep on 172 training and 39 validation flights. Two follow-ups use cross-validation only; no new flights are needed and no held-out claim is made.

1. Architecture sweep (M20). Data: the 289 training flights (250 development + 39 strong-headwind), confirmation flights excluded. Inputs: F3 with tilt-limit speed. Candidates: the same 15 network shapes, seeds 20260924–20260926, Huber loss, L2 0.001, Adam learning rate 0.001, plus Ridge alpha 1.0. Outer five-fold cross-validation (fold seed 20260923). Inside each outer fold, the tilt-limit speed model is fitted on the outer-training flights only (out-of-fold speeds for those flights), the network's epoch count comes from early stopping on a fixed 20% of the outer-training flights (patience 45, at most 600 epochs), and the network is then refitted on all outer-training flights for that many epochs. Report out-of-fold MAE per shape from the three-seed mean prediction, the spread across seeds, strong-headwind MAE (estimated headwind at least 4 m/s), worst underprediction, and paired flight-bootstrap differences against the 1,024-unit shape and against Ridge.

2. Final model (M21). Data: all 349 flights. Nested five-fold cross-validation estimates the accuracy of the whole procedure "choose between Ridge and the network shapes by inner five-fold cross-validation, then refit". The inner candidate set is fixed after M20 and reported with it. The final model is that procedure applied to all 349 flights. Its accuracy is the nested estimate, not a held-out test.

Addendum after M20: the M21 inner candidate set is Ridge alpha 1.0 and three networks, each reported as its three-seed mean: 256 units (lowest M20 cross-validated MAE), 1,024 units (the confirmation architecture) and 64–32–16 (best multi-layer shape). The other shapes were within noise of these or clearly worse. Networks use the M20 recipe (early stopping on a fixed 20% of the training flights, then refit). M20 used 289 of the 349 M21 flights, so choosing this candidate set makes the nested estimate slightly optimistic; this is stated with the result.

## D033: Input comparison with the cross-validated architecture

- Date: 2026-09-24
- Status: accepted at user's direction; recorded before M21 finished and before this analysis was run

The earlier input comparison used a 64–32 network chosen on 39 validation flights. Repeat it with the architecture supported by cross-validation, so the question "do inputs or architecture matter more?" is answered on equal footing.

- Architecture: M21's final choice if it is a network; if M21 chooses Ridge, the lowest-MAE network in M20 (256 units). Same recipe as M20: Huber loss, L2 0.001, Adam 0.001, early stopping on a fixed 20% of the training flights, refit, three-seed mean (seeds 20260924–20260926).
- Inputs: F0, F1, F2 and F3 with tilt-limit speed. F3's speed model is fitted inside each fold.
- Models: that network, Ridge alpha 1.0, and XGBoost with the M6 F2 settings (depth 3, 300 trees, learning rate 0.1, subsample 1.0, column sample 0.8, minimum child weight 5, seed 20260928).
- Data and evaluation: all 349 flights, five-fold cross-validation (fold seed 20260923). Report MAE with a flight-bootstrap interval, strong-headwind MAE (at least 4 m/s estimated headwind), worst underprediction, and paired differences between input sets within each model and between models within each input set. No held-out claim; no selection from these results.

## D034: Check whether the epoch cap held back deeper networks

- Date: 2026-09-24
- Status: accepted; recorded before running

In M20 several multi-layer shapes stopped at or near the 600-epoch cap (median 582 for 64–32–16, 597 for 5 × 16), so the fixed recipe may have undertrained them. Repeat M20 exactly (289 training flights, same folds, F3 tilt-limit inputs, same seeds, loss, L2, learning rate, 20% early-stopping split and patience 45) for 256, 1,024, 64–32, 128–64, 64–32–16, 128–64–32, 128–64–32–16, 5 × 16 and 10 × 16, changing only the epoch cap from 600 to 3,000. Report MAE, median stopping epoch and paired differences against the same shape under the 600 cap. This checks one training setting; it is not a new architecture search, and M21's final model is not changed by it.
