# Predicting multicopter return charge in ArduPilot SITL

## Summary

At the moment an ArduPilot quadcopter is commanded to return home (RTL), how much battery charge will the return use? The final model, a neural network with one hidden layer of 256 units, predicts it within 21.6 mAh on average, about 1.2% of the charge, by nested cross-validation on all 349 simulated flights. In an earlier untouched test on 60 new flights, half of them in strong headwind, the best models were within 38–39 mAh (about 2%), against 196 mAh for a simple time × current rule.

The key to the result was a physical insight about the flight controller. In strong headwind the vehicle reaches its 30° lean limit and cannot hold its 10 m/s return speed. At that limit its airspeed is roughly fixed, and heavier vehicles fly faster into the wind. A two-parameter formula built on this predicts return speed, and return speed drives most of the charge.

Inputs mattered more than architecture. Physics-based inputs cut Ridge and XGBoost errors by two thirds or more, while networks of 128 units or more, with one layer or several, all performed within about 2.5 mAh of each other. The network beat Ridge and XGBoost on every input set. Everything here is simulation of one vehicle in a bounded envelope; it is not a safety guarantee.

## Question and setup

The model predicts the charge in mAh from RTL start to automatic disarm, using only information the vehicle has when RTL is commanded. Comparing that prediction with remaining charge is simple arithmetic and outside the machine-learning task.

- **Simulator.** ArduPilot ArduCopter SITL at a pinned commit, one 3.0 kg quad frame, EKF3 as the onboard estimator. Each flight takes off, flies out on a set bearing and distance, records a 10-second decision window, then returns and lands.
- **Hidden payload.** A small simulation-only patch adds a centred payload mass (0–1 kg) without changing the motor or drag model. The model never sees the mass; it has to infer it from current, throttle and voltage.
- **Wind.** Constant and uniform for the whole flight, 0–6 m/s from any direction. The model sees only the EKF3 wind estimate, whose mean error was 0.47 m/s in validation flights.
- **Target.** Logged consumed charge at disarm minus consumed charge at RTL start. An independent integration of battery current agrees within 0.47 mAh on every retained flight.
- **Envelope.** Distance 200–1,500 m, decision altitude 30–120 m, wind 0–6 m/s, payload 0–1 kg.

## Data

| Set | Flights | Scenarios | Used for |
| --- | ---: | --- | --- |
| Train | 172 | Latin hypercube over the envelope | Fitting models |
| Validation | 39 | Same | Choosing settings and exploring inputs |
| Original test | 39 | Same | Development comparison |
| Strong-headwind training | 39 | 4–6 m/s wind within 30° of straight into the return path | Extra training data for the final models |
| Confirmation | 60 | 30 strong-headwind + 30 across the full envelope | One final, predeclared evaluation |

Every retained flight completed RTL with automatic disarm and passed eight data-quality checks. Splits are by scenario, so no scenario appears in two sets. The development data had only 17 flights with at least 4 m/s headwind, so the strong-headwind training flights were added before the final evaluation. One of 40 planned strong-headwind training flights was excluded: it touched down normally but ArduPilot's landing detector did not disarm it on either attempt, which occasionally happens near 6 m/s wind.

## Inputs

The input sets differ in how much physics is computed before the model sees the data. None of them include true wind or payload mass.

| Set | Inputs | Number of inputs |
| --- | --- | ---: |
| F0 | Route (distance, altitude, bearing), battery voltage, EKF wind estimate | 15 |
| F1 | F0 + 10-second summaries of current, throttle, motor outputs, attitude and motion | 36 |
| F2 | F1 + headwind, crosswind, nominal return time, required airspeed, current × return time | 47 |
| F3 | F1 + headwind, crosswind and a predicted cruise speed, with return time, airspeed, current × time and a drag term (time × airspeed³ / voltage) all derived from that one speed | 43 |

Any input built from travel time needs a cruise speed, and a fixed 10 m/s is wrong in strong headwind. Replacing it with a learned speed helped every input set on validation: for Ridge, F0 plus travel time went from 76.9 to 55.5 mAh and F1 plus travel time from 70.7 to 54.2. Adding a speed column on its own, without turning it into a time, made results worse. The model needed the physically meaningful combination.

## Models and architectures

- **Baselines.** Training mean; nominal return time × recent current; a PX4-style time × current estimate without wind correction.
- **Ridge.** Standardized linear regression with a coefficient penalty. An extra baseline beyond the course material.
- **XGBoost.** Gradient-boosted trees, depth 2–3, 300 trees.
- **Neural networks.** ReLU networks written in NumPy, trained with Adam and Huber loss, L2 penalty 0.001 and early stopping.

A sweep trained 15 network shapes on 10 input variants with 3 seeds each (450 networks) on the same training and validation flights. Mean validation MAE in mAh:

| Shape | F0 | F1 | F3 | Seed spread |
| --- | ---: | ---: | ---: | ---: |
| 16 | 76.8 | 80.0 | 65.2 | 17.0 |
| 64 | 71.7 | 66.9 | 54.5 | 20.4 |
| 256 | 58.6 | 54.9 | 36.5 | 3.6 |
| 1,024 | 52.6 | 56.7 | 30.3 | 14.9 |
| 64–32 | 60.6 | 62.3 | 39.5 | 8.6 |
| 128–64–32 | 56.4 | 60.9 | 38.3 | 5.5 |
| 5 × 16 | 112.4 | 134.2 | 59.7 | 35.6 |
| 10 × 16 | 211.7 | 365.2 | 87.9 | 127.2 |

Wide networks (128 units or more) mostly landed within about 10 mAh of each other; very deep narrow ones often failed to train with this recipe. The 1,024-unit single layer scored best on F3, but it was chosen from 450 fits on the same 39 validation flights, so that score is optimistic. Inputs moved results far more than architecture did. XGBoost trailed throughout (57–108 mAh): with a few hundred flights and smooth physics, trees split the inputs into steps and cannot extrapolate.

## Speed models

RTL targets 10 m/s ground speed. In strong headwind the vehicle hits its 30° lean limit and slows: median speed was 8.9 m/s at 2–4 m/s headwind and 7.0 m/s at 4–6 m/s. At the limit airspeed is nearly constant (11.8 ± 0.8 m/s in training) and rises with payload, from 11.2 to 12.7 m/s across payload thirds, because horizontal thrust at a fixed lean angle is proportional to weight. Throttle reveals the hidden weight, which gives:

```text
V_max = a + b × throttle
ground speed = min(10, sqrt(V_max² − crosswind²) − headwind)
```

For the untouched test (289 training flights), a = 4.92 m/s and b = 16.10 m/s; for the final model (349 flights), a = 5.38 m/s and b = 15.09 m/s.

| Speed model | Validation speed MAE (m/s) | Confirmation speed MAE (m/s) | Confirmation, strong headwind (m/s) |
| --- | ---: | ---: | ---: |
| Fixed 10 m/s | 0.84 | 1.76 | 2.88 |
| Random forest on decision-time inputs | 0.25 | 0.29 | 0.39 |
| Tilt-limit formula | 0.15 | 0.24 | 0.35 |

On validation, before the extra strong-headwind training flights, the forest could not predict below about 7.2 m/s, because trees cannot predict outside their training range. With those flights added, it nearly caught up. Feeding the measured speed into F3 on validation cut the worst miss from 414 to 60 mAh, so speed prediction is the main remaining lever.

## Development results

Every model was trained on the 211 train and validation flights and scored on the original 39 test flights. MAE in mAh:

| Model | F0 | F1 | F2 |
| --- | ---: | ---: | ---: |
| Ridge | 77.2 | 72.4 | 26.7 |
| Neural network (64–32) | 34.9 | 53.3 | 25.6 |
| XGBoost | 48.3 | 53.4 | 44.3 |

Physics-derived inputs cut Ridge's error by about two thirds, while the network reached 34.9 mAh from F0 alone by learning much of that physics itself. A network given the simulator's true wind scored 28.3 mAh against 25.6 with the onboard estimate, so knowing the true wind did not help in this constant-wind setup. This test set had been viewed during development and held only two flights with at least 4 m/s headwind, which is why a fresh confirmation set was run.

## Final evaluation

The candidates, seeds, epoch rule and comparisons were written down before any confirmation flight was scored (`DECISIONS.md`, D029 and D031). All candidates trained on 289 flights and were scored once on the 60 confirmation flights. Mean charge on those flights was 1,809 mAh.

| Model | MAE (mAh) | 95% interval | Strong headwind | General | Worst underprediction (mAh) |
| --- | ---: | --- | ---: | ---: | ---: |
| A: F2, Ridge | 44.8 | 35.2–55.1 | 54.0 | 35.6 | 177 |
| B: F2, 64–32 network (3-seed mean) | 38.2 | 28.5–49.0 | 44.9 | 31.4 | 188 |
| C: F3 tilt-limit, Ridge | 38.3 | 29.0–48.4 | 52.1 | 24.5 | 149 |
| D: F3 tilt-limit, 1,024 network (3-seed mean) | 38.9 | 29.7–49.5 | 51.7 | 26.1 | 127 |
| E: F3 forest speed, Ridge | 41.6 | 30.5–54.7 | 55.5 | 27.8 | 170 |
| Nominal time × current | 195.5 | 152.3–242.1 | 253.3 | 137.7 | 306 |

Planned paired comparisons (negative favours the first model): C − A = −6.5 mAh (−15.4 to +1.9); D − B = +0.7 (−5.0 to +6.5); C − E = −3.3 (−9.7 to +3.1). None excludes zero, so with 60 flights the differences between the top models are not conclusive. F3 Ridge is consistently ahead of F2 Ridge, most clearly on general flights; the F3 network did not beat the F2 network on average but cut its worst miss from 188 to 127 mAh. Single network seeds varied by up to 8 mAh, so only three-seed means are reported.

By headwind estimate, MAE in mAh:

| Headwind | Flights | F2 Ridge | F2 network | F3 Ridge | F3 network | F3 forest-speed Ridge |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tailwind | 13 | 32.6 | 15.2 | 14.3 | 13.8 | 22.0 |
| 0–2 m/s | 9 | 12.4 | 16.6 | 21.5 | 20.7 | 14.9 |
| 2–4 m/s | 7 | 38.4 | 42.5 | 41.3 | 28.2 | 38.1 |
| ≥4 m/s | 31 | 60.8 | 53.1 | 52.5 | 57.1 | 58.5 |

Below 2 m/s headwind, the learned models err by about 1–2% of the charge and none is reliably better than another. The models differ mainly in strong headwind.

## Worst cases

Strong headwind produced errors about twice those on general flights. There, F3 Ridge's charge error tracks its speed error (correlation −0.81), and the two biggest misses followed speed overestimates of 0.7–0.9 m/s. All models overpredicted slightly on average (10–14 mAh), so typical errors lean to the safe side.

The models predict the expected charge, so the real return often costs a little more: the final model under-predicted 47% of the flights in nested cross-validation. Anyone using these predictions should add a reserve on top, sized from held-out errors so that a chosen share of flights, for example 95%, uses less than the prediction plus the reserve. Because errors roughly double in strong headwind, that reserve should grow with headwind.

## Cross-validation on all the data

The untouched test fixed its network (1,024 units) in advance from the first sweep. To check that choice with more data, the sweep was repeated with five-fold cross-validation on the 289 training flights, refitting the speed model, scaling and early stopping inside each fold. Out-of-fold MAE, three-seed mean:

| Model | Parameters | CV MAE (mAh) | vs 1,024 units (95% interval) |
| --- | ---: | ---: | --- |
| 256 units | 11,521 | 20.5 | −1.4 (−3.1 to +0.3) |
| 512 units | 23,041 | 20.8 | −1.1 |
| 64–32–16 | 5,441 | 21.8 | −0.1 |
| 1,024 units | 46,081 | 21.9 | — |
| 128–64–32 | 16,001 | 22.1 | +0.3 |
| Ridge (linear) | 44 | 22.6 | +0.7 |
| 16 units | 721 | 25.9 | +4.1 |
| 10 × 16 | 3,169 | 50.8 | +29.0 |

All but the smallest and deepest shapes landed within about 2.5 mAh. Averaging three seeds beat every single seed by 2–4 mAh. Some multi-layer networks had stopped near the 600-epoch limit, so the sweep was repeated with a 3,000-epoch limit: no shape moved by more than 0.7 mAh, and the deeper networks stopped on their own at about 400–1,000 epochs. They were not undertrained; with physics-based inputs, what remains to learn is smooth enough for one wide layer. The ten-layer 16-unit network still failed, a training difficulty rather than a lack of time.

**Final model.** On all 349 flights, nested cross-validation evaluated the procedure "pick among Ridge, 256, 1,024 and 64–32–16 units by inner cross-validation, then refit". It chose 256 units in 4 of 5 outer folds and on the full data. Estimated MAE 21.6 mAh (19.1–24.2); 13.1 in tailwind, 13.3 at 0–2 m/s headwind, 24.2 at 2–4 m/s and 41.2 at 4 m/s or more; worst underprediction 130 mAh. This estimate is slightly optimistic because the candidate list was chosen after the sweep.

**Inputs vs models, all 349 flights.** MAE in mAh; every difference discussed has a 95% interval that excludes zero.

| Inputs | Ridge | XGBoost | 256-unit network |
| --- | ---: | ---: | ---: |
| F0 | 92.1 | 66.0 | 33.3 |
| F1 | 74.7 | 66.1 | 33.1 |
| F2 | 33.7 | 51.6 | 23.7 |
| F3 | 25.1 | 28.4 | 20.9 |

F3 beat F2 for every model (Ridge −8.6, XGBoost −23.2, network −2.8 mAh). The network beat XGBoost on every input set, by 7.5 mAh on F3 and up to 33 mAh on raw inputs. Physics inputs mattered most for the models least able to learn it: Ridge cannot build nonlinear combinations, and trees cannot extrapolate the slow tilt-limited returns without the speed formula.

## Limitations

- One quad in SITL's built-in physics; no other airframes, higher-fidelity simulator or real logs.
- Constant, uniform wind with no gusts or altitude change, the best case for an onboard wind estimate.
- A centred point-mass payload with its drag and inertia ignored.
- Results apply only inside the envelope above.
- With 60 confirmation flights, differences of a few mAh between models cannot be resolved.
- The models predict expected charge, not a guaranteed reserve.

## Next steps

Improve speed prediction at the tilt limit, for example by estimating the hidden mass from current as well as throttle, and size a headwind-dependent reserve from held-out errors, or train a model on a high percentile of charge.

## Reproduce

Commands, pinned versions and the location of every result are in [REPRODUCE.md](REPRODUCE.md). The decision record, including the protocol written down before the confirmation flights were scored, is in [DECISIONS.md](DECISIONS.md).
