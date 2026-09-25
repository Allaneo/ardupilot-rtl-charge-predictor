# ArduPilot RTL charge predictor

At the moment an ArduPilot quadcopter is commanded to Return to Launch (RTL), how much battery charge will the trip home use? This project predicts it from what the vehicle knows at that moment, using 349 flights of the real ArduPilot flight code in SITL (software in the loop).

Built as the applied project for *Advanced Learning Algorithms*, the second course of the Machine Learning Specialization: NumPy neural networks, gradient-boosted trees (XGBoost) and Ridge regression as a linear baseline.

## Main results

- **The lean limit drives the hard cases.** RTL holds 10 m/s over the ground until strong headwind pushes the vehicle to its 30° lean limit; from there it slows, and heavier vehicles hold more speed into the wind. A two-parameter formula captures it:

  ```text
  V_max = a + b × throttle
  ground speed = min(10, sqrt(V_max² − crosswind²) − headwind)
  ```

- **Untouched test, 60 new flights scored once** (half in strong headwind, protocol written down in advance in [DECISIONS.md](DECISIONS.md), D029 and D031): the leading models missed by 38 to 39 mAh on average, about 2% of the charge, against 196 mAh for a return-time × current rule.
- **Final model:** a 256-unit single-layer network on the tilt-limit inputs, three-seed average. Nested cross-validation on all 349 flights: 21.6 mAh mean absolute error (95% interval 19.1 to 24.2), about 1.2% of the charge; 41 mAh in 4 to 6 m/s headwind.
- **Inputs mattered more than architecture.** Networks from about 5,000 to 46,000 parameters landed within about 2.5 mAh of each other; physics-based inputs cut Ridge and XGBoost errors by two thirds or more.

The full write-up, with every comparison, is in [COURSE_REPORT.md](COURSE_REPORT.md).

## Setup

- ArduPilot ArduCopter SITL at a pinned commit ([config/ardupilot.lock.yaml](config/ardupilot.lock.yaml)), built-in quad frame, EKF3 with drag-based wind estimation.
- A small simulator patch ([patches/](patches/)) adds a centred payload of 0 to 1 kg without changing the airframe's propulsion calibration. The model never sees the payload or the true wind.
- Each flight flies out, hovers for a 10 second decision window, then returns. The target is the logged charge from RTL start to automatic disarm, cross-checked against integrated current.
- Envelope: 200 to 1,500 m from home, 30 to 120 m altitude, constant wind up to 6 m/s, payload up to 1 kg.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/rtl_charge/` | Flight control over MAVLink, log processing, leakage guards, models (NumPy network, Ridge, XGBoost) |
| `scripts/` | Scenario generation, flight runner, dataset builder, and one script per reported result |
| `config/` | Pinned versions, SITL parameters, input-set definitions |
| `data/` | Scenario manifests, quality reports and the processed datasets (raw logs are not included) |
| `reports/` | Saved outputs behind every number in the report |
| `tests/` | Unit tests (`python -m pytest -q`) |

[REPRODUCE.md](REPRODUCE.md) lists the commands, and which script produces which result.

## Limitations

Everything here is simulation of one quadcopter in SITL's built-in physics, with constant, uniform wind and a point-mass payload; nothing has been validated on real flights. The models predict expected charge, not a guaranteed reserve.
