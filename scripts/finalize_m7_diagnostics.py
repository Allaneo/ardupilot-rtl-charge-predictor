#!/usr/bin/env python3
"""Create descriptive M7 strata and figures from the already-scored flights."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/m7"
DATA = ROOT / "data/processed/pipeline_dataset.csv"


def main() -> None:
    predictions = pd.read_csv(REPORT / "test_predictions.csv")
    run_ids = predictions.loc[predictions["model"] == "ridge__f2_physics_informed", "run_id"].tolist()
    columns = ["split", "run_id", "true_wind_north_m_s", "true_wind_east_m_s",
               "ekf_wind_north_mean_m_s", "ekf_wind_east_mean_m_s", "route_bearing_cos",
               "route_bearing_sin", "payload_mass_kg", "distance_home_m", "relative_altitude_m"]
    frame = pd.read_csv(DATA, usecols=columns)
    train = frame.loc[frame["split"] == "train"].set_index("run_id")
    test = frame.loc[frame["run_id"].isin(run_ids)].set_index("run_id").loc[run_ids]
    if len(test) != 39:
        raise ValueError("saved M7 prediction rows do not match the frozen test split")

    n = test["route_bearing_cos"].to_numpy(float)
    e = test["route_bearing_sin"].to_numpy(float)
    along = test["true_wind_north_m_s"].to_numpy(float) * n + test["true_wind_east_m_s"].to_numpy(float) * e
    wind_error = np.hypot(test["true_wind_north_m_s"] - test["ekf_wind_north_mean_m_s"],
                          test["true_wind_east_m_s"] - test["ekf_wind_east_mean_m_s"])

    definitions = {
        "true_wind_speed": (np.hypot(test["true_wind_north_m_s"], test["true_wind_east_m_s"]),
                            np.hypot(train["true_wind_north_m_s"], train["true_wind_east_m_s"]), ["low", "mid", "high"]),
        "wind_route_regime": (pd.Series(np.where(along < -0.25, "headwind", np.where(along > 0.25, "tailwind", "crosswind")), index=test.index), None, None),
        "payload_mass": (test["payload_mass_kg"], train["payload_mass_kg"], ["low", "mid", "high"]),
        "distance_home": (test["distance_home_m"], train["distance_home_m"], ["near", "mid", "far"]),
        "altitude": (test["relative_altitude_m"], train["relative_altitude_m"], ["low", "mid", "high"]),
        "ekf_wind_error": (pd.Series(wind_error, index=test.index),
                            np.hypot(train["true_wind_north_m_s"] - train["ekf_wind_north_mean_m_s"],
                                     train["true_wind_east_m_s"] - train["ekf_wind_east_mean_m_s"]),
                            ["low", "mid", "high"]),
    }
    rows, cuts_report = [], {}
    for variable, (values, train_values, labels) in definitions.items():
        if train_values is None:
            categories = values
            cuts_report[variable] = "true along-route wind: below -0.25 m/s headwind, above +0.25 m/s tailwind, otherwise crosswind"
        else:
            edges = np.unique(np.quantile(np.asarray(train_values, dtype=float), [0, 1/3, 2/3, 1]))
            cuts_report[variable] = [float(x) for x in edges[1:-1]]
            if len(edges) < 3:
                categories = pd.Series("all", index=test.index)
            else:
                edges[0], edges[-1] = -np.inf, np.inf
                categories = pd.cut(values, bins=edges, labels=labels[:len(edges)-1], include_lowest=True)
        for model, group in predictions.groupby("model", sort=False):
            indexed = group.set_index("run_id")
            for category in pd.unique(np.asarray(categories.astype(str))):
                ids = test.index[np.asarray(categories.astype(str)) == category]
                part = indexed.loc[indexed.index.intersection(ids)]
                if part.empty:
                    continue
                rows.append({"variable": variable, "bin": category, "model": model,
                             "n_flights": len(part), "small_sample_warning": len(part) < 5,
                             "mae_mah": float(part["absolute_error_mah"].mean()),
                             "mean_signed_error_mah": float(part["error_mah"].mean()),
                             "mean_underprediction_mah": float(part["underprediction_mah"].mean()),
                             "p95_underprediction_mah": float(part["underprediction_mah"].quantile(.95))})
    breakdown = pd.DataFrame(rows)
    breakdown.to_csv(REPORT / "error_breakdowns.csv", index=False)
    summary_path = REPORT / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["subgroup_thresholds"] = cuts_report
    summary["subgroup_caution"] = "39 test flights; all bins include n; bins below five are flagged and descriptive only"
    summary["artifacts"] = list(dict.fromkeys([*[
        item for item in summary["artifacts"] if item != "f2_ablation.png"],
        "test_mae_comparison.png", "actual_vs_predicted.png",
        "residual_distributions.png", "f2_model_comparison.png", "feature_set_ablation.png",
        "error_breakdowns.png", "underprediction_distribution.png",
        "wind_oracle_gap.png"]))
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    metrics = pd.read_csv(REPORT / "test_metrics.csv")
    order = metrics.sort_values("mae_mah")
    labels = order["model"].str.replace("__", " · ", regex=False).str.replace("_", " ", regex=False)
    fig, ax = plt.subplots(figsize=(10, 7))
    mid = order["mae_bootstrap_95"].map(lambda s: json.loads(s.replace("'", '"')) if isinstance(s, str) else s)
    low = np.array([row["ci95_low"] for row in mid]); high = np.array([row["ci95_high"] for row in mid])
    ax.barh(labels, order["mae_mah"], color="#2A6F97")
    ax.errorbar(order["mae_mah"], np.arange(len(order)), xerr=[order["mae_mah"]-low, high-order["mae_mah"]], fmt="none", ecolor="#222", capsize=2)
    ax.invert_yaxis(); ax.set_xlabel("Test MAE (mAh), 95% flight-bootstrap interval"); ax.grid(axis="x", alpha=.2)
    fig.tight_layout(); fig.savefig(REPORT / "test_mae_comparison.png", dpi=170); plt.close(fig)

    main_names = ["ridge__f2_physics_informed", "neural_network__f2_physics_informed", "xgboost__f2_physics_informed"]
    fig, ax = plt.subplots(figsize=(6, 6))
    for name in main_names:
        part = predictions[predictions.model == name]
        ax.scatter(part.actual_rtl_charge_mah, part.predicted_rtl_charge_mah, label=name.replace("__", " · "), alpha=.8)
    lo = min(predictions.actual_rtl_charge_mah.min(), predictions.predicted_rtl_charge_mah.min())
    hi = max(predictions.actual_rtl_charge_mah.max(), predictions.predicted_rtl_charge_mah.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1); ax.set(xlabel="Actual RTL charge (mAh)", ylabel="Predicted RTL charge (mAh)")
    ax.legend(fontsize=8); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(REPORT / "actual_vs_predicted.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    data = [predictions.loc[predictions.model == name, "error_mah"].to_numpy() for name in main_names]
    ax.boxplot(data, tick_labels=[x.replace("__", " · ").replace("_", " ") for x in main_names], showfliers=True)
    ax.axhline(0, color="black", linewidth=1); ax.set_ylabel("Prediction − actual (mAh)"); ax.tick_params(axis="x", rotation=12)
    ax.grid(axis="y", alpha=.2); fig.tight_layout(); fig.savefig(REPORT / "residual_distributions.png", dpi=170); plt.close(fig)

    f2_names = ["ridge__f2_physics_informed", "neural_network__f2_physics_informed", "xgboost__f2_physics_informed"]
    f2_labels = ["F2 Ridge", "F2 Neural network", "F2 XGBoost"]
    fig, ax = plt.subplots(figsize=(6, 4))
    values = [float(metrics.loc[metrics.model == name, "mae_mah"].iloc[0]) for name in f2_names]
    ax.bar(f2_labels, values, color=["#2A6F97", "#4C956C", "#F4A261"])
    ax.set(ylabel="Test MAE (mAh)", title="Models using the F2 feature set")
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(REPORT / "f2_model_comparison.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model_name, label, marker in (("ridge", "Ridge", "o"),
                                      ("neural_network", "Neural network", "s"),
                                      ("xgboost", "XGBoost", "^")):
        names = [f"{model_name}__{feature_set}" for feature_set in
                 ("f0_route_and_wind", "f1_vehicle_symptoms", "f2_physics_informed")]
        model_values = metrics.set_index("model").loc[names, "mae_mah"].to_numpy()
        ax.plot(["F0", "F1", "F2"], model_values, marker=marker, label=label)
    ax.set(xlabel="Feature set", ylabel="Test MAE (mAh)",
           title="Feature ablation on the same 39 test flights")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(REPORT / "feature_set_ablation.png", dpi=170); plt.close(fig)

    selected = predictions[predictions.model.isin(main_names)]
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, part in selected.groupby("model", sort=False):
        ax.hist(part.underprediction_mah, bins=12, alpha=.45, label=name.replace("__", " · "))
    ax.set(xlabel="Underprediction (mAh)", ylabel="Flight count"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(REPORT / "underprediction_distribution.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    oracle_names = ["neural_network__f2_physics_informed", "neural_network_true_wind_oracle", "xgboost__f2_physics_informed", "xgboost_true_wind_oracle"]
    oracle_metrics = metrics.set_index("model").loc[oracle_names]
    ax.bar(["NN EKF", "NN true-wind", "XGB EKF", "XGB true-wind"], oracle_metrics.mae_mah, color="#9B5DE5")
    ax.set_ylabel("Test MAE (mAh)"); ax.grid(axis="y", alpha=.2); fig.tight_layout(); fig.savefig(REPORT / "wind_oracle_gap.png", dpi=170); plt.close(fig)

    # Show all subgroup models in a multi-panel summary, retain tabular detail as source of truth.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, variable in zip(axes, ("true_wind_speed", "payload_mass", "distance_home")):
        part = breakdown[(breakdown.variable == variable) & (breakdown.model.isin(main_names))]
        for name, group in part.groupby("model", sort=False):
            ax.plot(group["bin"], group.mae_mah, marker="o", label=name.replace("__", " · "))
        ax.set_title(variable.replace("_", " ")); ax.set_ylabel("MAE (mAh)"); ax.tick_params(axis="x", rotation=20); ax.grid(alpha=.2)
    axes[-1].legend(fontsize=7); fig.tight_layout(); fig.savefig(REPORT / "error_breakdowns.png", dpi=170); plt.close(fig)
    print(f"M7 diagnostic tables/figures finalized from saved predictions; scored rows unchanged ({len(predictions)})")


if __name__ == "__main__":
    main()
