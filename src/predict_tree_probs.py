"""Refit the already-tuned tree models with their saved best params,
calibrate them, and append per-match home-win probabilities to the
existing other_models/home_win_prob_*.csv files.

This avoids re-running the expensive RandomizedSearch; we just re-fit
each model once at the params that already won.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models_tree import (
    REGIME_FILES, REGIME_LABELS,
    load_common_subset, build_regime, calibrate,
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


TREE_RESULTS = Path("outputs/models_tree/results.json")
OTHER_DIR    = Path("outputs/other_models")
RANDOM_STATE = 42


def best_params_by(regime: str, model: str) -> dict:
    data = json.load(open(TREE_RESULTS))
    for r in data:
        if r["regime"] == regime and r["model"] == model:
            return r["best_params"]
    raise KeyError(f"{regime}/{model} not in {TREE_RESULTS}")


def make_rf(params: dict) -> RandomForestClassifier:
    return RandomForestClassifier(
        random_state=RANDOM_STATE, n_jobs=-1,
        class_weight="balanced", **params,
    )


def make_xgb(params: dict) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", tree_method="hist",
        random_state=RANDOM_STATE, n_jobs=-1,
        **params,
    )


def _test_match_ids(key: str, common: set) -> list[str]:
    df = pd.read_csv(Path("data") / REGIME_FILES[key], parse_dates=["date"])
    df = df[df["match_id"].isin(common)].dropna().sort_values("date").reset_index(drop=True)
    return df[df["season"] == 2023]["match_id"].tolist()


def main() -> None:
    common = load_common_subset(Path("data"))
    for key in REGIME_FILES:
        regime = build_regime(key, Path("data"), common)
        test_ids = _test_match_ids(key, common)
        print(f"\n=== {REGIME_LABELS[key]} ===")

        rf = make_rf(best_params_by(key, "RandomForest"))
        rf_cal = calibrate(rf, regime.X_train, regime.y_train)
        rf_proba = rf_cal.predict_proba(regime.X_test)

        xgb = make_xgb(best_params_by(key, "XGBoost"))
        xgb_cal = calibrate(xgb, regime.X_train, regime.y_train)
        xgb_proba = xgb_cal.predict_proba(regime.X_test)

        ens_proba = 0.5 * rf_proba + 0.5 * xgb_proba

        csv_path = OTHER_DIR / f"home_win_prob_{key}.csv"
        df = pd.read_csv(csv_path)
        # Drop tree cols if they already exist (idempotent re-runs).
        for c in ("RandomForest_home_win_prob",
                  "XGBoost_home_win_prob",
                  "Ensemble(RF+XGB)_home_win_prob"):
            if c in df.columns:
                df = df.drop(columns=c)

        new = pd.DataFrame({
            "match_id": test_ids,
            "RandomForest_home_win_prob":     rf_proba[:, 0],
            "XGBoost_home_win_prob":          xgb_proba[:, 0],
            "Ensemble(RF+XGB)_home_win_prob": ens_proba[:, 0],
        })
        merged = df.merge(new, on="match_id", how="left")
        merged.to_csv(csv_path, index=False)
        print(f"  added RF/XGB/Ensemble cols  ->  {csv_path}")


if __name__ == "__main__":
    main()
