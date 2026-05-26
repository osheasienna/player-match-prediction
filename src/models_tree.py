"""Tree-based models for match outcome prediction: Random Forest + XGBoost.

Mirrors the 3-regime setup of `notebooks/02_modeling.ipynb` (team / +FBref / +player),
restricted to the common NaN-free subset and the season-2023 temporal holdout.

Adds, on top of logistic regression:
    - Randomized hyperparameter search under temporal CV (TimeSeriesSplit).
    - Probability calibration (isotonic) via CalibratedClassifierCV on a temporal
      inner split — applied after the search picks the best estimator.
    - Per-regime feature-importance ranking.
    - Side-by-side comparison vs the Understat baseline.

Usage:
    from src.models_tree import run_all
    results = run_all(data_dir="data", outputs_dir="outputs/models_tree")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight

from xgboost import XGBClassifier


RANDOM_STATE = 42
TEST_SEASON = 2023
CLASS_LABELS = [0, 1, 2]  # H, D, A — matches result_code in the upstream CSVs

EXCLUDE_COLS = {
    "match_id", "date", "season", "match_week", "home_team", "away_team",
    "result", "result_code",
    "home_win_prob", "draw_prob", "away_win_prob",  # baseline — never a feature
}

REGIME_FILES = {
    "A_team":   "ligue1_features.csv",
    "B_fbref":  "ligue1_fbref_features.csv",
    "C_player": "ligue1_player_features.csv",
}
REGIME_LABELS = {
    "A_team":   "A: team-level",
    "B_fbref":  "B: +FBref",
    "C_player": "C: +player",
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def brier_multiclass(y_true: np.ndarray, proba: np.ndarray, n_classes: int = 3) -> float:
    y_one_hot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((proba - y_one_hot) ** 2, axis=1)))


def score_predictions(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    return {
        "log_loss": float(log_loss(y_true, proba, labels=CLASS_LABELS)),
        "brier":    brier_multiclass(y_true, proba),
        "accuracy": float(accuracy_score(y_true, proba.argmax(axis=1))),
    }


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class RegimeData:
    name: str
    features: list[str]
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    baseline_proba_test: np.ndarray
    train_seasons: tuple[int, ...]


def load_common_subset(data_dir: str | Path) -> set:
    """Match IDs that are NaN-free in every regime CSV (apples-to-apples comparison)."""
    data_dir = Path(data_dir)
    common: set | None = None
    for path in REGIME_FILES.values():
        d = pd.read_csv(data_dir / path, parse_dates=["date"])
        ids = set(d.dropna()["match_id"])
        common = ids if common is None else common & ids
    assert common is not None
    return common


def build_regime(name: str, data_dir: str | Path, common_ids: set) -> RegimeData:
    df = pd.read_csv(Path(data_dir) / REGIME_FILES[name], parse_dates=["date"])
    df = df[df["match_id"].isin(common_ids)].dropna().sort_values("date").reset_index(drop=True)

    features = [c for c in df.columns if c not in EXCLUDE_COLS]

    train = df[df["season"] < TEST_SEASON].reset_index(drop=True)
    test  = df[df["season"] == TEST_SEASON].reset_index(drop=True)

    assert train["date"].max() < test["date"].min(), "temporal leakage"

    return RegimeData(
        name=name,
        features=features,
        X_train=train[features].values,
        y_train=train["result_code"].values,
        X_test=test[features].values,
        y_test=test["result_code"].values,
        baseline_proba_test=test[["home_win_prob", "draw_prob", "away_win_prob"]].values,
        train_seasons=tuple(sorted(train["season"].unique().tolist())),
    )


# ---------------------------------------------------------------------------
# Model search + calibration
# ---------------------------------------------------------------------------

# Hyperparameter spaces — kept small enough to run in seconds on ~400-row training sets.
RF_PARAM_DIST = {
    "n_estimators":      [200, 400, 600, 800],
    "max_depth":         [3, 5, 7, 10, None],
    "min_samples_leaf":  [1, 2, 4, 8],
    "min_samples_split": [2, 5, 10],
    "max_features":      ["sqrt", "log2", 0.5],
}

XGB_PARAM_DIST = {
    "n_estimators":     [200, 400, 600, 800],
    "max_depth":        [3, 4, 5, 6, 8],
    "learning_rate":    [0.01, 0.03, 0.05, 0.1],
    "subsample":        [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "reg_lambda":       [0.5, 1.0, 2.0, 5.0],
}


def _class_weight_dict(y: np.ndarray) -> dict[int, float]:
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes.tolist(), weights.tolist()))


def _xgb_sample_weights(y: np.ndarray) -> np.ndarray:
    cw = _class_weight_dict(y)
    return np.array([cw[int(c)] for c in y])


def tune_random_forest(X: np.ndarray, y: np.ndarray, *, n_iter: int = 30,
                        n_splits: int = 5,
                        search_seed: int = RANDOM_STATE) -> tuple[RandomForestClassifier, dict]:
    base = RandomForestClassifier(
        random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced",
    )
    search = RandomizedSearchCV(
        base, RF_PARAM_DIST, n_iter=n_iter,
        scoring="neg_log_loss",
        cv=TimeSeriesSplit(n_splits=n_splits),
        random_state=search_seed, n_jobs=-1, refit=True, verbose=0,
    )
    search.fit(X, y)
    return search.best_estimator_, {"best_params": search.best_params_,
                                     "cv_log_loss": float(-search.best_score_)}


def tune_xgboost(X: np.ndarray, y: np.ndarray, *, n_iter: int = 30,
                  n_splits: int = 5,
                  search_seed: int = RANDOM_STATE) -> tuple[XGBClassifier, dict]:
    # XGBoost uses sample_weight for imbalance (no class_weight kwarg).
    sw = _xgb_sample_weights(y)
    base = XGBClassifier(
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", tree_method="hist",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    search = RandomizedSearchCV(
        base, XGB_PARAM_DIST, n_iter=n_iter,
        scoring="neg_log_loss",
        cv=TimeSeriesSplit(n_splits=n_splits),
        random_state=search_seed, n_jobs=-1, refit=True, verbose=0,
    )
    search.fit(X, y, sample_weight=sw)
    return search.best_estimator_, {"best_params": search.best_params_,
                                     "cv_log_loss": float(-search.best_score_)}


def calibrate(estimator, X: np.ndarray, y: np.ndarray, *, method: str = "isotonic",
              n_splits: int = 3):
    """Wrap a fitted-style estimator in a temporally-cross-validated calibrator.

    CalibratedClassifierCV refits the base estimator on each CV fold internally,
    so we pass an unfitted clone-equivalent (the tuned estimator with its
    best_params_); fit happens here.
    """
    calibrated = CalibratedClassifierCV(
        estimator=estimator, method=method, cv=TimeSeriesSplit(n_splits=n_splits),
    )
    # XGB needs sample_weight to honor class imbalance during refits.
    if isinstance(estimator, XGBClassifier):
        calibrated.fit(X, y, sample_weight=_xgb_sample_weights(y))
    else:
        calibrated.fit(X, y)
    return calibrated


# ---------------------------------------------------------------------------
# Per-regime runner
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    regime: str
    model: str
    n_features: int
    n_train: int
    n_test: int
    best_params: dict
    cv_log_loss: float
    test_metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    feature_importances: dict[str, float] = field(default_factory=dict)


def _feature_importances(estimator, feature_names: list[str], top_k: int = 20) -> dict[str, float]:
    """Pull importances from a calibrated estimator by averaging the base estimators.

    CalibratedClassifierCV stores per-fold base estimators in `calibrated_classifiers_`,
    each with an `.estimator` attribute (sklearn >= 1.2). Averaging gives a stable rank.
    """
    bases = []
    for cc in getattr(estimator, "calibrated_classifiers_", []):
        inner = getattr(cc, "estimator", None)
        if inner is not None and hasattr(inner, "feature_importances_"):
            bases.append(inner.feature_importances_)
    if not bases:
        return {}
    avg = np.mean(np.stack(bases, axis=0), axis=0)
    order = np.argsort(avg)[::-1][:top_k]
    return {feature_names[i]: float(avg[i]) for i in order}


# Fixed per-regime seeds: deterministic search per regime, but the three regimes
# explore different corners of the param distribution so results aren't artificially
# coupled by a single shared RNG.
REGIME_SEEDS = {"A_team": 42, "B_fbref": 1337, "C_player": 2024}


def run_regime(regime: RegimeData, *, n_iter: int = 30) -> list[ModelResult]:
    out: list[ModelResult] = []
    baseline = score_predictions(regime.y_test, regime.baseline_proba_test)
    seed = REGIME_SEEDS.get(regime.name, RANDOM_STATE)

    # --- Random Forest -----------------------------------------------------
    rf_best, rf_meta = tune_random_forest(
        regime.X_train, regime.y_train, n_iter=n_iter, search_seed=seed,
    )
    rf_cal = calibrate(rf_best, regime.X_train, regime.y_train)
    rf_proba = rf_cal.predict_proba(regime.X_test)
    out.append(ModelResult(
        regime=regime.name, model="RandomForest",
        n_features=len(regime.features),
        n_train=len(regime.y_train), n_test=len(regime.y_test),
        best_params=rf_meta["best_params"], cv_log_loss=rf_meta["cv_log_loss"],
        test_metrics=score_predictions(regime.y_test, rf_proba),
        baseline_metrics=baseline,
        feature_importances=_feature_importances(rf_cal, regime.features),
    ))

    # --- XGBoost -----------------------------------------------------------
    xgb_best, xgb_meta = tune_xgboost(
        regime.X_train, regime.y_train, n_iter=n_iter, search_seed=seed,
    )
    xgb_cal = calibrate(xgb_best, regime.X_train, regime.y_train)
    xgb_proba = xgb_cal.predict_proba(regime.X_test)
    out.append(ModelResult(
        regime=regime.name, model="XGBoost",
        n_features=len(regime.features),
        n_train=len(regime.y_train), n_test=len(regime.y_test),
        best_params=xgb_meta["best_params"], cv_log_loss=xgb_meta["cv_log_loss"],
        test_metrics=score_predictions(regime.y_test, xgb_proba),
        baseline_metrics=baseline,
        feature_importances=_feature_importances(xgb_cal, regime.features),
    ))

    # --- Ensemble: simple average of RF + XGB calibrated probabilities -----
    ens_proba = 0.5 * rf_proba + 0.5 * xgb_proba
    # CV log-loss for the ensemble = average of the two component CV scores; not
    # a re-fit, just a label so the column is populated.
    out.append(ModelResult(
        regime=regime.name, model="Ensemble(RF+XGB)",
        n_features=len(regime.features),
        n_train=len(regime.y_train), n_test=len(regime.y_test),
        best_params={"weights": {"RF": 0.5, "XGB": 0.5}},
        cv_log_loss=float(np.mean([rf_meta["cv_log_loss"], xgb_meta["cv_log_loss"]])),
        test_metrics=score_predictions(regime.y_test, ens_proba),
        baseline_metrics=baseline,
        feature_importances={},
    ))

    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_logloss_comparison(results: list[ModelResult], out_path: Path) -> None:
    df = pd.DataFrame([
        {"regime": REGIME_LABELS[r.regime], "model": r.model,
         "log_loss": r.test_metrics["log_loss"],
         "baseline": r.baseline_metrics["log_loss"]}
        for r in results
    ])
    regimes = list(dict.fromkeys(df["regime"]))
    models = ["RandomForest", "XGBoost", "Ensemble(RF+XGB)"]
    x = np.arange(len(regimes))
    width = 0.27

    fig, ax = plt.subplots(figsize=(9, 5))
    offsets = [-width, 0.0, width]
    for i, m in enumerate(models):
        vals = [df[(df.regime == r) & (df.model == m)]["log_loss"].iloc[0] for r in regimes]
        bars = ax.bar(x + offsets[i], vals, width=width, label=m)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9)

    baseline_val = df["baseline"].iloc[0]
    ax.axhline(baseline_val, color="crimson", linestyle="--", linewidth=2,
               label=f"Understat baseline = {baseline_val:.3f}")
    ax.axhline(-np.log(1 / 3), color="gray", linestyle=":", alpha=0.7,
               label="Uniform = 1.099")
    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.set_ylabel("Log-loss (lower is better)")
    ax.set_title("Tree models — log-loss across feature regimes (test season 2023/24)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_feature_importances(result: ModelResult, out_path: Path, top_k: int = 15) -> None:
    if not result.feature_importances:
        return
    items = list(result.feature_importances.items())[:top_k]
    names, vals = zip(*items)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(names))))
    ax.barh(range(len(names)), vals, color="steelblue")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Mean importance (averaged across calibration folds)")
    ax.set_title(f"{result.model} — top {len(names)} features ({REGIME_LABELS[result.regime]})")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def summary_table(results: list[ModelResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "regime": REGIME_LABELS[r.regime],
            "model": r.model,
            "n_features": r.n_features,
            "n_train": r.n_train,
            "n_test": r.n_test,
            "cv_log_loss": round(r.cv_log_loss, 4),
            "test_log_loss": round(r.test_metrics["log_loss"], 4),
            "test_brier":   round(r.test_metrics["brier"], 4),
            "test_acc":     round(r.test_metrics["accuracy"] * 100, 1),
            "baseline_log_loss": round(r.baseline_metrics["log_loss"], 4),
            "delta_vs_baseline": round(
                r.test_metrics["log_loss"] - r.baseline_metrics["log_loss"], 4
            ),
        })
    return pd.DataFrame(rows)


def run_all(data_dir: str | Path = "data",
            outputs_dir: str | Path = "outputs/models_tree",
            *, n_iter: int = 30, save_plots: bool = True) -> dict[str, Any]:
    """End-to-end: build regimes, tune+calibrate RF & XGB, save plots + JSON."""
    data_dir = Path(data_dir)
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    common_ids = load_common_subset(data_dir)
    print(f"Common NaN-free subset: {len(common_ids)} matches")

    all_results: list[ModelResult] = []
    for key in REGIME_FILES:
        regime = build_regime(key, data_dir, common_ids)
        print(f"\n=== Regime {REGIME_LABELS[key]} "
              f"({len(regime.features)} features, "
              f"train={len(regime.y_train)}, test={len(regime.y_test)}) ===")
        results = run_regime(regime, n_iter=n_iter)
        for r in results:
            print(f"  {r.model:13s}  cv_log_loss={r.cv_log_loss:.4f}  "
                  f"test_log_loss={r.test_metrics['log_loss']:.4f}  "
                  f"baseline={r.baseline_metrics['log_loss']:.4f}")
        all_results.extend(results)

    table = summary_table(all_results)
    print("\n=== Summary ===")
    print(table.to_string(index=False))

    table.to_csv(outputs_dir / "summary.csv", index=False)
    with open(outputs_dir / "results.json", "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2, default=str)

    if save_plots:
        plot_logloss_comparison(all_results, outputs_dir / "logloss_comparison.png")
        for r in all_results:
            tag = f"{r.regime}_{r.model}".lower()
            plot_feature_importances(r, outputs_dir / f"importance_{tag}.png")

    return {"results": all_results, "summary": table}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--outputs-dir", default="outputs/models_tree")
    p.add_argument("--n-iter", type=int, default=30)
    args = p.parse_args()
    run_all(args.data_dir, args.outputs_dir, n_iter=args.n_iter)
