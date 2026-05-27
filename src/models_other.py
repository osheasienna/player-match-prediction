"""Additional baseline models for match outcome prediction.

Same 3-regime setup as `models_tree.py` (A team / B +FBref / C +player), same
temporal holdout, same Understat baseline reference. Models compared here:

    - Polynomial logistic regression (degree 2, L2)
    - Ridge logistic regression (L2)
    - Lasso logistic regression (L1, saga)
    - K-Nearest Neighbours
    - Decision Tree

All wrapped in a StandardScaler pipeline where it matters, tuned with
RandomizedSearchCV under TimeSeriesSplit, and scored on log-loss / Brier /
accuracy against the held-out 2023/24 season.

Usage:
    python -m src.models_other --n-iter 30
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
TEST_SEASON = 2023
CLASS_LABELS = [0, 1, 2]

EXCLUDE_COLS = {
    "match_id", "date", "season", "match_week", "home_team", "away_team",
    "result", "result_code",
    "home_win_prob", "draw_prob", "away_win_prob",
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

REGIME_SEEDS = {"A_team": 42, "B_fbref": 1337, "C_player": 2024}


# ---------------------------------------------------------------------------
# Metrics + data
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


@dataclass
class RegimeData:
    name: str
    features: list[str]
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    baseline_proba_test: np.ndarray
    test_meta: pd.DataFrame


def load_common_subset(data_dir: Path) -> set:
    common: set | None = None
    for path in REGIME_FILES.values():
        d = pd.read_csv(data_dir / path, parse_dates=["date"])
        ids = set(d.dropna()["match_id"])
        common = ids if common is None else common & ids
    assert common is not None
    return common


def build_regime(name: str, data_dir: Path, common_ids: set) -> RegimeData:
    df = pd.read_csv(data_dir / REGIME_FILES[name], parse_dates=["date"])
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
        test_meta=test[["match_id", "date", "home_team", "away_team", "result_code"]].copy(),
    )


# ---------------------------------------------------------------------------
# Model specs
# ---------------------------------------------------------------------------

def _ts_cv(n_splits: int = 5) -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=n_splits)


def make_models() -> dict[str, tuple[Pipeline, dict]]:
    """Return {name: (pipeline, param_distribution)} for RandomizedSearchCV."""

    poly = Pipeline([
        ("scale", StandardScaler()),
        ("poly",  PolynomialFeatures(interaction_only=False, include_bias=False)),
        ("clf",   LogisticRegression(
            penalty="l2", solver="lbfgs", max_iter=2000,
            class_weight="balanced", multi_class="multinomial",
            random_state=RANDOM_STATE,
        )),
    ])
    poly_dist = {
        "poly__degree": [2],
        "clf__C": [0.01, 0.1, 1.0, 10.0],
    }

    ridge = Pipeline([
        ("scale", StandardScaler()),
        ("clf",   LogisticRegression(
            penalty="l2", solver="lbfgs", max_iter=2000,
            class_weight="balanced", multi_class="multinomial",
            random_state=RANDOM_STATE,
        )),
    ])
    ridge_dist = {"clf__C": np.logspace(-3, 2, 20).tolist()}

    lasso = Pipeline([
        ("scale", StandardScaler()),
        ("clf",   LogisticRegression(
            penalty="l1", solver="saga", max_iter=5000,
            class_weight="balanced", multi_class="multinomial",
            random_state=RANDOM_STATE,
        )),
    ])
    lasso_dist = {"clf__C": np.logspace(-3, 1, 15).tolist()}

    knn = Pipeline([
        ("scale", StandardScaler()),
        ("clf",   KNeighborsClassifier()),
    ])
    knn_dist = {
        "clf__n_neighbors": [5, 10, 15, 20, 30, 50],
        "clf__weights":     ["uniform", "distance"],
        "clf__p":           [1, 2],
    }

    dt = Pipeline([
        ("clf",   DecisionTreeClassifier(
            class_weight="balanced", random_state=RANDOM_STATE,
        )),
    ])
    dt_dist = {
        "clf__max_depth":         [3, 5, 7, 10, 15, None],
        "clf__min_samples_leaf":  [1, 2, 4, 8, 16],
        "clf__min_samples_split": [2, 5, 10, 20],
        "clf__criterion":         ["gini", "entropy"],
    }

    return {
        "PolynomialLogReg": (poly,  poly_dist),
        "Ridge":            (ridge, ridge_dist),
        "Lasso":            (lasso, lasso_dist),
        "KNN":              (knn,   knn_dist),
        "DecisionTree":     (dt,    dt_dist),
    }


# ---------------------------------------------------------------------------
# Runner
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
    proba_test: np.ndarray = field(default_factory=lambda: np.empty(0))


def tune(name: str, pipe: Pipeline, dist: dict, X: np.ndarray, y: np.ndarray,
         *, n_iter: int, seed: int) -> tuple[Pipeline, dict]:
    # Cap n_iter by the actual grid size (poly has only 4 combos).
    n_combos = int(np.prod([len(v) for v in dist.values()]))
    effective = min(n_iter, n_combos)
    search = RandomizedSearchCV(
        pipe, dist, n_iter=effective,
        scoring="neg_log_loss",
        cv=_ts_cv(5),
        random_state=seed, n_jobs=-1, refit=True, verbose=0,
    )
    search.fit(X, y)
    return search.best_estimator_, {
        "best_params": search.best_params_,
        "cv_log_loss": float(-search.best_score_),
    }


def run_regime(regime: RegimeData, *, n_iter: int) -> list[ModelResult]:
    out: list[ModelResult] = []
    baseline = score_predictions(regime.y_test, regime.baseline_proba_test)
    seed = REGIME_SEEDS.get(regime.name, RANDOM_STATE)

    for name, (pipe, dist) in make_models().items():
        best, meta = tune(name, pipe, dist, regime.X_train, regime.y_train,
                          n_iter=n_iter, seed=seed)
        proba = best.predict_proba(regime.X_test)
        out.append(ModelResult(
            regime=regime.name, model=name,
            n_features=len(regime.features),
            n_train=len(regime.y_train), n_test=len(regime.y_test),
            best_params=meta["best_params"], cv_log_loss=meta["cv_log_loss"],
            test_metrics=score_predictions(regime.y_test, proba),
            baseline_metrics=baseline,
            proba_test=proba,
        ))
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summary_table(results: list[ModelResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "regime": REGIME_LABELS[r.regime],
            "model": r.model,
            "n_features": r.n_features,
            "cv_log_loss":     round(r.cv_log_loss, 4),
            "test_log_loss":   round(r.test_metrics["log_loss"], 4),
            "test_brier":      round(r.test_metrics["brier"], 4),
            "test_acc":        round(r.test_metrics["accuracy"] * 100, 1),
            "baseline_log_loss": round(r.baseline_metrics["log_loss"], 4),
            "delta_vs_baseline": round(
                r.test_metrics["log_loss"] - r.baseline_metrics["log_loss"], 4
            ),
        })
    return pd.DataFrame(rows)


def plot_logloss_comparison(results: list[ModelResult], out_path: Path) -> None:
    df = pd.DataFrame([
        {"regime": REGIME_LABELS[r.regime], "model": r.model,
         "log_loss": r.test_metrics["log_loss"],
         "baseline": r.baseline_metrics["log_loss"]}
        for r in results
    ])
    regimes = list(dict.fromkeys(df["regime"]))
    models = ["PolynomialLogReg", "Ridge", "Lasso", "KNN", "DecisionTree"]
    x = np.arange(len(regimes))
    width = 0.15
    offsets = np.linspace(-2, 2, len(models)) * width

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, m in enumerate(models):
        vals = [df[(df.regime == r) & (df.model == m)]["log_loss"].iloc[0] for r in regimes]
        bars = ax.bar(x + offsets[i], vals, width=width, label=m)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)

    baseline_val = df["baseline"].iloc[0]
    ax.axhline(baseline_val, color="crimson", linestyle="--", linewidth=2,
               label=f"Understat baseline = {baseline_val:.3f}")
    ax.axhline(-np.log(1 / 3), color="gray", linestyle=":", alpha=0.7,
               label="Uniform = 1.099")
    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.set_ylabel("Log-loss (lower is better)")
    ax.set_title("Other models — log-loss across feature regimes (test season 2023/24)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def home_win_prob_table(results: list[ModelResult], regime: RegimeData,
                        out_path: Path) -> pd.DataFrame:
    """Per-match home-win probability: each model vs Understat, on the test set."""
    df = regime.test_meta.copy()
    df["understat_home_win_prob"] = regime.baseline_proba_test[:, 0]
    for r in results:
        if r.regime != regime.name or r.proba_test.size == 0:
            continue
        df[f"{r.model}_home_win_prob"] = r.proba_test[:, 0]
    df["actual_home_win"] = (df["result_code"] == 0).astype(int)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_all(data_dir: str | Path = "data",
            outputs_dir: str | Path = "outputs/other_models",
            *, n_iter: int = 30) -> dict[str, Any]:
    data_dir = Path(data_dir)
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    common_ids = load_common_subset(data_dir)
    print(f"Common NaN-free subset: {len(common_ids)} matches")

    all_results: list[ModelResult] = []
    regimes: dict[str, RegimeData] = {}
    for key in REGIME_FILES:
        regime = build_regime(key, data_dir, common_ids)
        regimes[key] = regime
        print(f"\n=== Regime {REGIME_LABELS[key]} "
              f"({len(regime.features)} features, "
              f"train={len(regime.y_train)}, test={len(regime.y_test)}) ===")
        results = run_regime(regime, n_iter=n_iter)
        for r in results:
            print(f"  {r.model:18s}  cv_log_loss={r.cv_log_loss:.4f}  "
                  f"test_log_loss={r.test_metrics['log_loss']:.4f}  "
                  f"acc={r.test_metrics['accuracy']*100:.1f}%  "
                  f"baseline={r.baseline_metrics['log_loss']:.4f}")
        all_results.extend(results)

    table = summary_table(all_results)
    print("\n=== Summary ===")
    print(table.to_string(index=False))

    table.to_csv(outputs_dir / "summary.csv", index=False)
    with open(outputs_dir / "results.json", "w") as f:
        # Strip proba arrays from JSON dump (kept in-memory + the CSV below).
        payload = []
        for r in all_results:
            d = asdict(r)
            d.pop("proba_test", None)
            payload.append(d)
        json.dump(payload, f, indent=2, default=str)

    plot_logloss_comparison(all_results, outputs_dir / "logloss_comparison.png")

    # Per-regime home-win probability table (model vs Understat).
    for key, regime in regimes.items():
        home_win_prob_table(
            all_results, regime,
            outputs_dir / f"home_win_prob_{key}.csv",
        )

    return {"results": all_results, "summary": table}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--outputs-dir", default="outputs/other_models")
    p.add_argument("--n-iter", type=int, default=30)
    args = p.parse_args()
    run_all(args.data_dir, args.outputs_dir, n_iter=args.n_iter)
