"""Diagnostic plots: predicted probability vs actual outcomes.

Two views per regime:
    1. Calibration curve — predicted P(home win) vs empirical home-win
       frequency, binned. Classification analog of "true vs predicted".
    2. Per-match sorted scatter — matches sorted by predicted P(home win),
       with actual outcomes overlaid. Lets you see where each model puts
       its confidence and where reality agrees.

Usage:
    python -m src.plot_predictions
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("outputs/other_models")
OTHER_MODELS = ["PolynomialLogReg", "Ridge", "Lasso", "KNN", "DecisionTree"]
TREE_MODELS  = ["RandomForest", "XGBoost", "Ensemble(RF+XGB)"]
MODELS = OTHER_MODELS + TREE_MODELS
COLORS = {
    "PolynomialLogReg": "tab:purple",
    "Ridge":            "tab:blue",
    "Lasso":            "tab:cyan",
    "KNN":              "tab:green",
    "DecisionTree":     "tab:red",
    "RandomForest":     "tab:orange",
    "XGBoost":          "tab:brown",
    "Ensemble(RF+XGB)": "darkgoldenrod",
    "Understat":        "black",
}


def calibration_curve(probs: np.ndarray, actual: np.ndarray, n_bins: int = 10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(probs, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    mean_pred, mean_obs, counts = [], [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() < 3:
            continue
        mean_pred.append(probs[mask].mean())
        mean_obs.append(actual[mask].mean())
        counts.append(mask.sum())
    return np.array(mean_pred), np.array(mean_obs), np.array(counts)


def plot_calibration(df: pd.DataFrame, regime_label: str, out_path: Path,
                     models: list[str], title_suffix: str = "",
                     include_understat: bool = True) -> None:
    actual = df["actual_home_win"].values
    fig, ax = plt.subplots(figsize=(8, 6))

    # Diagonal — perfect calibration
    ax.plot([0, 1], [0, 1], "k:", alpha=0.6, label="Perfect calibration")

    if include_understat:
        p, o, _ = calibration_curve(df["understat_home_win_prob"].values, actual)
        ax.plot(p, o, "o-", color=COLORS["Understat"], lw=2, ms=8,
                label="Understat (baseline)")

    for m in models:
        col = f"{m}_home_win_prob"
        if col not in df.columns:
            continue
        p, o, _ = calibration_curve(df[col].values, actual)
        ax.plot(p, o, "o-", color=COLORS[m], alpha=0.85, label=m)

    title = f"Calibration — predicted vs actual home-win rate ({regime_label})"
    if title_suffix:
        title += f"\n{title_suffix}"
    ax.set_xlabel("Predicted P(home win)")
    ax.set_ylabel("Observed home-win frequency")
    ax.set_title(title)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sorted(df: pd.DataFrame, model: str, regime_label: str, out_path: Path) -> None:
    col = f"{model}_home_win_prob"
    if col not in df.columns:
        return
    d = df.sort_values(col).reset_index(drop=True)
    x = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(10, 5))
    # Scatter actual outcomes at 0 (no home win) or 1 (home win)
    ax.scatter(x, d["actual_home_win"], alpha=0.35, s=25,
               color="steelblue", label="Actual outcome (0 = no home win, 1 = home win)")
    ax.plot(x, d[col], color=COLORS.get(model, "red"), lw=2,
            label=f"{model} predicted P(home win)")
    ax.plot(x, d["understat_home_win_prob"], color="black", lw=1.5,
            linestyle="--", alpha=0.7, label="Understat baseline P(home win)")

    ax.set_xlabel("Match index (sorted by model's predicted P(home win), ascending)")
    ax.set_ylabel("Probability / outcome")
    ax.set_title(f"{model} — sorted predictions vs actual outcomes ({regime_label})")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    for key, label in [
        ("A_team",   "A: team-level"),
        ("B_fbref",  "B: +FBref"),
        ("C_player", "C: +player"),
    ]:
        csv = OUT / f"home_win_prob_{key}.csv"
        if not csv.exists():
            print(f"skip {csv}")
            continue
        df = pd.read_csv(csv)
        # All models on one chart
        plot_calibration(df, label, OUT / f"calibration_{key}.png",
                         models=MODELS)
        # Trees-only chart (RF, XGBoost, Ensemble) + Understat + diagonal
        plot_calibration(df, label, OUT / f"calibration_trees_{key}.png",
                         models=TREE_MODELS,
                         title_suffix="Tree models only (RF, XGBoost, Ensemble)")
        for m in MODELS:
            plot_sorted(df, m, label, OUT / f"sorted_{key}_{m}.png")
        print(f"wrote calibration + trees-only + sorted plots for {key}")


if __name__ == "__main__":
    main()
