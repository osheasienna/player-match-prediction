"""Feature selection on the combined Understat + FBref feature set.

Applies the same three-method protocol as feature_selection.py but on the
extended feature matrix that includes the new FBref signals:
  - Possession (rolling 5-match per team)
  - Formation (number of attackers)
  - Player season G+A/90 (aggregated to team level)
  - Squad depth (starter vs bench quality gap)

Outputs saved to outputs/fbref/:
  fbref_filter_mi_scores.png
  fbref_filter_correlation_heatmap.png
  fbref_wrapper_rfecv.png
  fbref_embedded_lasso.png
  fbref_embedded_rf_importance.png
  fbref_pca_scree.png
  fbref_feature_leaderboard.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).parent.parent / "data" / "ligue1_fbref_features.csv"
OUT_DIR = Path(__file__).parent.parent / "outputs" / "fbref"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Understat features already tested in feature_selection.py
UNDERSTAT_FEATURES = [
    "home_roll_xg_for", "home_roll_xg_against", "home_roll_xg_diff",
    "away_roll_xg_for", "away_roll_xg_against", "away_roll_xg_diff",
    "net_xg_diff",
    "home_roll_goals_for", "home_roll_goals_against", "home_roll_goal_diff",
    "away_roll_goals_for", "away_roll_goals_against", "away_roll_goal_diff",
    "net_goal_diff",
    "home_roll_won", "away_roll_won", "net_win_rate",
    "home_roll_points", "away_roll_points", "net_points",
    "home_squad_value", "away_squad_value",
    "squad_value_ratio", "log_squad_value_ratio", "squad_value_diff",
    "home_avg_player_value", "away_avg_player_value",
    "home_squad_age", "away_squad_age",
    "home_win_prob", "draw_prob", "away_win_prob",
]

# New FBref-only features
FBREF_FEATURES = [
    "home_roll_poss", "away_roll_poss", "net_poss",
    "home_attackers", "away_attackers", "net_attackers",
    "home_player_ga90", "away_player_ga90", "net_player_ga90",
    "home_starter_ga90", "away_starter_ga90", "net_starter_ga90",
    "home_squad_depth", "away_squad_depth", "net_squad_depth",
    "home_minutes_conc", "away_minutes_conc",
]

ALL_FEATURES = UNDERSTAT_FEATURES + FBREF_FEATURES


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    available = [c for c in ALL_FEATURES if c in df.columns]
    X = df[available].fillna(df[available].median())
    y = df["result_code"]
    return X, y, available


# ---------------------------------------------------------------------------
# Method 1: Filter
# ---------------------------------------------------------------------------

def filter_method(X: pd.DataFrame, y: pd.Series, features: list[str]) -> pd.Series:
    print("\n[1] Filter: Mutual Information ...")
    mi = mutual_info_classif(X[features], y, random_state=42, n_neighbors=5)
    mi_series = pd.Series(mi, index=features).sort_values(ascending=False)

    # Colour FBref features differently
    colors = ["#e74c3c" if f in FBREF_FEATURES else "#2c7bb6" for f in mi_series.index]
    fig, ax = plt.subplots(figsize=(11, 9))
    mi_series.plot.barh(ax=ax, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Mutual Information Score")
    ax.set_title("Filter Method — MI Score (red = FBref-only features)")
    ax.axvline(mi_series.mean(), color="grey", linestyle="--", label="Mean MI")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#e74c3c", label="FBref feature"),
        Patch(color="#2c7bb6", label="Understat feature"),
    ])
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fbref_filter_mi_scores.png", dpi=150)
    plt.close(fig)
    print(f"  Top-5 FBref features by MI: {[f for f in mi_series.index if f in FBREF_FEATURES][:5]}")
    print(f"  Saved fbref_filter_mi_scores.png")

    # Correlation heatmap — FBref features vs top Understat features
    top_understat = [f for f in mi_series.index if f in UNDERSTAT_FEATURES][:12]
    fbref_present = [f for f in FBREF_FEATURES if f in features]
    heatmap_cols = top_understat + fbref_present
    corr = X[heatmap_cols].corr()

    fig, ax = plt.subplots(figsize=(16, 13))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(heatmap_cols)))
    ax.set_yticks(range(len(heatmap_cols)))
    ax.set_xticklabels(heatmap_cols, rotation=90, fontsize=7)
    ax.set_yticklabels(heatmap_cols, fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # Draw dividing line between Understat and FBref blocks
    div = len(top_understat) - 0.5
    ax.axhline(div, color="black", linewidth=1.5)
    ax.axvline(div, color="black", linewidth=1.5)
    ax.set_title("Correlation Heatmap — top Understat features + all FBref features\n(dividing line separates the two groups)")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fbref_filter_correlation_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved fbref_filter_correlation_heatmap.png")

    return mi_series


# ---------------------------------------------------------------------------
# Method 2: Wrapper — RFECV
# ---------------------------------------------------------------------------

def wrapper_method(X: pd.DataFrame, y: pd.Series, features: list[str]) -> pd.Series:
    print("\n[2] Wrapper: RFECV ...")
    tscv = TimeSeriesSplit(n_splits=5)
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rfecv = RFECV(estimator=rf, step=1, cv=tscv, scoring="neg_log_loss",
                  min_features_to_select=5, n_jobs=-1)
    rfecv.fit(X[features].values, y.values)

    selected_mask = rfecv.support_
    ranking = rfecv.ranking_
    rfecv_score = pd.Series(
        [1.0 if s else 1.0 / r for s, r in zip(selected_mask, ranking)],
        index=features,
    ).sort_values(ascending=False)

    cv_results = rfecv.cv_results_
    mean_scores = -cv_results["mean_test_score"]
    std_scores = cv_results["std_test_score"]
    n_range = range(1, len(mean_scores) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(n_range, mean_scores, yerr=std_scores, fmt="o-",
                color="#d7191c", capsize=3, linewidth=1.5, markersize=4)
    ax.axvline(rfecv.n_features_, color="green", linestyle="--",
               label=f"Optimal: {rfecv.n_features_} features")
    ax.set_xlabel("Number of features")
    ax.set_ylabel("Log-loss (lower = better)")
    ax.set_title("Wrapper Method — RFECV: CV log-loss vs. feature count (with FBref)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fbref_wrapper_rfecv.png", dpi=150)
    plt.close(fig)

    selected = [f for f, s in zip(features, selected_mask) if s]
    fbref_selected = [f for f in selected if f in FBREF_FEATURES]
    print(f"  Optimal features: {rfecv.n_features_}")
    print(f"  FBref features selected: {fbref_selected}")
    print(f"  Saved fbref_wrapper_rfecv.png")
    return rfecv_score


# ---------------------------------------------------------------------------
# Method 3: Embedded
# ---------------------------------------------------------------------------

def embedded_method(
    X: pd.DataFrame, y: pd.Series, features: list[str]
) -> tuple[pd.Series, pd.Series]:
    print("\n[3a] Embedded: L1 Logistic Regression ...")
    lasso = LogisticRegression(solver="saga", l1_ratio=1.0, C=0.5,
                               max_iter=2000, random_state=42)
    pipe_X = StandardScaler().fit_transform(X[features])
    lasso.fit(pipe_X, y)
    coef = np.abs(lasso.coef_).mean(axis=0)
    lasso_imp = pd.Series(coef, index=features).sort_values(ascending=False)

    colors = ["#e74c3c" if f in FBREF_FEATURES else "#1a9641" for f in lasso_imp.index]
    fig, ax = plt.subplots(figsize=(11, 9))
    lasso_imp.plot.barh(ax=ax, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |L1 coefficient| across classes")
    ax.set_title("Embedded — L1 Logistic Regression (red = FBref features)")
    ax.axvline(lasso_imp.mean(), color="grey", linestyle="--")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#e74c3c", label="FBref"), Patch(color="#1a9641", label="Understat")])
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fbref_embedded_lasso.png", dpi=150)
    plt.close(fig)
    print(f"  Zeroed-out: {(lasso_imp == 0).sum()} features")
    print(f"  Saved fbref_embedded_lasso.png")

    print("\n[3b] Embedded: Random Forest Importance ...")
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X[features], y)
    rf_imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

    colors2 = ["#e74c3c" if f in FBREF_FEATURES else "#fdae61" for f in rf_imp.index]
    fig, ax = plt.subplots(figsize=(11, 9))
    rf_imp.plot.barh(ax=ax, color=colors2)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Decrease in Impurity")
    ax.set_title("Embedded — RF Importance (red = FBref features)")
    ax.axvline(rf_imp.mean(), color="grey", linestyle="--")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#e74c3c", label="FBref"), Patch(color="#fdae61", label="Understat")])
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fbref_embedded_rf_importance.png", dpi=150)
    plt.close(fig)
    print(f"  Saved fbref_embedded_rf_importance.png")

    return lasso_imp, rf_imp


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def pca_diagnostic(X: pd.DataFrame, features: list[str]) -> None:
    print("\n[PCA] Scree plot ...")
    X_scaled = StandardScaler().fit_transform(X[features])
    pca = PCA(random_state=42)
    pca.fit(X_scaled)

    exp_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)
    n = len(exp_var)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.bar(range(1, n + 1), exp_var * 100, color="#abd9e9")
    ax1.plot(range(1, n + 1), exp_var * 100, "o-", color="#2c7bb6", markersize=3)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance (%)")
    ax1.set_title("PCA Scree — Individual (Understat + FBref)")

    ax2.plot(range(1, n + 1), cum_var * 100, "o-", color="#d7191c", markersize=3)
    n80 = int(np.searchsorted(cum_var, 0.80)) + 1
    n95 = int(np.searchsorted(cum_var, 0.95)) + 1
    ax2.axhline(80, color="grey", linestyle="--", alpha=0.6, label="80%")
    ax2.axhline(95, color="grey", linestyle="-.", alpha=0.6, label="95%")
    ax2.axvline(n80, color="#2c7bb6", linestyle=":", label=f"{n80} PCs → 80%")
    ax2.axvline(n95, color="#1a9641", linestyle=":", label=f"{n95} PCs → 95%")
    ax2.set_xlabel("Number of PCs")
    ax2.set_ylabel("Cumulative Variance (%)")
    ax2.set_title("PCA Scree — Cumulative (Understat + FBref)")
    ax2.legend(fontsize=8)

    plt.suptitle(f"PCA Diagnostic — {len(features)} features (Understat + FBref)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fbref_pca_scree.png", dpi=150)
    plt.close(fig)
    print(f"  PCs for 80%: {n80},  for 95%: {n95}  (vs. {7} and {12} without FBref features)")
    print(f"  Saved fbref_pca_scree.png")


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

def build_leaderboard(mi, rfecv_s, lasso, rf_imp, features):
    def pct(s): return s.rank(pct=True)
    board = pd.DataFrame({
        "mi_score":       mi,
        "mi_rank_pct":    pct(mi),
        "rfecv_score":    rfecv_s,
        "rfecv_rank_pct": pct(rfecv_s),
        "lasso_coef":     lasso,
        "lasso_rank_pct": pct(lasso),
        "rf_importance":  rf_imp,
        "rf_rank_pct":    pct(rf_imp),
    }, index=features)
    board["composite_score"] = board[["mi_rank_pct","rfecv_rank_pct","lasso_rank_pct","rf_rank_pct"]].mean(axis=1)
    board["is_fbref"] = board.index.map(lambda f: f in FBREF_FEATURES)
    board = board.sort_values("composite_score", ascending=False)
    board.insert(0, "overall_rank", range(1, len(board) + 1))
    return board


def print_leaderboard(board: pd.DataFrame) -> None:
    print("\n" + "=" * 95)
    print("FEATURE LEADERBOARD — Understat + FBref  (★ = FBref-only feature)")
    print("=" * 95)
    display = board[["composite_score", "mi_score", "rfecv_score", "lasso_coef", "rf_importance", "is_fbref"]].copy()
    display.index = [f"★ {f}" if f in FBREF_FEATURES else f"  {f}" for f in display.index]
    print(display.round(4).to_string())
    print("=" * 95)

    top10 = board.head(10)
    fbref_in_top10 = top10[top10["is_fbref"]].index.tolist()
    print(f"\nFBref features in top-10: {len(fbref_in_top10)}")
    for f in fbref_in_top10:
        rank = board.loc[f, "overall_rank"]
        score = board.loc[f, "composite_score"]
        print(f"  Rank {rank}: {f}  (composite={score:.3f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading combined feature matrix ...")
    X, y, features = load()
    understat_present = [f for f in features if f in UNDERSTAT_FEATURES]
    fbref_present = [f for f in features if f in FBREF_FEATURES]
    print(f"  {len(X)} matches, {len(features)} features total")
    print(f"  Understat: {len(understat_present)}   FBref-new: {len(fbref_present)}")

    mi_scores  = filter_method(X, y, features)
    rfecv_scores = wrapper_method(X, y, features)
    lasso_scores, rf_scores = embedded_method(X, y, features)
    pca_diagnostic(X, features)

    board = build_leaderboard(mi_scores, rfecv_scores, lasso_scores, rf_scores, features)
    print_leaderboard(board)

    board.to_csv(OUT_DIR / "fbref_feature_leaderboard.csv")
    print(f"\nLeaderboard saved to {OUT_DIR / 'fbref_feature_leaderboard.csv'}")


if __name__ == "__main__":
    main()
