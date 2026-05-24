"""Feature selection on the combined Understat + player-level FBref feature set.

Applies the same three-method protocol as feature_selection.py on the
extended feature matrix that includes true player-level signals:
  - Per-player rolling shots/SoT/defensive actions/goals/assists per 90
  - Aggregated to starting-XI level weighted by minutes played
  - Tracked per individual (not per team) so transfers don't reset form

Player features added by feature_engineering_player.py (prefixed home_pl_ / away_pl_ / net_pl_):
  *_sh_p90        — shots per 90 (weighted mean of starting XI rolling)
  *_sot_p90       — shots on target per 90
  *_def_p90       — (TklW + Int) per 90
  *_gls_p90       — goals per 90
  *_ast_p90       — assists per 90
  *_starter_mins  — mean rolling minutes (fitness/regularity proxy)
  *_pct_fw        — fraction of starters who are forwards
  *_depth_sh      — shot gap between top-half and bottom-half starters

Outputs saved to outputs/player/:
  player_filter_mi_scores.png
  player_filter_correlation_heatmap.png
  player_wrapper_rfecv.png
  player_embedded_lasso.png
  player_embedded_rf_importance.png
  player_pca_scree.png
  player_feature_leaderboard.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).parent.parent / "data" / "ligue1_player_features.csv"

# FBref team-level features (from feature_engineering_fbref.py)
FBREF_TEAM_FEATURES = [
    "home_roll_poss", "away_roll_poss", "net_poss",
    "home_attackers", "away_attackers", "net_attackers",
    "home_player_ga90", "away_player_ga90", "net_player_ga90",
    "home_starter_ga90", "away_starter_ga90", "net_starter_ga90",
    "home_squad_depth", "away_squad_depth", "net_squad_depth",
    "home_minutes_conc", "away_minutes_conc",
]
OUT_DIR   = Path(__file__).parent.parent / "outputs" / "player"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Understat + Transfermarkt features already tested in feature_selection.py
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

# True player-level features from feature_engineering_player.py
PLAYER_FEATURES = [
    "home_pl_sh_p90",    "away_pl_sh_p90",    "net_pl_sh_p90",
    "home_pl_sot_p90",   "away_pl_sot_p90",   "net_pl_sot_p90",
    "home_pl_def_p90",   "away_pl_def_p90",   "net_pl_def_p90",
    "home_pl_gls_p90",   "away_pl_gls_p90",   "net_pl_gls_p90",
    "home_pl_ast_p90",   "away_pl_ast_p90",   "net_pl_ast_p90",
    "home_pl_starter_mins", "away_pl_starter_mins", "net_pl_starter_mins",
    "home_pl_pct_fw",    "away_pl_pct_fw",    "net_pl_pct_fw",
    "home_pl_depth_sh",  "away_pl_depth_sh",  "net_pl_depth_sh",
]

ALL_FEATURES = UNDERSTAT_FEATURES + FBREF_TEAM_FEATURES + PLAYER_FEATURES


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    available = [c for c in ALL_FEATURES if c in df.columns]
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        print(f"  Warning: {len(missing)} features not found in file: {missing[:5]}{'...' if len(missing)>5 else ''}")
    X = df[available].fillna(df[available].median())
    y = df["result_code"]
    return X, y, available


# ---------------------------------------------------------------------------
# Method 1: Filter — MI scores + cross-group correlation heatmap
# ---------------------------------------------------------------------------

def filter_method(X: pd.DataFrame, y: pd.Series, features: list[str]) -> pd.Series:
    print("\n[1] Filter: Mutual Information ...")
    mi = mutual_info_classif(X[features], y, random_state=42, n_neighbors=5)
    mi_series = pd.Series(mi, index=features).sort_values(ascending=False)

    def _color(f):
        if f in PLAYER_FEATURES: return "#e74c3c"
        if f in FBREF_TEAM_FEATURES: return "#9b59b6"
        return "#2c7bb6"

    colors = [_color(f) for f in mi_series.index]
    fig, ax = plt.subplots(figsize=(11, max(9, len(features) * 0.22)))
    mi_series.plot.barh(ax=ax, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Mutual Information Score")
    ax.set_title("Filter Method — MI Score\n(red = player-level, purple = FBref team-level, blue = Understat/Transfermarkt)")
    ax.axvline(mi_series.mean(), color="grey", linestyle="--", label="Mean MI")
    ax.legend(handles=[
        Patch(color="#e74c3c", label="Player-level feature"),
        Patch(color="#9b59b6", label="FBref team-level feature"),
        Patch(color="#2c7bb6", label="Understat/Transfermarkt feature"),
    ])
    plt.tight_layout()
    fig.savefig(OUT_DIR / "player_filter_mi_scores.png", dpi=150)
    plt.close(fig)

    player_present = [f for f in features if f in PLAYER_FEATURES]
    fbref_team_present = [f for f in features if f in FBREF_TEAM_FEATURES]
    top5_player = [f for f in mi_series.index if f in PLAYER_FEATURES][:5]
    print(f"  Player features present: {len(player_present)}")
    print(f"  FBref team-level features present: {len(fbref_team_present)}")
    print(f"  Top-5 player features by MI: {top5_player}")
    print(f"  Saved player_filter_mi_scores.png")

    # Heatmap: top-12 Understat features vs all player features
    top_understat = [f for f in mi_series.index if f in UNDERSTAT_FEATURES][:12]
    player_present_list = [f for f in PLAYER_FEATURES if f in features]
    heatmap_cols = top_understat + player_present_list
    corr = X[heatmap_cols].corr()

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(heatmap_cols)))
    ax.set_yticks(range(len(heatmap_cols)))
    ax.set_xticklabels(heatmap_cols, rotation=90, fontsize=7)
    ax.set_yticklabels(heatmap_cols, fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    div = len(top_understat) - 0.5
    ax.axhline(div, color="black", linewidth=1.5)
    ax.axvline(div, color="black", linewidth=1.5)
    ax.set_title(
        "Correlation Heatmap — top Understat features vs. player-level features\n"
        "(dividing line separates the two groups; high cross-group correlation = redundancy)"
    )
    plt.tight_layout()
    fig.savefig(OUT_DIR / "player_filter_correlation_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved player_filter_correlation_heatmap.png")

    return mi_series


# ---------------------------------------------------------------------------
# Method 2: Wrapper — RFECV with temporal CV
# ---------------------------------------------------------------------------

def wrapper_method(X: pd.DataFrame, y: pd.Series, features: list[str]) -> pd.Series:
    print("\n[2] Wrapper: RFECV (TimeSeriesSplit, n_splits=5) ...")
    tscv = TimeSeriesSplit(n_splits=5)
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rfecv = RFECV(
        estimator=rf, step=1, cv=tscv, scoring="neg_log_loss",
        min_features_to_select=5, n_jobs=-1,
    )
    rfecv.fit(X[features].values, y.values)

    selected_mask = rfecv.support_
    ranking = rfecv.ranking_
    rfecv_score = pd.Series(
        [1.0 if s else 1.0 / r for s, r in zip(selected_mask, ranking)],
        index=features,
    ).sort_values(ascending=False)

    mean_scores = -rfecv.cv_results_["mean_test_score"]
    std_scores   =  rfecv.cv_results_["std_test_score"]
    n_range = range(1, len(mean_scores) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(n_range, mean_scores, yerr=std_scores, fmt="o-",
                color="#d7191c", capsize=3, linewidth=1.5, markersize=4)
    ax.axvline(rfecv.n_features_, color="green", linestyle="--",
               label=f"Optimal: {rfecv.n_features_} features")
    ax.set_xlabel("Number of features")
    ax.set_ylabel("Log-loss (lower = better)")
    ax.set_title("Wrapper Method — RFECV: CV log-loss vs. feature count (Understat + player-level)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT_DIR / "player_wrapper_rfecv.png", dpi=150)
    plt.close(fig)

    selected = [f for f, s in zip(features, selected_mask) if s]
    player_selected = [f for f in selected if f in PLAYER_FEATURES]
    print(f"  Optimal features: {rfecv.n_features_}")
    print(f"  Player features selected by RFECV: {player_selected}")
    print(f"  Saved player_wrapper_rfecv.png")
    return rfecv_score


# ---------------------------------------------------------------------------
# Method 3: Embedded — L1 logistic + RF importance
# ---------------------------------------------------------------------------

def embedded_method(
    X: pd.DataFrame, y: pd.Series, features: list[str]
) -> tuple[pd.Series, pd.Series]:
    print("\n[3a] Embedded: L1 Logistic Regression ...")
    lasso = LogisticRegression(
        solver="saga", l1_ratio=1.0, C=0.5, max_iter=2000, random_state=42
    )
    X_scaled = StandardScaler().fit_transform(X[features])
    lasso.fit(X_scaled, y)
    coef = np.abs(lasso.coef_).mean(axis=0)
    lasso_imp = pd.Series(coef, index=features).sort_values(ascending=False)

    def _lc(f):
        if f in PLAYER_FEATURES: return "#e74c3c"
        if f in FBREF_TEAM_FEATURES: return "#9b59b6"
        return "#1a9641"

    colors = [_lc(f) for f in lasso_imp.index]
    fig, ax = plt.subplots(figsize=(11, max(9, len(features) * 0.22)))
    lasso_imp.plot.barh(ax=ax, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |L1 coefficient| across classes")
    ax.set_title("Embedded — L1 Logistic Regression\n(red = player-level, purple = FBref team, green = Understat/Transfermarkt)")
    ax.axvline(lasso_imp.mean(), color="grey", linestyle="--")
    ax.legend(handles=[
        Patch(color="#e74c3c", label="Player-level"),
        Patch(color="#9b59b6", label="FBref team-level"),
        Patch(color="#1a9641", label="Understat/Transfermarkt"),
    ])
    plt.tight_layout()
    fig.savefig(OUT_DIR / "player_embedded_lasso.png", dpi=150)
    plt.close(fig)

    player_zeroed = [f for f in PLAYER_FEATURES if f in features and lasso_imp[f] == 0]
    print(f"  Total zeroed-out: {(lasso_imp == 0).sum()}, player features zeroed: {len(player_zeroed)}")
    print(f"  Saved player_embedded_lasso.png")

    print("\n[3b] Embedded: Random Forest Importance ...")
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X[features], y)
    rf_imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

    def _rc(f):
        if f in PLAYER_FEATURES: return "#e74c3c"
        if f in FBREF_TEAM_FEATURES: return "#9b59b6"
        return "#fdae61"

    colors2 = [_rc(f) for f in rf_imp.index]
    fig, ax = plt.subplots(figsize=(11, max(9, len(features) * 0.22)))
    rf_imp.plot.barh(ax=ax, color=colors2)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Decrease in Impurity")
    ax.set_title("Embedded — RF Importance\n(red = player-level, purple = FBref team, orange = Understat/Transfermarkt)")
    ax.axvline(rf_imp.mean(), color="grey", linestyle="--")
    ax.legend(handles=[
        Patch(color="#e74c3c", label="Player-level"),
        Patch(color="#9b59b6", label="FBref team-level"),
        Patch(color="#fdae61", label="Understat/Transfermarkt"),
    ])
    plt.tight_layout()
    fig.savefig(OUT_DIR / "player_embedded_rf_importance.png", dpi=150)
    plt.close(fig)
    print(f"  Saved player_embedded_rf_importance.png")

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
    n80 = int(np.searchsorted(cum_var, 0.80)) + 1
    n95 = int(np.searchsorted(cum_var, 0.95)) + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.bar(range(1, n + 1), exp_var * 100, color="#abd9e9")
    ax1.plot(range(1, n + 1), exp_var * 100, "o-", color="#2c7bb6", markersize=3)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance (%)")
    ax1.set_title("PCA Scree — Individual (Understat + player-level)")

    ax2.plot(range(1, n + 1), cum_var * 100, "o-", color="#d7191c", markersize=3)
    ax2.axhline(80, color="grey", linestyle="--", alpha=0.6, label="80%")
    ax2.axhline(95, color="grey", linestyle="-.", alpha=0.6, label="95%")
    ax2.axvline(n80, color="#2c7bb6", linestyle=":", label=f"{n80} PCs → 80%")
    ax2.axvline(n95, color="#1a9641", linestyle=":", label=f"{n95} PCs → 95%")
    ax2.set_xlabel("Number of PCs")
    ax2.set_ylabel("Cumulative Variance (%)")
    ax2.set_title("PCA Scree — Cumulative (Understat + player-level)")
    ax2.legend(fontsize=8)

    plt.suptitle(
        f"PCA Diagnostic — {len(features)} features "
        f"({len([f for f in features if f in PLAYER_FEATURES])} player-level, "
        f"{len([f for f in features if f in UNDERSTAT_FEATURES])} Understat)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(OUT_DIR / "player_pca_scree.png", dpi=150)
    plt.close(fig)
    print(f"  PCs for 80% variance: {n80},  for 95%: {n95}")
    print(f"  Saved player_pca_scree.png")


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

def build_leaderboard(
    mi: pd.Series, rfecv_s: pd.Series, lasso: pd.Series, rf_imp: pd.Series, features: list[str]
) -> pd.DataFrame:
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
    board["composite_score"] = board[
        ["mi_rank_pct", "rfecv_rank_pct", "lasso_rank_pct", "rf_rank_pct"]
    ].mean(axis=1)

    def _source(f):
        if f in PLAYER_FEATURES:    return "player"
        if f in FBREF_TEAM_FEATURES: return "fbref_team"
        return "understat"

    board["source"] = board.index.map(_source)
    board = board.sort_values("composite_score", ascending=False)
    board.insert(0, "overall_rank", range(1, len(board) + 1))
    return board


def print_leaderboard(board: pd.DataFrame) -> None:
    print("\n" + "=" * 105)
    print("FEATURE LEADERBOARD — All sources  (★ = player-level, • = FBref team-level)")
    print("=" * 105)
    display = board[[
        "composite_score", "mi_score", "rfecv_score", "lasso_coef", "rf_importance", "source"
    ]].copy()

    def _tag(f):
        if f in PLAYER_FEATURES:     return f"★ {f}"
        if f in FBREF_TEAM_FEATURES: return f"• {f}"
        return f"  {f}"

    display.index = [_tag(f) for f in display.index]
    print(display.round(4).to_string())
    print("=" * 105)

    top10 = board.head(10)
    for src, label in [("player", "Player-level"), ("fbref_team", "FBref team-level")]:
        in_top10 = top10[top10["source"] == src].index.tolist()
        print(f"\n{label} features in top-10: {len(in_top10)}")
        for f in in_top10:
            rank = board.loc[f, "overall_rank"]
            score = board.loc[f, "composite_score"]
            print(f"  Rank {rank}: {f}  (composite={score:.3f})")

    print("\nAll player-level features ranked:")
    for _, row in board[board["source"] == "player"].iterrows():
        print(f"  Rank {int(row['overall_rank'])}: {row.name}  "
              f"(composite={row['composite_score']:.3f}, MI={row['mi_score']:.4f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Loading player feature matrix from {DATA_PATH} ...")
    X, y, features = load()
    understat_present  = [f for f in features if f in UNDERSTAT_FEATURES]
    fbref_team_present = [f for f in features if f in FBREF_TEAM_FEATURES]
    player_present     = [f for f in features if f in PLAYER_FEATURES]
    print(f"  {len(X)} matches, {len(features)} features total")
    print(f"  Understat/Transfermarkt: {len(understat_present)}   "
          f"FBref team-level: {len(fbref_team_present)}   "
          f"Player-level: {len(player_present)}")
    print(f"  Class distribution: {y.value_counts().to_dict()}")

    if len(player_present) == 0:
        print("\nERROR: No player-level features found in the dataset.")
        print("Run src/feature_engineering_player.py first to generate ligue1_player_features.csv.")
        return

    mi_scores    = filter_method(X, y, features)
    rfecv_scores = wrapper_method(X, y, features)
    lasso_scores, rf_scores = embedded_method(X, y, features)
    pca_diagnostic(X, features)

    board = build_leaderboard(mi_scores, rfecv_scores, lasso_scores, rf_scores, features)
    print_leaderboard(board)

    board.to_csv(OUT_DIR / "player_feature_leaderboard.csv")
    print(f"\nLeaderboard saved → {OUT_DIR / 'player_feature_leaderboard.csv'}")
    print("\nAll outputs saved to outputs/player/")


if __name__ == "__main__":
    main()
