"""Feature selection: three class methods + PCA diagnostic + leaderboard.

Methods (matching course material):
  1. Filter   — mutual information score + Pearson correlation heatmap
  2. Wrapper  — RFECV with Random Forest (temporal CV, not k-fold)
  3. Embedded — L1-penalised Logistic Regression + Random Forest importances

Outputs saved to outputs/:
  filter_mi_scores.png
  filter_correlation_heatmap.png
  wrapper_rfecv.png
  embedded_lasso.png
  embedded_rf_importance.png
  pca_scree.png
  feature_leaderboard.csv   (feature × method scores + composite rank)
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).parent.parent / "data" / "ligue1_features.csv"
OUT_DIR = Path(__file__).parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# All numeric features to evaluate (excludes IDs, raw team names, target)
FEATURE_COLS = [
    "home_roll_xg_for",
    "home_roll_xg_against",
    "home_roll_xg_diff",
    "away_roll_xg_for",
    "away_roll_xg_against",
    "away_roll_xg_diff",
    "net_xg_diff",
    "home_roll_goals_for",
    "home_roll_goals_against",
    "home_roll_goal_diff",
    "away_roll_goals_for",
    "away_roll_goals_against",
    "away_roll_goal_diff",
    "net_goal_diff",
    "home_roll_won",
    "away_roll_won",
    "net_win_rate",
    "home_roll_points",
    "away_roll_points",
    "net_points",
    "home_squad_value",
    "away_squad_value",
    "squad_value_ratio",
    "log_squad_value_ratio",
    "squad_value_diff",
    "home_avg_player_value",
    "away_avg_player_value",
    "home_squad_age",
    "away_squad_age",
    "home_win_prob",
    "draw_prob",
    "away_win_prob",
]


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].fillna(df[available].median())
    y = df["result_code"]
    return X, y, available


# ---------------------------------------------------------------------------
# Method 1: Filter — Mutual Information + Correlation Heatmap
# ---------------------------------------------------------------------------

def filter_method(X: pd.DataFrame, y: pd.Series, features: list[str]) -> pd.Series:
    print("\n[1] Filter method: Mutual Information ...")
    mi = mutual_info_classif(X[features], y, random_state=42, n_neighbors=5)
    mi_series = pd.Series(mi, index=features).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    mi_series.plot.barh(ax=ax, color="#2c7bb6")
    ax.invert_yaxis()
    ax.set_xlabel("Mutual Information Score")
    ax.set_title("Filter Method — Mutual Information (vs. match result)")
    ax.axvline(mi_series.mean(), color="red", linestyle="--", label="Mean MI")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT_DIR / "filter_mi_scores.png", dpi=150)
    plt.close(fig)
    print(f"  Saved filter_mi_scores.png")

    # Correlation heatmap (select top-20 to keep readable)
    top20 = mi_series.head(20).index.tolist()
    corr = X[top20].corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(top20)))
    ax.set_yticks(range(len(top20)))
    ax.set_xticklabels(top20, rotation=90, fontsize=8)
    ax.set_yticklabels(top20, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation Heatmap — Top-20 features by MI")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "filter_correlation_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved filter_correlation_heatmap.png")

    return mi_series


# ---------------------------------------------------------------------------
# Method 2: Wrapper — RFECV with Random Forest (temporal CV)
# ---------------------------------------------------------------------------

def wrapper_method(X: pd.DataFrame, y: pd.Series, features: list[str]) -> pd.Series:
    print("\n[2] Wrapper method: RFECV with Random Forest ...")
    tscv = TimeSeriesSplit(n_splits=5)
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rfecv = RFECV(
        estimator=rf,
        step=1,
        cv=tscv,
        scoring="neg_log_loss",
        min_features_to_select=5,
        n_jobs=-1,
    )
    rfecv.fit(X[features].values, y.values)

    selected_mask = rfecv.support_
    ranking = rfecv.ranking_

    # Score = 1 if selected (rank 1), else 1/rank
    rfecv_score = pd.Series(
        [1.0 if s else 1.0 / r for s, r in zip(selected_mask, ranking)],
        index=features,
    ).sort_values(ascending=False)

    # Plot CV score vs. number of features
    cv_results = rfecv.cv_results_
    mean_scores = -cv_results["mean_test_score"]  # negate neg_log_loss
    std_scores = cv_results["std_test_score"]
    n_features_range = range(1, len(mean_scores) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(n_features_range, mean_scores, yerr=std_scores,
                fmt="o-", color="#d7191c", capsize=3, linewidth=1.5, markersize=4)
    ax.axvline(rfecv.n_features_, color="green", linestyle="--",
               label=f"Optimal: {rfecv.n_features_} features")
    ax.set_xlabel("Number of features selected")
    ax.set_ylabel("Log-loss (lower = better)")
    ax.set_title("Wrapper Method — RFECV: CV log-loss vs. feature count")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT_DIR / "wrapper_rfecv.png", dpi=150)
    plt.close(fig)
    print(f"  Optimal features: {rfecv.n_features_}")
    print(f"  Selected: {[f for f, s in zip(features, selected_mask) if s]}")
    print(f"  Saved wrapper_rfecv.png")

    return rfecv_score


# ---------------------------------------------------------------------------
# Method 3: Embedded — L1 Logistic + Random Forest importance
# ---------------------------------------------------------------------------

def embedded_method(X: pd.DataFrame, y: pd.Series, features: list[str]) -> tuple[pd.Series, pd.Series]:
    print("\n[3a] Embedded — L1 Logistic Regression ...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[features])

    # l1_ratio=1 → pure L1; saga solver required for elasticnet
    lasso_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", LogisticRegression(
            solver="saga", l1_ratio=1.0, C=0.5,
            max_iter=2000, random_state=42
        )),
    ])
    lasso_pipe.fit(X[features], y)
    coef = lasso_pipe.named_steps["lasso"].coef_  # shape (3, n_features)
    lasso_importance = pd.Series(
        np.abs(coef).mean(axis=0), index=features
    ).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    lasso_importance.plot.barh(ax=ax, color="#1a9641")
    ax.invert_yaxis()
    ax.set_xlabel("Mean |coefficient| across classes")
    ax.set_title("Embedded Method — L1 Logistic Regression Coefficients")
    ax.axvline(lasso_importance.mean(), color="red", linestyle="--", label="Mean")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT_DIR / "embedded_lasso.png", dpi=150)
    plt.close(fig)
    print(f"  Zeroed-out features: {(lasso_importance == 0).sum()}")
    print(f"  Saved embedded_lasso.png")

    print("\n[3b] Embedded — Random Forest Feature Importance ...")
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X[features], y)
    rf_importance = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    rf_importance.plot.barh(ax=ax, color="#fdae61")
    ax.invert_yaxis()
    ax.set_xlabel("Mean Decrease in Impurity")
    ax.set_title("Embedded Method — Random Forest Feature Importance")
    ax.axvline(rf_importance.mean(), color="red", linestyle="--", label="Mean")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT_DIR / "embedded_rf_importance.png", dpi=150)
    plt.close(fig)
    print(f"  Saved embedded_rf_importance.png")

    return lasso_importance, rf_importance


# ---------------------------------------------------------------------------
# PCA scree plot
# ---------------------------------------------------------------------------

def pca_diagnostic(X: pd.DataFrame, features: list[str]) -> None:
    print("\n[PCA] Scree plot ...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[features])
    pca = PCA(random_state=42)
    pca.fit(X_scaled)

    exp_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)
    n_components = len(exp_var)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(range(1, n_components + 1), exp_var * 100, color="#abd9e9")
    ax1.plot(range(1, n_components + 1), exp_var * 100, "o-", color="#2c7bb6", markersize=4)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance (%)")
    ax1.set_title("PCA Scree Plot — Individual Variance")
    ax1.axhline(1 / n_components * 100, color="red", linestyle="--", label="Uniform baseline")
    ax1.legend()

    ax2.plot(range(1, n_components + 1), cum_var * 100, "o-", color="#d7191c", markersize=4)
    ax2.axhline(80, color="grey", linestyle="--", alpha=0.7, label="80% threshold")
    ax2.axhline(95, color="grey", linestyle="-.", alpha=0.7, label="95% threshold")
    n80 = int(np.searchsorted(cum_var, 0.80)) + 1
    n95 = int(np.searchsorted(cum_var, 0.95)) + 1
    ax2.axvline(n80, color="#2c7bb6", linestyle=":", alpha=0.8, label=f"{n80} PCs → 80%")
    ax2.axvline(n95, color="#1a9641", linestyle=":", alpha=0.8, label=f"{n95} PCs → 95%")
    ax2.set_xlabel("Number of Principal Components")
    ax2.set_ylabel("Cumulative Explained Variance (%)")
    ax2.set_title("PCA Scree Plot — Cumulative Variance")
    ax2.legend(fontsize=8)

    plt.suptitle("PCA Diagnostic — All Features", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "pca_scree.png", dpi=150)
    plt.close(fig)
    print(f"  PCs for 80% variance: {n80},  for 95%: {n95}")
    print(f"  Saved pca_scree.png")


# ---------------------------------------------------------------------------
# Feature importance leaderboard
# ---------------------------------------------------------------------------

def build_leaderboard(
    mi: pd.Series,
    rfecv_scores: pd.Series,
    lasso: pd.Series,
    rf_imp: pd.Series,
    features: list[str],
) -> pd.DataFrame:
    """Rank features by how consistently they score across all three methods."""

    def rank_col(s: pd.Series) -> pd.Series:
        """Percentile rank within [0,1] — higher = better."""
        return s.rank(pct=True)

    board = pd.DataFrame({
        "mi_score": mi,
        "mi_rank_pct": rank_col(mi),
        "rfecv_score": rfecv_scores,
        "rfecv_rank_pct": rank_col(rfecv_scores),
        "lasso_coef": lasso,
        "lasso_rank_pct": rank_col(lasso),
        "rf_importance": rf_imp,
        "rf_rank_pct": rank_col(rf_imp),
    }, index=features)

    # Composite = equal-weight average of per-method percentile ranks
    board["composite_score"] = board[["mi_rank_pct", "rfecv_rank_pct", "lasso_rank_pct", "rf_rank_pct"]].mean(axis=1)
    board = board.sort_values("composite_score", ascending=False)
    board["overall_rank"] = range(1, len(board) + 1)

    return board


def print_leaderboard(board: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE LEADERBOARD")
    print("(ranked by composite percentile score across all three selection methods)")
    print("=" * 80)
    display_cols = ["composite_score", "mi_score", "rfecv_score", "lasso_coef", "rf_importance"]
    print(board[display_cols].round(4).to_string())
    print("=" * 80)

    top10 = board.head(10).index.tolist()
    print(f"\nTop-10 features that survive all methods:")
    for i, f in enumerate(top10, 1):
        print(f"  {i:2d}. {f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading feature matrix ...")
    X, y, features = load()
    print(f"  {len(X)} matches × {len(features)} features")
    print(f"  Target distribution: {y.value_counts().to_dict()}")

    mi_scores = filter_method(X, y, features)
    rfecv_scores = wrapper_method(X, y, features)
    lasso_scores, rf_scores = embedded_method(X, y, features)
    pca_diagnostic(X, features)

    board = build_leaderboard(mi_scores, rfecv_scores, lasso_scores, rf_scores, features)
    print_leaderboard(board)

    board_path = OUT_DIR / "feature_leaderboard.csv"
    board.to_csv(board_path)
    print(f"\nLeaderboard saved to {board_path}")

    # Also persist top features list to data/
    top_features = board.head(15).index.tolist()
    top_path = Path(__file__).parent.parent / "data" / "top_features.txt"
    top_path.write_text("\n".join(top_features))
    print(f"Top-15 features saved to {top_path}")


if __name__ == "__main__":
    main()
