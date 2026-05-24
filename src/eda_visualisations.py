"""
EDA Visualisations — Ligue 1 Player Match Prediction
=====================================================
Generates the full exploratory analysis suite:

  1.  Target distribution (bar + pie)
  2.  Feature histograms by result (top-20 features, 3 colours)
  3.  Box plots by result (top-20 features)
  4.  Scatter matrix of top-8 features coloured by result
  5.  Scatter: bookmaker probs vs xG rolling form
  6.  Scatter: squad value ratio vs net xG diff
  7.  Scatter: player-level features vs bookmaker probs
  8.  Time-series: average feature values per season
  9.  Null / missingness heatmap
  10. Per-season class distribution
  11. Pairwise scatter grid — player-level vs understat top features
  12. Feature importance summary (from player_feature_leaderboard.csv)

All outputs saved to  outputs/eda/
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

# ── paths ──────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "ligue1_player_features.csv"
BOARD_PATH = ROOT / "outputs" / "player" / "player_feature_leaderboard.csv"
OUT_DIR   = ROOT / "outputs" / "eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── colour palette ─────────────────────────────────────────────────────────
RESULT_PALETTE = {"H": "#2ecc71", "D": "#f39c12", "A": "#e74c3c"}
SOURCE_PALETTE = {
    "understat":  "#2c7bb6",
    "fbref_team": "#9b59b6",
    "player":     "#e74c3c",
}
LABEL_MAP = {"H": "Home Win", "D": "Draw", "A": "Away Win"}

# ── feature groups ─────────────────────────────────────────────────────────
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
FBREF_TEAM_FEATURES = [
    "home_roll_poss", "away_roll_poss", "net_poss",
    "home_attackers", "away_attackers", "net_attackers",
    "home_player_ga90", "away_player_ga90", "net_player_ga90",
    "home_starter_ga90", "away_starter_ga90", "net_starter_ga90",
    "home_squad_depth", "away_squad_depth", "net_squad_depth",
    "home_minutes_conc", "away_minutes_conc",
]

ALL_FEATURES = UNDERSTAT_FEATURES + FBREF_TEAM_FEATURES + PLAYER_FEATURES


def feat_color(f: str) -> str:
    if f in PLAYER_FEATURES:    return SOURCE_PALETTE["player"]
    if f in FBREF_TEAM_FEATURES: return SOURCE_PALETTE["fbref_team"]
    return SOURCE_PALETTE["understat"]


# ═══════════════════════════════════════════════════════════════════════════
# 1. Load data
# ═══════════════════════════════════════════════════════════════════════════
def load() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2. Target distribution
# ═══════════════════════════════════════════════════════════════════════════
def plot_target_distribution(df: pd.DataFrame) -> None:
    counts = df["result"].value_counts().reindex(["H", "D", "A"])
    pcts   = counts / counts.sum() * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # bar
    bars = ax1.bar(
        [LABEL_MAP[r] for r in counts.index],
        counts.values,
        color=[RESULT_PALETTE[r] for r in counts.index],
        edgecolor="white", linewidth=0.8,
    )
    for bar, cnt, pct in zip(bars, counts.values, pcts.values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                 f"{cnt}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=11)
    ax1.set_title("Match Outcome Distribution — Ligue 1 2018–2023", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Number of Matches")
    ax1.set_ylim(0, counts.max() * 1.18)
    ax1.spines[["top", "right"]].set_visible(False)

    # pie
    wedges, texts, autotexts = ax2.pie(
        counts.values,
        labels=[LABEL_MAP[r] for r in counts.index],
        colors=[RESULT_PALETTE[r] for r in counts.index],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
    )
    for at in autotexts:
        at.set_fontsize(12)
    ax2.set_title("Class Imbalance Overview", fontsize=13, fontweight="bold")

    plt.suptitle(
        f"Total matches: {len(df)} | Seasons 2018–2023 | 29 teams",
        fontsize=10, color="grey",
    )
    plt.tight_layout()
    fig.savefig(OUT_DIR / "01_target_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 01_target_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Per-season class distribution
# ═══════════════════════════════════════════════════════════════════════════
def plot_season_distribution(df: pd.DataFrame) -> None:
    seasons = sorted(df["season"].unique())
    data = {}
    for s in seasons:
        sub = df[df["season"] == s]["result"].value_counts().reindex(["H", "D", "A"], fill_value=0)
        data[s] = sub

    season_df = pd.DataFrame(data).T
    pct_df    = season_df.div(season_df.sum(axis=1), axis=0) * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

    bottom = np.zeros(len(seasons))
    for outcome in ["H", "D", "A"]:
        vals = season_df[outcome].values
        ax1.bar(range(len(seasons)), vals, bottom=bottom,
                color=RESULT_PALETTE[outcome], label=LABEL_MAP[outcome],
                edgecolor="white", linewidth=0.6)
        bottom += vals
    ax1.set_xticks(range(len(seasons)))
    ax1.set_xticklabels(seasons)
    ax1.set_ylabel("Number of Matches")
    ax1.set_title("Match Outcomes per Season (absolute count)", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.spines[["top", "right"]].set_visible(False)

    bottom2 = np.zeros(len(seasons))
    for outcome in ["H", "D", "A"]:
        vals = pct_df[outcome].values
        ax2.bar(range(len(seasons)), vals, bottom=bottom2,
                color=RESULT_PALETTE[outcome], label=LABEL_MAP[outcome],
                edgecolor="white", linewidth=0.6)
        bottom2 += vals
    ax2.set_xticks(range(len(seasons)))
    ax2.set_xticklabels(seasons)
    ax2.set_ylabel("Share (%)")
    ax2.set_title("Match Outcomes per Season (normalised %)", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right")
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "02_season_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 02_season_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Missingness heatmap
# ═══════════════════════════════════════════════════════════════════════════
def plot_missingness(df: pd.DataFrame) -> None:
    feat_cols = [c for c in ALL_FEATURES if c in df.columns]
    miss = df[feat_cols].isnull().mean() * 100
    miss = miss[miss > 0].sort_values(ascending=False)
    if miss.empty:
        print("  ✓ No missing values — skipping missingness plot")
        return

    colors = [feat_color(f) for f in miss.index]
    fig, ax = plt.subplots(figsize=(10, max(5, len(miss) * 0.28)))
    bars = ax.barh(miss.index[::-1], miss.values[::-1], color=colors[::-1],
                   edgecolor="white", linewidth=0.6)
    for bar, val in zip(bars, miss.values[::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("Missing (%)")
    ax.set_title("Feature Missingness (% null per feature)", fontsize=12, fontweight="bold")
    ax.axvline(25, color="grey", linestyle="--", alpha=0.5, label="25% threshold")
    ax.legend(handles=[
        Patch(color=SOURCE_PALETTE["player"],    label="Player-level"),
        Patch(color=SOURCE_PALETTE["fbref_team"], label="FBref team-level"),
        Patch(color=SOURCE_PALETTE["understat"], label="Understat/Transfermarkt"),
        Patch(color="none", label=""),
    ] + [plt.Line2D([0], [0], color="grey", linestyle="--", label="25% threshold")],
    fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "03_missingness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 03_missingness.png")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Feature histograms by result — top-20 from leaderboard
# ═══════════════════════════════════════════════════════════════════════════
def plot_histograms_by_result(df: pd.DataFrame, top_features: list[str]) -> None:
    feat_cols = [f for f in top_features if f in df.columns][:20]
    n         = len(feat_cols)
    ncols     = 4
    nrows     = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5))
    axes = axes.flatten()

    for i, feat in enumerate(feat_cols):
        ax = axes[i]
        for result in ["H", "D", "A"]:
            sub = df[df["result"] == result][feat].dropna()
            ax.hist(sub, bins=30, alpha=0.55, color=RESULT_PALETTE[result],
                    label=LABEL_MAP[result], density=True, edgecolor="none")
        ax.set_title(feat, fontsize=9, fontweight="bold")
        ax.set_xlabel("Value", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        # colour the title by feature group
        ax.title.set_color(feat_color(feat))

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # shared legend
    handles = [Patch(color=RESULT_PALETTE[r], label=LABEL_MAP[r]) for r in ["H", "D", "A"]]
    fig.legend(handles=handles, loc="lower right", fontsize=10, framealpha=0.9)
    fig.suptitle(
        "Feature Distributions by Match Outcome — Top-20 Features\n"
        "(title colour: red=player-level, purple=FBref-team, blue=Understat)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(OUT_DIR / "04_histograms_by_result.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 04_histograms_by_result.png")


# ═══════════════════════════════════════════════════════════════════════════
# 6. Box plots by result — top-20 features
# ═══════════════════════════════════════════════════════════════════════════
def plot_boxplots_by_result(df: pd.DataFrame, top_features: list[str]) -> None:
    feat_cols = [f for f in top_features if f in df.columns][:20]
    n     = len(feat_cols)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.8))
    axes = axes.flatten()

    for i, feat in enumerate(feat_cols):
        ax = axes[i]
        data_by_result = [
            df[df["result"] == r][feat].dropna().values for r in ["H", "D", "A"]
        ]
        bp = ax.boxplot(
            data_by_result,
            patch_artist=True,
            widths=0.55,
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker=".", markersize=3, alpha=0.4),
        )
        for patch, result in zip(bp["boxes"], ["H", "D", "A"]):
            patch.set_facecolor(RESULT_PALETTE[result])
            patch.set_alpha(0.8)

        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["H", "D", "A"], fontsize=9)
        ax.set_title(feat, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.title.set_color(feat_color(feat))

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    handles = [Patch(color=RESULT_PALETTE[r], label=LABEL_MAP[r]) for r in ["H", "D", "A"]]
    fig.legend(handles=handles, loc="lower right", fontsize=10, framealpha=0.9)
    fig.suptitle(
        "Feature Box Plots by Match Outcome — Top-20 Features\n"
        "(title colour: red=player-level, purple=FBref-team, blue=Understat)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(OUT_DIR / "05_boxplots_by_result.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 05_boxplots_by_result.png")


# ═══════════════════════════════════════════════════════════════════════════
# 7. Scatter: bookmaker probs vs rolling xG form
# ═══════════════════════════════════════════════════════════════════════════
def plot_bookmaker_vs_xg(df: pd.DataFrame) -> None:
    pairs = [
        ("home_win_prob", "home_roll_xg_diff",  "Home win prob vs Home rolling xG diff"),
        ("away_win_prob", "away_roll_xg_diff",  "Away win prob vs Away rolling xG diff"),
        ("home_win_prob", "net_xg_diff",        "Home win prob vs Net xG diff"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (xf, yf, title) in zip(axes, pairs):
        if xf not in df.columns or yf not in df.columns:
            ax.set_visible(False)
            continue
        for result in ["H", "D", "A"]:
            sub = df[df["result"] == result]
            ax.scatter(sub[xf], sub[yf], c=RESULT_PALETTE[result],
                       alpha=0.3, s=14, label=LABEL_MAP[result], linewidths=0)
        # regression line
        valid = df[[xf, yf]].dropna()
        if len(valid) > 10:
            m, b = np.polyfit(valid[xf], valid[yf], 1)
            x_line = np.linspace(valid[xf].min(), valid[xf].max(), 100)
            ax.plot(x_line, m * x_line + b, "k--", linewidth=1.2, alpha=0.7)
        ax.set_xlabel(xf, fontsize=9)
        ax.set_ylabel(yf, fontsize=9)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)

    handles = [Patch(color=RESULT_PALETTE[r], label=LABEL_MAP[r]) for r in ["H", "D", "A"]]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Bookmaker Probabilities vs Rolling xG Form", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(OUT_DIR / "06_bookmaker_vs_xg_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 06_bookmaker_vs_xg_scatter.png")


# ═══════════════════════════════════════════════════════════════════════════
# 8. Scatter: squad value ratio vs net xG diff
# ═══════════════════════════════════════════════════════════════════════════
def plot_value_vs_xg(df: pd.DataFrame) -> None:
    pairs = [
        ("squad_value_ratio",     "net_xg_diff",         "Squad value ratio vs Net xG diff"),
        ("log_squad_value_ratio", "home_win_prob",        "Log squad value ratio vs Home win prob"),
        ("squad_value_ratio",     "home_win_prob",        "Squad value ratio vs Home win prob"),
        ("squad_value_diff",      "net_goal_diff",        "Squad value diff vs Net goal diff"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (xf, yf, title) in zip(axes, pairs):
        if xf not in df.columns or yf not in df.columns:
            ax.set_visible(False)
            continue
        for result in ["H", "D", "A"]:
            sub = df[df["result"] == result]
            ax.scatter(sub[xf], sub[yf], c=RESULT_PALETTE[result],
                       alpha=0.3, s=14, label=LABEL_MAP[result], linewidths=0)
        valid = df[[xf, yf]].dropna()
        if len(valid) > 10:
            m, b = np.polyfit(valid[xf], valid[yf], 1)
            x_line = np.linspace(valid[xf].quantile(0.01), valid[xf].quantile(0.99), 100)
            ax.plot(x_line, m * x_line + b, "k--", linewidth=1.2, alpha=0.7)
        ax.set_xlabel(xf, fontsize=9)
        ax.set_ylabel(yf, fontsize=9)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)

    handles = [Patch(color=RESULT_PALETTE[r], label=LABEL_MAP[r]) for r in ["H", "D", "A"]]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Squad Market Value vs xG / Bookmaker Features", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUT_DIR / "07_squad_value_vs_xg_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 07_squad_value_vs_xg_scatter.png")


# ═══════════════════════════════════════════════════════════════════════════
# 9. Player-level features vs bookmaker probs
# ═══════════════════════════════════════════════════════════════════════════
def plot_player_vs_bookmaker(df: pd.DataFrame) -> None:
    player_x = [f for f in PLAYER_FEATURES if f in df.columns]
    focus = [
        "net_pl_pct_fw", "net_pl_starter_mins", "net_pl_sot_p90",
        "home_pl_sot_p90", "home_pl_def_p90", "home_pl_sh_p90",
    ]
    focus = [f for f in focus if f in df.columns][:6]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for ax, feat in zip(axes, focus):
        for result in ["H", "D", "A"]:
            sub = df[df["result"] == result].dropna(subset=[feat, "home_win_prob"])
            ax.scatter(sub[feat], sub["home_win_prob"],
                       c=RESULT_PALETTE[result], alpha=0.3, s=14,
                       label=LABEL_MAP[result], linewidths=0)
        valid = df[[feat, "home_win_prob"]].dropna()
        if len(valid) > 10:
            m, b = np.polyfit(valid[feat], valid["home_win_prob"], 1)
            x_line = np.linspace(valid[feat].quantile(0.02), valid[feat].quantile(0.98), 100)
            ax.plot(x_line, m * x_line + b, color="#e74c3c", linestyle="--",
                    linewidth=1.4, alpha=0.8)
        ax.set_xlabel(feat, fontsize=9)
        ax.set_ylabel("home_win_prob", fontsize=9)
        ax.set_title(f"{feat}\nvs home_win_prob", fontsize=8, fontweight="bold", color="#e74c3c")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)

    handles = [Patch(color=RESULT_PALETTE[r], label=LABEL_MAP[r]) for r in ["H", "D", "A"]]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        "Player-Level Features vs Bookmaker Home Win Probability\n"
        "(red dashed = linear trend)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(OUT_DIR / "08_player_features_vs_bookmaker.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 08_player_features_vs_bookmaker.png")


# ═══════════════════════════════════════════════════════════════════════════
# 10. Scatter matrix of top-8 features
# ═══════════════════════════════════════════════════════════════════════════
def plot_scatter_matrix(df: pd.DataFrame, top_features: list[str]) -> None:
    cols = [f for f in top_features if f in df.columns][:8]
    sub  = df[cols + ["result"]].dropna(subset=cols)

    fig_size = len(cols) * 2.4
    fig, axes = plt.subplots(len(cols), len(cols), figsize=(fig_size, fig_size))

    for i, fi in enumerate(cols):
        for j, fj in enumerate(cols):
            ax = axes[i][j]
            if i == j:
                # diagonal: histogram
                for result in ["H", "D", "A"]:
                    vals = sub[sub["result"] == result][fi].values
                    ax.hist(vals, bins=20, alpha=0.5, color=RESULT_PALETTE[result],
                            density=True)
                ax.set_title(fi, fontsize=6, fontweight="bold", color=feat_color(fi))
            else:
                for result in ["H", "D", "A"]:
                    s = sub[sub["result"] == result]
                    ax.scatter(s[fj], s[fi], c=RESULT_PALETTE[result],
                               alpha=0.2, s=6, linewidths=0)
            ax.tick_params(labelsize=5, left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
            if j == 0:
                ax.set_ylabel(fi, fontsize=6, rotation=0, ha="right", va="center")
            if i == len(cols) - 1:
                ax.set_xlabel(fj, fontsize=6, rotation=30, ha="right")

    handles = [Patch(color=RESULT_PALETTE[r], label=LABEL_MAP[r]) for r in ["H", "D", "A"]]
    fig.legend(handles=handles, loc="upper right", fontsize=9, bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Scatter Matrix — Top-8 Features by Composite Score", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 0.97, 0.97])
    fig.savefig(OUT_DIR / "09_scatter_matrix_top8.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 09_scatter_matrix_top8.png")


# ═══════════════════════════════════════════════════════════════════════════
# 11. Time series: average feature values per season
# ═══════════════════════════════════════════════════════════════════════════
def plot_feature_trends(df: pd.DataFrame) -> None:
    feats = ["home_win_prob", "away_win_prob", "squad_value_ratio",
             "net_xg_diff", "net_poss", "net_pl_pct_fw", "net_pl_sot_p90"]
    feats = [f for f in feats if f in df.columns]

    seasons = sorted(df["season"].unique())
    fig, axes = plt.subplots(len(feats), 1, figsize=(12, len(feats) * 2.2), sharex=True)
    if len(feats) == 1:
        axes = [axes]

    for ax, feat in zip(axes, feats):
        for result in ["H", "D", "A"]:
            vals = []
            for s in seasons:
                sub = df[(df["season"] == s) & (df["result"] == result)][feat].dropna()
                vals.append(sub.mean() if len(sub) else np.nan)
            ax.plot(range(len(seasons)), vals, "o-", color=RESULT_PALETTE[result],
                    label=LABEL_MAP[result], linewidth=1.8, markersize=5)
        ax.set_ylabel(feat, fontsize=8, color=feat_color(feat))
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.set_xticks(range(len(seasons)))

    axes[-1].set_xticklabels(seasons, fontsize=9)
    axes[-1].set_xlabel("Season", fontsize=10)

    handles = [Patch(color=RESULT_PALETTE[r], label=LABEL_MAP[r]) for r in ["H", "D", "A"]]
    fig.legend(handles=handles, loc="lower right", fontsize=10)
    fig.suptitle("Average Feature Values per Season (by Outcome)", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 0.92, 0.97])
    fig.savefig(OUT_DIR / "10_feature_trends_by_season.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 10_feature_trends_by_season.png")


# ═══════════════════════════════════════════════════════════════════════════
# 12. Player-level features: home vs away comparison
# ═══════════════════════════════════════════════════════════════════════════
def plot_home_vs_away_player(df: pd.DataFrame) -> None:
    pairs = [
        ("home_pl_sot_p90",  "away_pl_sot_p90",  "Shots on Target p90"),
        ("home_pl_sh_p90",   "away_pl_sh_p90",   "Shots p90"),
        ("home_pl_def_p90",  "away_pl_def_p90",  "Defensive Actions p90"),
        ("home_pl_gls_p90",  "away_pl_gls_p90",  "Goals p90"),
        ("home_pl_ast_p90",  "away_pl_ast_p90",  "Assists p90"),
        ("home_pl_pct_fw",   "away_pl_pct_fw",   "% Forwards in XI"),
    ]
    pairs = [(h, a, t) for h, a, t in pairs if h in df.columns and a in df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for ax, (hf, af, title) in zip(axes, pairs):
        for result in ["H", "D", "A"]:
            sub = df[df["result"] == result].dropna(subset=[hf, af])
            ax.scatter(sub[hf], sub[af], c=RESULT_PALETTE[result],
                       alpha=0.3, s=14, label=LABEL_MAP[result], linewidths=0)
        lims = [
            min(df[hf].quantile(0.01), df[af].quantile(0.01)),
            max(df[hf].quantile(0.99), df[af].quantile(0.99)),
        ]
        ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="Equal")
        ax.set_xlabel(f"Home {hf.replace('home_pl_', '')}", fontsize=9)
        ax.set_ylabel(f"Away {af.replace('away_pl_', '')}", fontsize=9)
        ax.set_title(title, fontsize=9, fontweight="bold", color="#e74c3c")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)

    handles = [Patch(color=RESULT_PALETTE[r], label=LABEL_MAP[r]) for r in ["H", "D", "A"]]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        "Player-Level Features: Home vs Away Team Comparison\n"
        "(dashed diagonal = equal; above = away team stronger)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(OUT_DIR / "11_player_home_vs_away.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 11_player_home_vs_away.png")


# ═══════════════════════════════════════════════════════════════════════════
# 13. Feature importance summary leaderboard chart
# ═══════════════════════════════════════════════════════════════════════════
def plot_feature_leaderboard(df_feat: pd.DataFrame) -> None:
    top = df_feat.head(30).copy()
    colors = [SOURCE_PALETTE.get(s, "#888") for s in top["source"]]

    fig, axes = plt.subplots(1, 2, figsize=(18, 10))

    # composite score bar
    ax = axes[0]
    ax.barh(top.index[::-1], top["composite_score"].values[::-1],
            color=colors[::-1], edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Composite Score (mean rank percentile)", fontsize=10)
    ax.set_title("Top-30 Features — Composite Score\n(4-method average)", fontsize=11, fontweight="bold")
    ax.axvline(top["composite_score"].mean(), color="grey", linestyle="--", alpha=0.7, label="Mean")
    ax.legend(handles=[
        Patch(color=SOURCE_PALETTE["player"],     label="Player-level"),
        Patch(color=SOURCE_PALETTE["fbref_team"], label="FBref team-level"),
        Patch(color=SOURCE_PALETTE["understat"],  label="Understat/Transfermarkt"),
    ], fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)

    # RF importance bar
    ax2 = axes[1]
    ax2.barh(top.index[::-1], top["rf_importance"].values[::-1],
             color=colors[::-1], edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("Random Forest Importance (MDI)", fontsize=10)
    ax2.set_title("Top-30 Features — RF Importance", fontsize=11, fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(labelsize=8)

    plt.suptitle("Feature Selection Leaderboard — All Methods", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "12_feature_leaderboard_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 12_feature_leaderboard_summary.png")


# ═══════════════════════════════════════════════════════════════════════════
# 14. Violin plots for top player-level features by result
# ═══════════════════════════════════════════════════════════════════════════
def plot_violin_player_features(df: pd.DataFrame) -> None:
    player_in_data = [f for f in PLAYER_FEATURES if f in df.columns]
    # pick net_ and home_ variants
    focus = [f for f in player_in_data if f.startswith("net_pl_") or f.startswith("home_pl_")][:8]
    if not focus:
        print("  ✗ No player features available for violin plot")
        return

    ncols = 4
    nrows = (len(focus) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4))
    axes = axes.flatten()

    for i, feat in enumerate(focus):
        ax = axes[i]
        data = [df[df["result"] == r][feat].dropna().values for r in ["H", "D", "A"]]
        vp = ax.violinplot(data, positions=[1, 2, 3], showmedians=True, showextrema=False)
        for body, result in zip(vp["bodies"], ["H", "D", "A"]):
            body.set_facecolor(RESULT_PALETTE[result])
            body.set_alpha(0.7)
        vp["cmedians"].set_color("white")
        vp["cmedians"].set_linewidth(2)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["H", "D", "A"], fontsize=9)
        ax.set_title(feat, fontsize=9, fontweight="bold", color="#e74c3c")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    handles = [Patch(color=RESULT_PALETTE[r], label=LABEL_MAP[r]) for r in ["H", "D", "A"]]
    fig.legend(handles=handles, loc="lower right", fontsize=10)
    fig.suptitle("Violin Plots — Player-Level Features by Match Outcome", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(OUT_DIR / "13_violin_player_features.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 13_violin_player_features.png")


# ═══════════════════════════════════════════════════════════════════════════
# 15. Correlation heatmap — top-15 features only
# ═══════════════════════════════════════════════════════════════════════════
def plot_top_corr_heatmap(df: pd.DataFrame, top_features: list[str]) -> None:
    cols = [f for f in top_features if f in df.columns][:15]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)

    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6.5, color="black" if abs(val) < 0.6 else "white")

    ax.set_title("Correlation Heatmap — Top-15 Features by Composite Score",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "14_top15_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 14_top15_correlation_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    print(f"\nLoading data from {DATA_PATH} ...")
    df = load()
    print(f"  {len(df)} matches, {df.shape[1]} columns, seasons: {sorted(df['season'].unique())}")

    # load leaderboard
    board = pd.read_csv(BOARD_PATH, index_col=0)
    top_features = board.sort_values("composite_score", ascending=False).index.tolist()

    print(f"\nGenerating EDA visualisations → {OUT_DIR}\n")

    plot_target_distribution(df)
    plot_season_distribution(df)
    plot_missingness(df)
    plot_histograms_by_result(df, top_features)
    plot_boxplots_by_result(df, top_features)
    plot_bookmaker_vs_xg(df)
    plot_value_vs_xg(df)
    plot_player_vs_bookmaker(df)
    plot_scatter_matrix(df, top_features)
    plot_feature_trends(df)
    plot_home_vs_away_player(df)
    plot_feature_leaderboard(board)
    plot_violin_player_features(df)
    plot_top_corr_heatmap(df, top_features)

    files = sorted(OUT_DIR.glob("*.png"))
    print(f"\n{'='*55}")
    print(f"  Done — {len(files)} plots saved to outputs/eda/")
    print(f"{'='*55}")
    for f in files:
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
