"""Build match-level features from player-level FBref match stats.

Pipeline
--------
1. Load raw/fbref_player_match.csv  — one row per player per match appearance
2. Per player, compute rolling 5-match stats (shift(1) lag guard — never leaks)
3. Per match side (home / away), aggregate the starting XI's individual rolling
   stats into a single team-level feature, weighted by minutes played
4. Merge onto the Understat match frame and save data/ligue1_player_features.csv

Player-level stats available (from FBref match report "summary" table)
-----------------------------------------------------------------------
  Min          — minutes played (used as weight)
  Pos          — position (FW / MF / DF / GK)
  Gls, Ast     — goals, assists
  Sh, SoT      — shots, shots on target
  TklW, Int    — tackles won, interceptions  (defensive)
  Fls, Fld     — fouls committed, fouls drawn
  CrdY, CrdR   — yellow/red cards

Per-player rolling stats (over prior 5 appearances, any team):
  roll_sh_p90   — shots per 90 min
  roll_sot_p90  — shots on target per 90 min
  roll_def_p90  — (TklW + Int) per 90 min
  roll_gls_p90  — goals per 90 min
  roll_ast_p90  — assists per 90 min
  roll_fls_p90  — fouls committed per 90 min
  roll_min      — mean minutes played (proxy for fitness/selection regularity)

Match-level aggregate features (weighted mean across starting XI, weight = Min):
  home_sh_p90        away_sh_p90        net_sh_p90
  home_sot_p90       away_sot_p90       net_sot_p90
  home_def_p90       away_def_p90       net_def_p90
  home_gls_p90       away_gls_p90       net_gls_p90
  home_ast_p90       away_ast_p90       net_ast_p90
  home_pct_fw        away_pct_fw        — fraction of starters who are forwards
  home_starter_mins  away_starter_mins  — mean minutes of starting XI (fitness proxy)
  home_depth_sh      away_depth_sh      — shot gap between top-6 and bottom-5 starters
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR  = Path(__file__).parent.parent / "raw"
DATA_DIR = Path(__file__).parent.parent / "data"
OUT_PATH = DATA_DIR / "ligue1_player_features.csv"

WINDOW     = 5
STARTER_MIN = 20   # min minutes to count as a "participant" in a match


# ---------------------------------------------------------------------------
# 1. Load + clean player match data
# ---------------------------------------------------------------------------

def load_player_match() -> pd.DataFrame:
    path = RAW_DIR / "fbref_player_match.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run src/collect_fbref.py first "
            "(fetch_player_match_stats must complete)"
        )

    df = pd.read_csv(path)

    # Normalise column names — soccerdata may produce mixed cases
    df.columns = [c.strip() for c in df.columns]
    col_map = {c: c.lower() for c in df.columns}
    # Keep canonical names for stat columns
    for orig in df.columns:
        low = orig.lower()
        if "performance" in low:
            stat = orig.split("_")[-1] if "_" in orig else orig
            col_map[orig] = f"perf_{stat.lower()}"
    df = df.rename(columns=col_map)

    # Parse date — soccerdata embeds it in the 'game' index field or as a column
    if "date" not in df.columns:
        # Try to extract from 'game' string: "2022-08-05 Lyon vs Ajaccio"
        if "game" in df.columns:
            df["date"] = pd.to_datetime(
                df["game"].str.extract(r"(\d{4}-\d{2}-\d{2})")[0], errors="coerce"
            )
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.dropna(subset=["date", "player", "team"])

    # Minutes to numeric
    df["min"] = pd.to_numeric(df.get("min", df.get("perf_min", np.nan)), errors="coerce").fillna(0)

    # Core stat columns — use 0 if column is missing (older seasons lack some)
    stat_cols = {
        "perf_gls": "gls", "perf_ast": "ast",
        "perf_sh":  "sh",  "perf_sot": "sot",
        "perf_tklw":"tklw","perf_int": "int_",
        "perf_fls": "fls",
    }
    for src, dst in stat_cols.items():
        if src in df.columns:
            df[dst] = pd.to_numeric(df[src], errors="coerce").fillna(0)
        else:
            df[dst] = 0.0

    df["def_actions"] = df["tklw"] + df["int_"]

    # Season map: soccerdata compact code → Understat start-year
    fbref_to_us = {1819:2018, 1920:2019, 2021:2020, 2122:2021, 2223:2022, 2324:2023}
    if "season" in df.columns:
        df["us_season"] = pd.to_numeric(df["season"], errors="coerce").map(fbref_to_us)

    # Position: extract primary position (first token before comma)
    if "pos" in df.columns:
        df["pos_primary"] = df["pos"].str.split(",").str[0].str.strip().fillna("MF")
    else:
        df["pos_primary"] = "MF"

    return df.sort_values(["player", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Per-player rolling stats (lag-safe)
# ---------------------------------------------------------------------------

def per_90(series: pd.Series, minutes: pd.Series) -> pd.Series:
    """Convert a counting stat to per-90-minute rate, dividing by minutes/90."""
    mins = minutes.clip(lower=1)
    return series / (mins / 90)


def compute_player_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling 5-match per-90 stats for each player.

    Uses shift(1) so the current match is never included.
    Players are tracked across team changes — rolling form is individual.
    """
    df = df.sort_values(["player", "date"]).copy()

    # Per-90 rates for the current match (used as input to rolling)
    df["sh_p90"]  = per_90(df["sh"],          df["min"])
    df["sot_p90"] = per_90(df["sot"],         df["min"])
    df["def_p90"] = per_90(df["def_actions"], df["min"])
    df["gls_p90"] = per_90(df["gls"],         df["min"])
    df["ast_p90"] = per_90(df["ast"],         df["min"])
    df["fls_p90"] = per_90(df["fls"],         df["min"])

    roll_cols = ["sh_p90", "sot_p90", "def_p90", "gls_p90", "ast_p90", "fls_p90", "min"]
    for col in roll_cols:
        df[f"roll_{col}"] = (
            df.groupby("player")[col]
            .transform(lambda s: s.shift(1).rolling(WINDOW, min_periods=1).mean())
        )

    return df


# ---------------------------------------------------------------------------
# 3. Aggregate starting XI rolling stats to match level
# ---------------------------------------------------------------------------

def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted mean; returns NaN if total weight is zero."""
    total = weights.sum()
    return float(np.dot(values, weights) / total) if total > 0 else np.nan


def aggregate_side(group: pd.DataFrame) -> dict:
    """Aggregate one team's players in one match to a single feature vector.

    `group` is all rows for one (match, team) — starters + substitutes.
    """
    # Starters = players with >= STARTER_MIN minutes
    starters = group[group["min"] >= STARTER_MIN].copy()
    if starters.empty:
        starters = group.copy()  # fallback: use everyone

    mins = starters["min"].values
    roll_sh  = starters["roll_sh_p90"].values
    roll_sot = starters["roll_sot_p90"].values
    roll_def = starters["roll_def_p90"].values
    roll_gls = starters["roll_gls_p90"].values
    roll_ast = starters["roll_ast_p90"].values
    roll_min = starters["roll_min"].values

    # Fraction of starters who are forwards
    pct_fw = (starters["pos_primary"] == "FW").mean()

    # Squad depth: shot difference between top-half and bottom-half starters by roll_sh
    sorted_sh = np.sort(roll_sh)[::-1]
    mid = max(len(sorted_sh) // 2, 1)
    depth_sh = sorted_sh[:mid].mean() - sorted_sh[mid:].mean() if len(sorted_sh) > 1 else 0.0

    return {
        "sh_p90":       weighted_mean(roll_sh,  mins),
        "sot_p90":      weighted_mean(roll_sot, mins),
        "def_p90":      weighted_mean(roll_def, mins),
        "gls_p90":      weighted_mean(roll_gls, mins),
        "ast_p90":      weighted_mean(roll_ast, mins),
        "starter_mins": roll_min.mean(),   # mean of player rolling-min (fitness proxy)
        "pct_fw":       pct_fw,
        "depth_sh":     depth_sh,
        "n_players":    len(group),
    }


def build_match_side_features(player_df: pd.DataFrame) -> pd.DataFrame:
    """Build one row per (date, team) with aggregated player rolling stats."""
    records = []
    for (date, team), grp in player_df.groupby(["date", "team"]):
        feats = aggregate_side(grp)
        feats["date"] = date
        feats["team"] = team
        records.append(feats)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 4. Join onto match frame
# ---------------------------------------------------------------------------

def join_to_matches(match_df: pd.DataFrame, side_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot home/away side features onto the match frame."""
    home_side = side_df.rename(columns={
        c: f"home_pl_{c}" for c in side_df.columns if c not in ("date", "team")
    }).rename(columns={"team": "home_team"})

    away_side = side_df.rename(columns={
        c: f"away_pl_{c}" for c in side_df.columns if c not in ("date", "team")
    }).rename(columns={"team": "away_team"})

    df = match_df.merge(home_side, on=["date", "home_team"], how="left")
    df = df.merge(away_side, on=["date", "away_team"], how="left")

    # Net (home − away) differentials
    stat_stems = ["sh_p90", "sot_p90", "def_p90", "gls_p90", "ast_p90",
                  "starter_mins", "pct_fw", "depth_sh"]
    for stem in stat_stems:
        h = f"home_pl_{stem}"
        a = f"away_pl_{stem}"
        if h in df.columns and a in df.columns:
            df[f"net_pl_{stem}"] = df[h] - df[a]

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading player match stats ...")
    players = load_player_match()
    print(f"  {len(players):,} player-match rows, "
          f"{players['player'].nunique():,} unique players, "
          f"{players['date'].nunique():,} match dates")

    print(f"Computing per-player rolling {WINDOW}-match stats (lag-safe) ...")
    players = compute_player_rolling(players)

    print("Aggregating starting XI stats to match-side level ...")
    side_df = build_match_side_features(players)
    print(f"  {len(side_df):,} (date, team) rows")

    print("Loading full match frame (Understat + Transfermarkt + FBref team-level) ...")
    fbref_path = DATA_DIR / "ligue1_fbref_features.csv"
    base_path  = DATA_DIR / "ligue1_features.csv"
    matches = pd.read_csv(
        fbref_path if fbref_path.exists() else base_path,
        parse_dates=["date"],
    )
    print(f"  {len(matches):,} matches")

    print("Joining player features onto match frame ...")
    out = join_to_matches(matches, side_df)

    # Report coverage
    pl_cols = [c for c in out.columns if c.startswith(("home_pl_", "away_pl_", "net_pl_"))]
    print(f"\n  Player feature columns added: {len(pl_cols)}")
    print("  Missing value counts:")
    print(out[pl_cols].isnull().sum().to_string())

    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(out):,} rows × {len(out.columns)} columns → {OUT_PATH}")


if __name__ == "__main__":
    main()
