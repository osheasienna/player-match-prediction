
"""Engineer FBref features and merge with the existing Understat feature set.

New signals vs. Understat-only pipeline
----------------------------------------
  Possession (Poss)       — rolling 5-match mean per team  [team match schedule]
  Formation               — number of attackers in lineup  [team match schedule]
  Player quality (G+A/90) — aggregated season stats        [player season stats]
  Squad depth             — starter vs bench quality gap   [player season stats]

Season alignment
----------------
soccerdata uses compact codes (1819, 1920, 2021, 2223, 2324).
Understat uses start-year codes (2018, 2019, 2020, 2021, 2022, 2023).
The 2021/22 season (understat=2021, fbref=2122) is missing from the FBref pull.

Player quality features use the *previous* season's stats to avoid leakage.
Team rolling features use shift(1).rolling(5) — same guard as the Understat pipeline.

Output: data/ligue1_fbref_features.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).parent.parent / "raw"
DATA_DIR = Path(__file__).parent.parent / "data"
OUT_PATH = DATA_DIR / "ligue1_fbref_features.csv"

WINDOW = 5

# Map soccerdata compact season → Understat season (start year)
FBREF_TO_UNDERSTAT_SEASON = {
    1819: 2018,
    1920: 2019,
    2021: 2020,   # soccerdata's "2021" = 2020/21
    2122: 2021,   # missing from our pull
    2223: 2022,
    2324: 2023,
}

# For previous-season player stats lookup: understat season → fbref season to use
PREV_SEASON_MAP = {
    2018: None,   # no prior data
    2019: 1819,
    2020: 1920,
    2021: 2021,   # use 2020/21 stats (fbref code 2021)
    2022: 2021,   # 2122 is missing; fall back to 2021
    2023: 2223,
}


# ---------------------------------------------------------------------------
# 1. Load raw FBref files
# ---------------------------------------------------------------------------

def load_fbref_team_match() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "fbref_team_match.csv", parse_dates=["date"])
    df["understat_season"] = df["season"].map(FBREF_TO_UNDERSTAT_SEASON)
    df = df.dropna(subset=["understat_season"])
    df["understat_season"] = df["understat_season"].astype(int)
    # Keep only Ligue 1 matches (schedule includes all comps)
    df = df[df["league"] == "FRA-Ligue 1"].copy()
    # Normalise result to W/D/L
    df["won"] = (df["result"] == "W").astype(int)
    df["drew"] = (df["result"] == "D").astype(int)
    df["points"] = df["won"] * 3 + df["drew"]
    # Parse possession
    df["Poss"] = pd.to_numeric(df["Poss"], errors="coerce")
    return df


def load_fbref_player_season() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "fbref_player_season.csv")
    # Drop aggregate rows (e.g. '2 Clubs' rows with NaN player names)
    df = df.dropna(subset=["player"])
    df = df[~df["player"].str.contains("Players", na=False)]
    # Numeric coercion
    for col in ["Playing Time_Starts", "Playing Time_Min", "Playing Time_90s",
                "Per 90 Minutes_Gls", "Per 90 Minutes_Ast", "Per 90 Minutes_G+A"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Per 90 Minutes_G+A", "Playing Time_90s"])
    return df


# ---------------------------------------------------------------------------
# 2. Team possession + rolling features from match schedule
# ---------------------------------------------------------------------------

def team_rolling_poss(team_df: pd.DataFrame) -> pd.DataFrame:
    """Rolling 5-match possession and goal stats per team (lag-safe)."""
    df = team_df.sort_values(["team", "date"]).copy()
    for col, src in [("roll_poss", "Poss"), ("roll_gf", "GF"),
                     ("roll_ga", "GA"), ("roll_won", "won"), ("roll_pts", "points")]:
        df[col] = (
            df.groupby("team")[src]
            .transform(lambda s: s.shift(1).rolling(WINDOW, min_periods=1).mean())
        )
    return df


def extract_attackers(formation: str | float) -> float:
    """Return the number of forwards from a formation string ('4-3-3' → 3)."""
    if pd.isna(formation) or not isinstance(formation, str):
        return np.nan
    parts = formation.strip().split("-")
    try:
        return float(parts[-1])
    except (IndexError, ValueError):
        return np.nan


# ---------------------------------------------------------------------------
# 3. Player season quality features aggregated to team level
# ---------------------------------------------------------------------------

def build_player_quality(player_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-player season stats to team level.

    Returns one row per (fbref_season, team) with:
      team_ga_per90        — mean G+A/90 across all squad members
      team_ga_per90_start  — mean G+A/90 for regular starters (≥5 starts)
      team_ga_per90_bench  — mean G+A/90 for non-starters (<5 starts)
      squad_depth          — starter quality minus bench quality
      squad_minutes_top11  — share of minutes held by top-11 minute players
    """
    records = []
    for (season, team), grp in player_df.groupby(["season", "team"]):
        if len(grp) < 3:
            continue
        ga90 = grp["Per 90 Minutes_G+A"].values
        starts = grp["Playing Time_Starts"].fillna(0).values
        mins = grp["Playing Time_Min"].fillna(0).values

        starters_mask = starts >= 5
        bench_mask = ~starters_mask

        starter_qa = ga90[starters_mask].mean() if starters_mask.any() else np.nan
        bench_qa   = ga90[bench_mask].mean()    if bench_mask.any()    else np.nan

        # Squad depth: gap between starter and bench quality
        depth = (starter_qa - bench_qa) if not (np.isnan(starter_qa) or np.isnan(bench_qa)) else np.nan

        # Minutes concentration: share of total minutes for top-11 by minutes
        if mins.sum() > 0:
            top11_mins = np.sort(mins)[::-1][:11].sum()
            conc = top11_mins / mins.sum()
        else:
            conc = np.nan

        records.append({
            "season": season,
            "team": team,
            "team_ga_per90": ga90.mean(),
            "team_ga_per90_start": starter_qa,
            "team_ga_per90_bench": bench_qa,
            "squad_depth": depth,
            "squad_minutes_top11": conc,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 4. Build match-level feature matrix
# ---------------------------------------------------------------------------

def build_match_features(
    understat: pd.DataFrame,
    team_rolled: pd.DataFrame,
    player_quality: pd.DataFrame,
) -> pd.DataFrame:
    """Join all FBref features onto the Understat match frame."""

    # ---- 4a. Pivot team rolling stats to match level ----
    # team_rolled has one row per (team, match); split into home/away
    home_rolled = team_rolled[team_rolled["venue"] == "Home"][
        ["date", "team", "roll_poss", "roll_gf", "roll_ga", "roll_won", "roll_pts", "Formation"]
    ].rename(columns={
        "team": "home_team_fbref",
        "roll_poss":  "home_roll_poss",
        "roll_gf":    "home_roll_gf_fbref",
        "roll_ga":    "home_roll_ga_fbref",
        "roll_won":   "home_roll_won_fbref",
        "roll_pts":   "home_roll_pts_fbref",
        "Formation":  "home_formation",
    })
    away_rolled = team_rolled[team_rolled["venue"] == "Away"][
        ["date", "team", "roll_poss", "roll_gf", "roll_ga", "roll_won", "roll_pts", "Formation"]
    ].rename(columns={
        "team": "away_team_fbref",
        "roll_poss":  "away_roll_poss",
        "roll_gf":    "away_roll_gf_fbref",
        "roll_ga":    "away_roll_ga_fbref",
        "roll_won":   "away_roll_won_fbref",
        "roll_pts":   "away_roll_pts_fbref",
        "Formation":  "away_formation",
    })

    # Join home and away on (date, home_team, away_team)
    # FBref team names match Understat names so we join directly
    df = understat.merge(
        home_rolled.rename(columns={"home_team_fbref": "home_team"}),
        on=["date", "home_team"], how="left"
    ).merge(
        away_rolled.rename(columns={"away_team_fbref": "away_team"}),
        on=["date", "away_team"], how="left"
    )

    # ---- 4b. Formation encoding ----
    df["home_attackers"] = df["home_formation"].apply(extract_attackers)
    df["away_attackers"] = df["away_formation"].apply(extract_attackers)
    df["net_attackers"]  = df["home_attackers"] - df["away_attackers"]
    df["net_poss"]       = df["home_roll_poss"] - df["away_roll_poss"]

    # ---- 4c. Player quality features (previous season) ----
    # Build a lookup: (understat_season, team) → quality metrics
    quality_expanded = []
    for _, row in player_quality.iterrows():
        fbref_s = row["season"]
        # Map fbref season → which understat season this data applies to
        # (as "previous season" quality)
        for us, ps in PREV_SEASON_MAP.items():
            if ps == fbref_s:
                quality_expanded.append({**row.to_dict(), "understat_season": us})
    quality_df = pd.DataFrame(quality_expanded) if quality_expanded else pd.DataFrame()

    if not quality_df.empty:
        # Drop fbref season code; keep only the understat_season join key
        qdf = quality_df.drop(columns=["season"]).rename(columns={"understat_season": "season"})

        home_q = (
            qdf
            .rename(columns={
                "team": "home_team",
                "team_ga_per90":       "home_player_ga90",
                "team_ga_per90_start": "home_starter_ga90",
                "squad_depth":         "home_squad_depth",
                "squad_minutes_top11": "home_minutes_conc",
            })[["home_team", "season", "home_player_ga90", "home_starter_ga90",
                "home_squad_depth", "home_minutes_conc"]]
            .drop_duplicates(subset=["home_team", "season"])
        )
        away_q = (
            qdf
            .rename(columns={
                "team": "away_team",
                "team_ga_per90":       "away_player_ga90",
                "team_ga_per90_start": "away_starter_ga90",
                "squad_depth":         "away_squad_depth",
                "squad_minutes_top11": "away_minutes_conc",
            })[["away_team", "season", "away_player_ga90", "away_starter_ga90",
                "away_squad_depth", "away_minutes_conc"]]
            .drop_duplicates(subset=["away_team", "season"])
        )

        df = df.merge(home_q, on=["home_team", "season"], how="left")
        df = df.merge(away_q, on=["away_team", "season"], how="left")

        df["net_player_ga90"]  = df["home_player_ga90"]  - df["away_player_ga90"]
        df["net_starter_ga90"] = df["home_starter_ga90"] - df["away_starter_ga90"]
        df["net_squad_depth"]  = df["home_squad_depth"]  - df["away_squad_depth"]

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading FBref team match stats ...")
    team_match = load_fbref_team_match()
    print(f"  {len(team_match)} team-match rows, {team_match['understat_season'].nunique()} seasons")

    print("Computing rolling possession/form features ...")
    team_rolled = team_rolling_poss(team_match)

    print("Loading FBref player season stats ...")
    player_season = load_fbref_player_season()
    print(f"  {len(player_season)} player-season rows")

    print("Building player quality aggregates ...")
    player_quality = build_player_quality(player_season)
    print(f"  {len(player_quality)} team-season quality rows")

    print("Loading Understat match frame ...")
    understat = pd.read_csv(DATA_DIR / "ligue1_features.csv", parse_dates=["date"])
    print(f"  {len(understat)} matches")

    print("Joining FBref features onto match frame ...")
    merged = build_match_features(understat, team_rolled, player_quality)

    # Select final columns — FBref-only new features on top of existing ones
    new_fbref_cols = [
        "home_roll_poss", "away_roll_poss", "net_poss",
        "home_attackers", "away_attackers", "net_attackers",
        "home_roll_gf_fbref", "home_roll_ga_fbref",
        "away_roll_gf_fbref", "away_roll_ga_fbref",
        "home_player_ga90", "away_player_ga90", "net_player_ga90",
        "home_starter_ga90", "away_starter_ga90", "net_starter_ga90",
        "home_squad_depth", "away_squad_depth", "net_squad_depth",
        "home_minutes_conc", "away_minutes_conc",
    ]
    existing_cols = [c for c in understat.columns if c != "result_code"]
    all_cols = existing_cols + [c for c in new_fbref_cols if c in merged.columns] + ["result_code"]

    out = merged[[c for c in all_cols if c in merged.columns]]
    out = out.sort_values(["date", "match_id"]).reset_index(drop=True)
    out.to_csv(OUT_PATH, index=False)

    new_present = [c for c in new_fbref_cols if c in out.columns]
    print(f"\nSaved {len(out)} rows × {len(out.columns)} columns → {OUT_PATH}")
    print(f"New FBref features added: {len(new_present)}")
    print("\nMissing value counts for new FBref features:")
    print(out[new_present].isnull().sum().to_string())


if __name__ == "__main__":
    main()
