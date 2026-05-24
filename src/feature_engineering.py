"""Merge Understat + Transfermarkt and engineer all match-level features.

Output: data/ligue1_features.csv  (one row per match, home-team perspective)

Feature groups
--------------
Rolling stats (5-match window, lag-safe via shift(1)):
  home/away rolling xG for/against, goals for/against, win rate, points/game

Market value (Transfermarkt):
  home/away squad total value, ratio, log-ratio

Pre-match forecast (Understat model):
  home_win_prob, draw_prob, away_win_prob

Context:
  season, match_week, date

Target:
  result  (H / D / A)
  result_code  (0=H, 1=D, 2=A)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).parent.parent / "raw"
DATA_DIR = Path(__file__).parent.parent / "data"
OUT_PATH = DATA_DIR / "ligue1_features.csv"

WINDOW = 5

# Understat team names -> canonical names matching Transfermarkt reference table
UNDERSTAT_TO_CANONICAL: dict[str, str] = {
    # Exact Understat team name -> Transfermarkt club_name
    "Paris Saint Germain": "Paris Saint-Germain",
    "Lyon": "Olympique Lyonnais",
    "Monaco": "AS Monaco",
    "Marseille": "Olympique de Marseille",
    "Lille": "LOSC Lille",
    "Rennes": "Stade Rennais FC",
    "Nice": "OGC Nice",
    "Lens": "RC Lens",
    "Reims": "Stade de Reims",
    "Toulouse": "Toulouse FC",
    "Montpellier": "Montpellier HSC",
    "Nantes": "Nantes",
    "Strasbourg": "Strasbourg",
    "Bordeaux": "Girondins Bordeaux",
    "Saint-Etienne": "Saint-Étienne",
    "Brest": "Brest",
    "Angers": "Angers SCO",
    "Metz": "Metz",
    "Lorient": "Lorient",
    "Dijon": "Dijon FCO",
    "Nimes": "Nîmes Olympique",
    "Ajaccio": "Ajaccio",
    "Troyes": "Troyes",
    "Clermont Foot": "Clermont Foot",
    "Auxerre": "Auxerre",
    "Le Havre": "Le Havre",
    "Amiens": "Amiens SC",
    "Caen": "Stade Malherbe Caen",
    "Guingamp": "En Avant Guingamp",
}


# ---------------------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------------------

def load_understat() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "understat_ligue1_raw.csv", parse_dates=["date"])
    df["home_team_canonical"] = df["home_team"].map(UNDERSTAT_TO_CANONICAL).fillna(df["home_team"])
    df["away_team_canonical"] = df["away_team"].map(UNDERSTAT_TO_CANONICAL).fillna(df["away_team"])
    return df


def load_transfermarkt() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "transfermarkt_squads.csv")
    return df[["season", "club_name", "total_market_value_eur", "avg_player_value_eur", "mean_age"]]


# ---------------------------------------------------------------------------
# 2. Build team-level long format for rolling features
# ---------------------------------------------------------------------------

def to_long_format(matches: pd.DataFrame) -> pd.DataFrame:
    """Pivot each match into two rows (one per team) for rolling computation."""
    home = matches[["date", "season", "match_id", "home_team_canonical",
                    "home_goals", "away_goals", "home_xg", "away_xg", "result"]].copy()
    home = home.rename(columns={
        "home_team_canonical": "team",
        "home_goals": "goals_for",
        "away_goals": "goals_against",
        "home_xg": "xg_for",
        "away_xg": "xg_against",
    })
    home["is_home"] = 1
    home["won"] = (home["result"] == "H").astype(int)
    home["drew"] = (home["result"] == "D").astype(int)
    home["points"] = home["won"] * 3 + home["drew"]

    away = matches[["date", "season", "match_id", "away_team_canonical",
                    "home_goals", "away_goals", "home_xg", "away_xg", "result"]].copy()
    away = away.rename(columns={
        "away_team_canonical": "team",
        "away_goals": "goals_for",
        "home_goals": "goals_against",
        "away_xg": "xg_for",
        "home_xg": "xg_against",
    })
    away["is_home"] = 0
    away["won"] = (away["result"] == "A").astype(int)
    away["drew"] = (away["result"] == "D").astype(int)
    away["points"] = away["won"] * 3 + away["drew"]

    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(["team", "date", "match_id"]).reset_index(drop=True)
    return long


# ---------------------------------------------------------------------------
# 3. Compute rolling features (lag-safe)
# ---------------------------------------------------------------------------

def add_rolling_features(long: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Add rolling stats over the prior `window` matches for each team.

    Uses .shift(1) before .rolling() to exclude the current match (leakage guard).
    """
    cols = ["goals_for", "goals_against", "xg_for", "xg_against", "won", "points"]

    def team_rolling(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("date").copy()
        for col in cols:
            g[f"roll_{col}"] = (
                g[col].shift(1).rolling(window, min_periods=1).mean()
            )
        g["roll_xg_diff"] = g["roll_xg_for"] - g["roll_xg_against"]
        g["roll_goal_diff"] = g["roll_goals_for"] - g["roll_goals_against"]
        return g

    result = long.groupby("team", group_keys=False).apply(team_rolling)
    return result


# ---------------------------------------------------------------------------
# 4. Join rolling features back to match level
# ---------------------------------------------------------------------------

def build_match_features(matches: pd.DataFrame, long_with_roll: pd.DataFrame) -> pd.DataFrame:
    roll_cols = [c for c in long_with_roll.columns if c.startswith("roll_")]
    roll_home = (
        long_with_roll[long_with_roll["is_home"] == 1]
        [["match_id"] + roll_cols]
        .rename(columns={c: f"home_{c}" for c in roll_cols})
    )
    roll_away = (
        long_with_roll[long_with_roll["is_home"] == 0]
        [["match_id"] + roll_cols]
        .rename(columns={c: f"away_{c}" for c in roll_cols})
    )

    df = matches.merge(roll_home, on="match_id", how="left")
    df = df.merge(roll_away, on="match_id", how="left")

    # Net differentials (home minus away perspective)
    df["net_xg_diff"] = df["home_roll_xg_diff"] - df["away_roll_xg_diff"]
    df["net_goal_diff"] = df["home_roll_goal_diff"] - df["away_roll_goal_diff"]
    df["net_win_rate"] = df["home_roll_won"] - df["away_roll_won"]
    df["net_points"] = df["home_roll_points"] - df["away_roll_points"]
    return df


# ---------------------------------------------------------------------------
# 5. Add market value features
# ---------------------------------------------------------------------------

def add_market_values(df: pd.DataFrame, tm: pd.DataFrame) -> pd.DataFrame:
    home_val = tm.rename(columns={
        "club_name": "home_team_canonical",
        "total_market_value_eur": "home_squad_value",
        "avg_player_value_eur": "home_avg_player_value",
        "mean_age": "home_squad_age",
    })
    away_val = tm.rename(columns={
        "club_name": "away_team_canonical",
        "total_market_value_eur": "away_squad_value",
        "avg_player_value_eur": "away_avg_player_value",
        "mean_age": "away_squad_age",
    })

    df = df.merge(home_val, on=["season", "home_team_canonical"], how="left")
    df = df.merge(away_val, on=["season", "away_team_canonical"], how="left")

    df["squad_value_ratio"] = df["home_squad_value"] / df["away_squad_value"].replace(0, np.nan)
    df["log_squad_value_ratio"] = np.log(df["squad_value_ratio"].clip(lower=1e-3))
    df["squad_value_diff"] = df["home_squad_value"] - df["away_squad_value"]
    return df


# ---------------------------------------------------------------------------
# 6. Add match-week number (within each season)
# ---------------------------------------------------------------------------

def add_match_week(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["season", "date"]).copy()
    df["match_week"] = df.groupby("season")["date"].transform(
        lambda s: s.rank(method="dense").astype(int)
    )
    # Normalise to approx gameweek (38 GW per season)
    df["match_week"] = ((df["match_week"] - 1) // 2 + 1).clip(upper=38)
    return df


# ---------------------------------------------------------------------------
# 7. Encode target
# ---------------------------------------------------------------------------

def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {"H": 0, "D": 1, "A": 2}
    df["result_code"] = df["result"].map(mapping)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw data ...")
    matches = load_understat()
    tm = load_transfermarkt()
    print(f"  Understat: {len(matches)} matches, {matches['season'].nunique()} seasons")
    print(f"  Transfermarkt: {len(tm)} club-season rows")

    print("Converting to long format ...")
    long = to_long_format(matches)

    print(f"Computing rolling {WINDOW}-match features (lag-safe) ...")
    long_rolled = add_rolling_features(long)

    print("Joining rolling features to match level ...")
    df = build_match_features(matches, long_rolled)

    print("Adding market value features ...")
    df = add_market_values(df, tm)

    print("Adding match week and target encoding ...")
    df = add_match_week(df)
    df = encode_target(df)

    # Select and order final columns
    feature_cols = [
        # IDs / context
        "match_id", "date", "season", "match_week",
        "home_team", "away_team",
        # Rolling xG features
        "home_roll_xg_for", "home_roll_xg_against", "home_roll_xg_diff",
        "away_roll_xg_for", "away_roll_xg_against", "away_roll_xg_diff",
        "net_xg_diff",
        # Rolling goal features
        "home_roll_goals_for", "home_roll_goals_against", "home_roll_goal_diff",
        "away_roll_goals_for", "away_roll_goals_against", "away_roll_goal_diff",
        "net_goal_diff",
        # Rolling form
        "home_roll_won", "away_roll_won", "net_win_rate",
        "home_roll_points", "away_roll_points", "net_points",
        # Market value features
        "home_squad_value", "away_squad_value",
        "squad_value_ratio", "log_squad_value_ratio", "squad_value_diff",
        "home_avg_player_value", "away_avg_player_value",
        "home_squad_age", "away_squad_age",
        # Understat pre-match forecasts
        "home_win_prob", "draw_prob", "away_win_prob",
        # Target
        "result", "result_code",
    ]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"WARNING: missing columns {missing}")

    final_cols = [c for c in feature_cols if c in df.columns]
    out = df[final_cols].sort_values(["date", "match_id"]).reset_index(drop=True)

    # Drop matches with no rolling data at all (very first match of each team)
    roll_feature_cols = [c for c in final_cols if c.startswith(("home_roll_", "away_roll_", "net_"))]
    out = out.dropna(subset=roll_feature_cols[:1])  # keep if at least some history exists

    out.to_csv(OUT_PATH, index=False)

    print(f"\nSaved {len(out)} rows × {len(final_cols)} columns to {OUT_PATH}")
    print("\nMissing-value summary (features only):")
    feature_only = [c for c in roll_feature_cols + ["squad_value_ratio", "home_win_prob"]
                    if c in out.columns]
    print(out[feature_only].isnull().sum().to_string())
    print("\nTarget distribution:")
    print(out["result"].value_counts().to_string())


if __name__ == "__main__":
    main()
