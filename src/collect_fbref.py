"""Fetch FBref data for Ligue 1 seasons 2018-2023 via soccerdata.

Two datasets:
  1. raw/fbref_team_match.csv   — team-level stats per match (one row per team per match)
     Columns: xG, xGA, Poss, Sh, SoT, progressive passes, pressures, defensive actions
     Source: read_team_match_stats(stat_type='summary')
     Speed: ~5 min (20 teams × 6 seasons = 120 pages, cached after first run)

  2. raw/fbref_player_season.csv — player season totals (one row per player per season)
     Columns: Min, Gls, Ast, xG, xAG, npxG, progressive passes, pressures, tackles, etc.
     Source: read_player_season_stats(stat_type='summary')
     Speed: ~1 min (one league page per season × 6 seasons)

Both files are written even if one stat type is unavailable (graceful skip).
soccerdata caches all HTML to ~/soccerdata/data/ so re-runs are instant.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import soccerdata as sd

warnings.filterwarnings("ignore")

LEAGUE = "FRA-Ligue 1"
SEASONS = list(range(2018, 2024))
RAW_DIR = Path(__file__).parent.parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse MultiIndex columns to single-level 'Group_Stat' strings."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            f"{a}_{b}".strip("_") if a and not a.startswith("Unnamed") else b
            for a, b in df.columns
        ]
    return df


def fetch_team_match_stats() -> None:
    print("\n[1] Fetching team match stats (summary) ...")
    fbref = sd.FBref(leagues=LEAGUE, seasons=SEASONS)
    try:
        df = fbref.read_team_match_stats(stat_type="schedule")
        df = flatten_columns(df.reset_index())
        out = RAW_DIR / "fbref_team_match.csv"
        df.to_csv(out, index=False)
        print(f"    Shape: {df.shape}")
        print(f"    Columns: {df.columns.tolist()[:20]}")
        print(f"    Saved → {out}")
    except Exception as e:
        print(f"    ERROR: {e}")


def fetch_player_season_stats() -> None:
    print("\n[2] Fetching player season stats (summary) ...")
    fbref = sd.FBref(leagues=LEAGUE, seasons=SEASONS)
    try:
        df = fbref.read_player_season_stats(stat_type="standard")
        df = flatten_columns(df.reset_index())
        out = RAW_DIR / "fbref_player_season.csv"
        df.to_csv(out, index=False)
        print(f"    Shape: {df.shape}")
        print(f"    Columns: {df.columns.tolist()[:20]}")
        print(f"    Saved → {out}")
    except Exception as e:
        print(f"    ERROR: {e}")


def fetch_player_match_stats() -> None:
    """Per-player per-match stats.

    Fetches one game at a time so individual download failures are skipped
    rather than aborting the whole run. Saves a checkpoint pickle every 200
    games so progress survives restarts.
    """
    print("\n[3] Fetching player match stats (summary) ...")
    fbref = sd.FBref(leagues=LEAGUE, seasons=SEASONS)
    out = RAW_DIR / "fbref_player_match.csv"
    checkpoint = RAW_DIR / "fbref_player_match_checkpoint.pkl"

    # Get full game-ID list from cached schedule
    try:
        schedule = fbref.read_schedule(force_cache=True).reset_index()
    except Exception:
        schedule = fbref.read_schedule().reset_index()

    id_col = "game_id" if "game_id" in schedule.columns else schedule.index.name
    game_ids = (
        schedule[~schedule[id_col].isna()][id_col].tolist()
        if "match_report" not in schedule.columns
        else schedule[
            ~schedule[id_col].isna() & ~schedule["match_report"].isna()
        ][id_col].tolist()
    )
    print(f"  {len(game_ids)} games in schedule")

    # Resume from checkpoint if one exists
    dfs: list[pd.DataFrame] = []
    completed: set = set()
    if checkpoint.exists():
        saved = pd.read_pickle(checkpoint)
        dfs.append(saved)
        # Identify completed game IDs from the 'game' column (contains game_id substring)
        if "game_id" in saved.columns:
            completed = set(saved["game_id"].dropna().unique())
        print(f"  Resuming from checkpoint: {len(dfs[0])} rows already saved")

    remaining = [gid for gid in game_ids if gid not in completed]
    print(f"  {len(remaining)} games to fetch ({len(completed)} already done)")

    errors = 0
    for i, gid in enumerate(remaining):
        try:
            df = fbref.read_player_match_stats(stat_type="summary", match_id=gid)
            dfs.append(flatten_columns(df.reset_index()))
        except Exception as e:
            errors += 1
            if errors <= 20:
                print(f"    Skip {gid}: {e}")
            elif errors == 21:
                print("    (suppressing further skip messages ...)")
            continue

        if (i + 1) % 200 == 0:
            combined = pd.concat(dfs, ignore_index=True)
            combined.to_pickle(checkpoint)
            pct = 100 * (i + 1) / len(remaining)
            print(f"    [{i+1}/{len(remaining)}] {pct:.0f}% — checkpoint saved, {errors} errors so far")

    if not dfs:
        print("    ERROR: No data collected — all games failed")
        return

    result = pd.concat(dfs, ignore_index=True)
    result.to_csv(out, index=False)
    if checkpoint.exists():
        checkpoint.unlink()

    print(f"    Shape: {result.shape}")
    print(f"    Columns: {result.columns.tolist()[:20]}")
    print(f"    Games skipped due to errors: {errors}")
    print(f"    Saved → {out}")


if __name__ == "__main__":
    fetch_player_match_stats()
    print("\nDone.")
