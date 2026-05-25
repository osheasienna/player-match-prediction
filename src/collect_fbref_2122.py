"""Scrape FBref player match stats for the missing 2021/22 season only.

Runs with headless=False so a visible Chrome window opens — if FBref shows a
CAPTCHA, solve it in the browser and the scraper will automatically continue.

Once complete, this script APPENDS the 2021/22 rows to raw/fbref_player_match.csv
and re-sorts by season/date so the file is ready to re-run feature engineering.

Usage:
    python3 src/collect_fbref_2122.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import soccerdata as sd

LEAGUE  = "FRA-Ligue 1"
SEASON  = 2122          # soccerdata compact code for 2021/22
RAW_DIR = Path(__file__).parent.parent / "raw"
OUT_PATH = RAW_DIR / "fbref_player_match.csv"
CHECKPOINT = RAW_DIR / "fbref_player_match_2122_checkpoint.pkl"


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            f"{a}_{b}".strip("_") if a and not a.startswith("Unnamed") else b
            for a, b in df.columns
        ]
    return df


def main() -> None:
    print("Opening visible Chrome window — solve any CAPTCHA that appears, then wait.")
    print("DO NOT close the browser window while the scrape is running.\n")

    fbref = sd.FBref(
        leagues=LEAGUE,
        seasons=[SEASON],
        headless=False,       # visible browser so you can solve CAPTCHAs
        no_cache=False,       # still use any cached pages from prior runs
    )

    # Get the schedule for 2021/22 to find all game IDs
    print("Reading match schedule for 2021/22 ...")
    try:
        schedule = fbref.read_schedule(force_cache=True).reset_index()
    except Exception:
        schedule = fbref.read_schedule().reset_index()

    id_col = "game_id" if "game_id" in schedule.columns else "game_id"
    game_ids = schedule[id_col].dropna().tolist()
    print(f"  {len(game_ids)} games found in 2021/22 schedule")

    # Resume from checkpoint if one exists
    dfs: list[pd.DataFrame] = []
    completed: set = set()
    if CHECKPOINT.exists():
        saved = pd.read_pickle(CHECKPOINT)
        dfs.append(saved)
        if "game_id" in saved.columns:
            completed = set(saved["game_id"].dropna().unique())
        print(f"  Resuming from checkpoint: {len(dfs[0])} rows already saved")

    remaining = [gid for gid in game_ids if gid not in completed]
    print(f"  {len(remaining)} games to fetch\n")

    errors = 0
    for i, gid in enumerate(remaining, 1):
        try:
            df = fbref.read_player_match_stats(stat_type="summary", match_id=gid)
            dfs.append(flatten_columns(df.reset_index()))
        except Exception as e:
            errors += 1
            print(f"  Skip {gid}: {e}")
            continue

        if i % 50 == 0:
            combined = pd.concat(dfs, ignore_index=True)
            combined.to_pickle(CHECKPOINT)
            print(f"  [{i}/{len(remaining)}] {100*i/len(remaining):.0f}% — checkpoint saved, {errors} errors")

    if not dfs:
        print("\nERROR: No data collected.")
        return

    new_data = pd.concat(dfs, ignore_index=True)
    print(f"\n2021/22 rows collected: {len(new_data)}")

    # Merge with existing CSV
    if OUT_PATH.exists():
        existing = pd.read_csv(OUT_PATH)
        # Drop any 2122 rows that might already be there (partial data)
        if "season" in existing.columns:
            existing = existing[existing["season"] != 2122]
        combined = pd.concat([existing, new_data], ignore_index=True)
        # Sort by season — coerce to string first to handle mixed int/str types
        if "season" in combined.columns:
            combined["season"] = combined["season"].astype(str)
            combined = combined.sort_values("season").reset_index(drop=True)
        combined.to_csv(OUT_PATH, index=False)
        print(f"Merged and saved: {len(combined)} total rows → {OUT_PATH}")
    else:
        new_data.to_csv(OUT_PATH, index=False)
        print(f"Saved: {len(new_data)} rows → {OUT_PATH}")

    if CHECKPOINT.exists():
        CHECKPOINT.unlink()

    print(f"\nErrors: {errors}")
    print("\nDone. Now re-run:")
    print("  python3 src/feature_engineering_player.py")
    print("  python3 src/feature_selection_player.py")


if __name__ == "__main__":
    main()
