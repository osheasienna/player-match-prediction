"""Fetch Ligue 1 match results and xG data from Understat (seasons 2018-2023).

Saves raw/understat_ligue1_raw.csv with one row per match.
Columns: season, match_id, date, home_team, away_team,
         home_goals, away_goals, home_xg, away_xg,
         home_win_prob, draw_prob, away_win_prob, result
"""

import asyncio
import time
from pathlib import Path

import aiohttp
import pandas as pd
import understat

LEAGUE = "Ligue 1"
SEASONS = [2018, 2019, 2020, 2021, 2022, 2023]
OUT_PATH = Path(__file__).parent.parent / "raw" / "understat_ligue1_raw.csv"


async def fetch_season(session: aiohttp.ClientSession, season: int) -> list[dict]:
    u = understat.Understat(session)
    matches = await u.get_league_results(LEAGUE, season)
    rows = []
    for m in matches:
        if not m.get("isResult"):
            continue
        home_goals = int(m["goals"]["h"])
        away_goals = int(m["goals"]["a"])
        if home_goals > away_goals:
            result = "H"
        elif home_goals < away_goals:
            result = "A"
        else:
            result = "D"
        rows.append({
            "season": season,
            "match_id": m["id"],
            "date": m["datetime"][:10],
            "home_team": m["h"]["title"],
            "away_team": m["a"]["title"],
            "home_goals": home_goals,
            "away_goals": away_goals,
            "home_xg": float(m["xG"]["h"]),
            "away_xg": float(m["xG"]["a"]),
            "home_win_prob": float(m["forecast"]["w"]),
            "draw_prob": float(m["forecast"]["d"]),
            "away_win_prob": float(m["forecast"]["l"]),
            "result": result,
        })
    return rows


async def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    async with aiohttp.ClientSession() as session:
        for season in SEASONS:
            print(f"  Fetching season {season}/{season+1} ...", flush=True)
            try:
                rows = await fetch_season(session, season)
                print(f"    -> {len(rows)} matches")
                all_rows.extend(rows)
            except Exception as e:
                print(f"    -> ERROR: {e}")
            await asyncio.sleep(1)

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "match_id"]).reset_index(drop=True)

    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows to {OUT_PATH}")
    print(df.groupby("season")["match_id"].count().rename("matches").to_string())
    print("\nResult distribution:")
    print(df["result"].value_counts().to_string())


if __name__ == "__main__":
    asyncio.run(main())
