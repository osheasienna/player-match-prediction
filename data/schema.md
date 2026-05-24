# Dataset Schema — `ligue1_features.csv`

**Rows:** 2 092 matches · **Columns:** 40  
**Coverage:** Ligue 1, seasons 2018/19 – 2023/24  
**Target:** `result` (H / D / A from the home team's perspective)

---

## Identifiers & Context

| Column | Type | Description |
|--------|------|-------------|
| `match_id` | string | Understat match ID (unique per match) |
| `date` | date | Match date (YYYY-MM-DD) |
| `season` | int | Season start year (e.g. 2022 = 2022/23 season) |
| `match_week` | int | Approximate gameweek within the season (1–38) |
| `home_team` | string | Home team name (Understat convention) |
| `away_team` | string | Away team name (Understat convention) |

---

## Rolling xG Features (5-match window, lag-safe)

All rolling features are computed from the team's **prior** matches only.  
`shift(1).rolling(5, min_periods=1)` is applied before joining — the current match is never included (no leakage).

| Column | Type | Description |
|--------|------|-------------|
| `home_roll_xg_for` | float | Home team: rolling mean xG scored in last 5 matches |
| `home_roll_xg_against` | float | Home team: rolling mean xG conceded in last 5 matches |
| `home_roll_xg_diff` | float | `home_roll_xg_for − home_roll_xg_against` |
| `away_roll_xg_for` | float | Away team: rolling mean xG scored in last 5 matches |
| `away_roll_xg_against` | float | Away team: rolling mean xG conceded in last 5 matches |
| `away_roll_xg_diff` | float | `away_roll_xg_for − away_roll_xg_against` |
| `net_xg_diff` | float | `home_roll_xg_diff − away_roll_xg_diff` (positive = home advantage) |

---

## Rolling Goal Features (5-match window, lag-safe)

| Column | Type | Description |
|--------|------|-------------|
| `home_roll_goals_for` | float | Home team: rolling mean goals scored |
| `home_roll_goals_against` | float | Home team: rolling mean goals conceded |
| `home_roll_goal_diff` | float | `home_roll_goals_for − home_roll_goals_against` |
| `away_roll_goals_for` | float | Away team: rolling mean goals scored |
| `away_roll_goals_against` | float | Away team: rolling mean goals conceded |
| `away_roll_goal_diff` | float | `away_roll_goals_for − away_roll_goals_against` |
| `net_goal_diff` | float | `home_roll_goal_diff − away_roll_goal_diff` |

---

## Rolling Form Features (5-match window, lag-safe)

| Column | Type | Description |
|--------|------|-------------|
| `home_roll_won` | float | Home team: rolling win rate (0–1) |
| `away_roll_won` | float | Away team: rolling win rate (0–1) |
| `net_win_rate` | float | `home_roll_won − away_roll_won` |
| `home_roll_points` | float | Home team: rolling mean points per game (0–3) |
| `away_roll_points` | float | Away team: rolling mean points per game (0–3) |
| `net_points` | float | `home_roll_points − away_roll_points` |

---

## Market Value Features (Transfermarkt)

Sourced from the Transfermarkt reference table for the corresponding season.  
Values are in **EUR**. These are pre-season squad assessments — no within-season leakage.

| Column | Type | Description |
|--------|------|-------------|
| `home_squad_value` | float | Home squad total market value (EUR) |
| `away_squad_value` | float | Away squad total market value (EUR) |
| `squad_value_ratio` | float | `home_squad_value / away_squad_value` (>1 = home richer) |
| `log_squad_value_ratio` | float | Natural log of `squad_value_ratio` |
| `squad_value_diff` | float | `home_squad_value − away_squad_value` (EUR) |
| `home_avg_player_value` | float | Home squad mean player market value (EUR) |
| `away_avg_player_value` | float | Away squad mean player market value (EUR) |
| `home_squad_age` | float | Home squad mean player age |
| `away_squad_age` | float | Away squad mean player age |

---

## Pre-Match Forecast Features (Understat)

Understat publishes win-probability forecasts based on their xG model. These are derived from historical team xG and are available **before** the match — valid features.

| Column | Type | Description |
|--------|------|-------------|
| `home_win_prob` | float | Understat forecast: P(home win) |
| `draw_prob` | float | Understat forecast: P(draw) |
| `away_win_prob` | float | Understat forecast: P(away win) |

> **Note:** `home_win_prob + draw_prob + away_win_prob ≈ 1.0`

---

## Target Variables

| Column | Type | Description |
|--------|------|-------------|
| `result` | string | Match result from home team's perspective: `H` (home win), `D` (draw), `A` (away win) |
| `result_code` | int | Encoded target: `0` = H, `1` = D, `2` = A |

**Class distribution:**

| Class | Count | Share |
|-------|-------|-------|
| H (home win) | 879 | 42.0% |
| A (away win) | 664 | 31.7% |
| D (draw) | 549 | 26.2% |

---

## Feature Selection Results Summary

From `outputs/feature_leaderboard.csv`. Composite score = equal-weight average of percentile rank across Filter (MI), Wrapper (RFECV), and Embedded (L1 Lasso + RF) methods.

| Rank | Feature | Composite Score | Survives All 3 Methods? |
|------|---------|----------------|------------------------|
| 1 | `home_win_prob` | 0.93 | Yes |
| 2 | `away_win_prob` | 0.92 | Yes |
| 3 | `squad_value_ratio` | 0.86 | Yes |
| 4 | `log_squad_value_ratio` | 0.84 | Yes |
| 5 | `draw_prob` | 0.80 | Yes |
| 6 | `home_squad_value` | 0.77 | Yes |
| 7 | `away_avg_player_value` | 0.66 | Yes |
| 8 | `squad_value_diff` | 0.66 | Lasso zeroes it (correlated with ratio) |
| 9 | `home_roll_xg_against` | 0.64 | Yes |
| 10 | `home_roll_xg_diff` | 0.64 | Yes |
| 11 | `net_xg_diff` | 0.62 | Lasso zeroes it (collinear with components) |
| 12 | `home_avg_player_value` | 0.60 | Yes |
| 13 | `home_roll_xg_for` | 0.58 | Yes |
| 14 | `away_roll_xg_for` | 0.58 | Yes |
| 15 | `net_goal_diff` | 0.57 | Lasso zeroes it |

---

## PCA Diagnostic

- **7 principal components** explain 80% of total variance  
- **12 principal components** explain 95% of total variance  
- Confirms substantial multicollinearity across the 32 features — regularisation (L1/L2) or explicit feature selection is required for modelling.

---

## Notes on Data Quality

- Season 2019/20 has 279 matches (not 380) due to the COVID-19 interrupted season.
- Season 2023/24 has 306 matches at time of collection (season partially complete).
- 6 matches have missing away rolling features — these are the first-ever Ligue 1 matches for promoted clubs with no prior season history in the dataset.
- Transfermarkt squad values are from a publicly documented reference table (the API at `transfermarkt-api.fly.dev` was unavailable from this environment due to upstream blocking).
