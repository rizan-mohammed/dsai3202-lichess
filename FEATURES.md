# Features (v1) - Lichess standard rated, prediction point = after 20 plies (10 moves each)

Pipeline: bronze (raw) -> silver (`lichess_silver`: every valid game) -> gold (`lichess_gold`: games eligible for the 20-ply prediction + `label`) -> features (`curated/lichess_features_v1`, built by `make_features` in `lichess_00_common`).

Target `label`: 0 = White wins (1-0), 1 = draw (1/2-1/2), 2 = Black wins (0-1).
All features below are known once 20 plies have been played. Clocks are seconds remaining after each ply (`clocks20`, index 0 = White's 1st move).

| Feature | Type | Source column | Definition | Available at prediction time |
|---|---|---|---|---|
| game_id | string | Site header | Lichess game id (key, not a model input) | yes |
| game_ts | timestamp | UTCDate+UTCTime | Game start (UTC); used only for chronological splits | yes |
| white_elo, black_elo | int | WhiteElo, BlackElo | Ratings before the game | yes |
| elo_diff | int | derived | white_elo - black_elo | yes |
| elo_mean | double | derived | (white_elo + black_elo) / 2 | yes |
| tc_base, tc_inc | int | TimeControl | Base seconds, increment seconds | yes |
| speed_class | int | derived | base + 40*inc: <180 bullet(0), <480 blitz(1), <1500 rapid(2), else classical(3) | yes |
| eco, eco_family | string | ECO | Opening code, first letter | yes (see risk 3) |
| opening_family | string | Opening | Text before ':' | yes (see risk 3) |
| w_clk_1, b_clk_1 | int | clocks20[0], [1] | Clock after each side's first move | yes |
| w_used_10, b_used_10 | int | clocks20[0]-[18], [1]-[19] | Seconds used over the first 10 moves | yes |
| used_diff_10 | int | derived | w_used_10 - b_used_10 | yes |
| unequal_start | int | derived | 1 if first clocks differ (berserk / odds) | yes |

## Never features (leakage)
`result` (target), `white_diff`/`black_diff` (rating change, depends on result), `termination`, `n_plies` (full game length), `moves20`/`clocks20` raw arrays (not used in v1), player names/ids (dropped at parse).

## Leakage and bias risks to state in the README
1. Rating diffs and termination are only known after the game: excluded.
2. Selection bias: gold keeps only games reaching 20 plies with complete clocks, timed, in the month; early resignations/abandonments are quarantined with a reason (see `coverage.json` and the gate's `eligibility` rule).
3. Lichess assigns the opening name from the moves; a few deep lines may name moves beyond ply 20. Check by comparing model gain from opening features.
4. The May file also contains late-April games; they stay in silver but are quarantined as `outside_month` and counted.
