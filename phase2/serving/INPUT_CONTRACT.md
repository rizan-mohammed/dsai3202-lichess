# Input contract - Lichess outcome endpoint

**What it does.** Given one record per game after 20 plies (10 moves each), it returns P(White win), P(draw) and P(Black win).

**What the interface accepts: computed features, not raw games.** The 16 fields below are the Phase-1 features (`FEATURES.md`). To get them from a raw game, use
`project/features_from_game.py` (it implements the same rules as Phase 1's `make_features`; see "Computing the fields"). The service does not parse PGN.

## Request
`POST <scoring_uri>` with header `Authorization: Bearer <key>` and `Content-Type: application/json`. The body is JSON:
`{"data": [ {record}, ... ]}` - 1 to 1000 records. Each record has an optional `"id"` (string or number; echoed back; default = position 0, 1, 2 ...) and the 16 fields:

| Field | Type | Allowed | Meaning |
|---|---|---|---|
| white_elo, black_elo | number | 100..4000 | Rating of each player before the game |
| elo_diff | number | -4000..4000 | white_elo - black_elo |
| elo_mean | number | 100..4000 | (white_elo + black_elo) / 2 |
| tc_base | number | 0..21600 | Base time in seconds |
| tc_inc | number | 0..600 | Increment per move in seconds |
| speed_class | number | 0, 1, 2 or 3 | Class of tc_base + 40 * tc_inc: < 180 -> 0 bullet, < 480 -> 1 blitz, < 1500 -> 2 rapid, else 3 classical |
| eco | string | non-empty | ECO opening code, e.g. "C50" (unknown codes are allowed; the model treats them as unseen) |
| eco_family | string | non-empty | First letter of eco |
| opening_family | string | non-empty | Opening name before the first ':' (unseen names allowed) |
| w_clk_1, b_clk_1 | number | 0..100000 | Seconds left on each clock after that side's first move |
| w_used_10, b_used_10 | number | -100000..100000 | Seconds used over each side's first 10 moves (clock after move 1 minus after move 10; negative is possible with increments) |
| used_diff_10 | number | any in range | w_used_10 - b_used_10 |
| unequal_start | 0 or 1 | 0, 1 (true/false accepted) | 1 if w_clk_1 != b_clk_1 |

Nothing may be null. Numbers must be JSON numbers (not strings). Information after ply 20 (result, rating change, termination, game length) is never an input.
Derived fields that disagree with their parts (e.g. `elo_diff` != `white_elo - black_elo`) are **not refused**: the answer comes back with an entry in `warnings`.

## Response (HTTP 200)
```json
{"n": 3, "ids": ["a", "b", "c"], "classes": ["white_win", "draw", "black_win"],
 "predictions": [0, 2, 0], "labels": ["white_win", "black_win", "white_win"],
 "probabilities": [[0.525054, 0.038696, 0.43625], ...], "warnings": [], "model": "lichess-outcome:<version>"}
```
`probabilities` rows are in the order of `classes`, rounded to 6 decimals, and sum to 1. `predictions` is the index of the largest probability.
`model` names the exact registered model version that answered.

## Errors (the request is rejected as a whole; no partial answers)
| HTTP | `code` | When |
|---|---|---|
| 400 | empty_body | no body |
| 400 | invalid_json | body is not JSON |
| 400 | bad_shape | not an object with a non-empty list `data` of objects |
| 400 | too_many_rows | more than 1000 records |
| 400 | missing_field | a record lacks one of the 16 fields (the message names it) |
| 400 | invalid_value | wrong type, null, or outside the allowed range (the message names row and field) |
| 401 | - | missing or wrong key (answered by Azure before the script runs) |
| 500 | internal_error | the model failed on valid input (should not happen; traceback in the deployment log) |
Every error body is JSON: `{"error": "...", "code": "...", "expected_fields": [...], "model": "..."}`.

## Computing the fields from a raw game
`python3 project/features_from_game.py games.json > request.json` where `games.json` is a list of
`{"id", "white_elo", "black_elo", "time_control": "300+3", "eco", "opening", "clocks20": [20 clock values, White's first move first]}`.
Only timed games with 20 plies and clock annotations can be scored; the model's prediction point is ply 20.

## Example
`serving/sample-request.json` (three valid records) and `serving/bad-requests/` (five invalid ones) are the examples used by `project/client_tests.py`.

## Calling it
```
SCORING_URI=$(az ml online-endpoint show -n <endpoint> --query scoring_uri -o tsv)
KEY=$(az ml online-endpoint get-credentials -n <endpoint> --query primaryKey -o tsv)
curl -s -X POST "$SCORING_URI" -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" --data-binary @serving/sample-request.json
```
The key is a password: never paste it into the README, slides or screenshots.
