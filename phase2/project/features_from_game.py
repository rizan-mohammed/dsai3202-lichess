"""features_from_game.py - compute the 16 model inputs from one raw Lichess game (the Phase-1 `make_features` rules, in Python).

    python3 project/features_from_game.py games.json > request.json
games.json: a list of {"id", "white_elo", "black_elo", "time_control": "300+3", "eco": "C50", "opening": "Italian Game: ...",
"clocks20": [seconds left after each of the first 20 plies, White's first move first]}.
Rules (FEATURES.md): elo_diff = w - b; elo_mean = (w + b) / 2; tc_base/tc_inc from 'base+inc'; speed_class from base + 40*inc:
<180 -> 0, <480 -> 1, <1500 -> 2, else 3; eco_family = first letter of eco; opening_family = text before ':';
w_clk_1 = clocks20[0], b_clk_1 = clocks20[1]; w_used_10 = clocks20[0] - clocks20[18]; b_used_10 = clocks20[1] - clocks20[19];
used_diff_10 = w_used_10 - b_used_10; unequal_start = 1 if w_clk_1 != b_clk_1.
Only a game with 20 plies, clock annotations and a timed control can be scored (the model's prediction point is ply 20).
"""
import json
import sys


def features_from_game(g):
    if g.get("time_control", "-") in ("-", "", None) or "+" not in g["time_control"]:
        raise ValueError("untimed or malformed time_control: only 'base+increment' games can be scored")
    base, inc = (int(x) for x in g["time_control"].split("+"))
    k = g["clocks20"]
    if len(k) < 20 or any(x is None for x in k[:20]):
        raise ValueError("need 20 clock values (the first 20 plies)")
    w, b = int(g["white_elo"]), int(g["black_elo"])
    t = base + 40 * inc
    used_w, used_b = k[0] - k[18], k[1] - k[19]
    return {"id": g.get("id"), "white_elo": w, "black_elo": b, "elo_diff": w - b, "elo_mean": (w + b) / 2.0,
            "tc_base": base, "tc_inc": inc, "speed_class": 0 if t < 180 else 1 if t < 480 else 2 if t < 1500 else 3,
            "eco": g["eco"], "eco_family": g["eco"][:1], "opening_family": g["opening"].split(":")[0].strip(),
            "w_clk_1": k[0], "b_clk_1": k[1], "w_used_10": used_w, "b_used_10": used_b, "used_diff_10": used_w - used_b,
            "unequal_start": int(k[0] != k[1])}


if __name__ == "__main__":
    games = json.load(open(sys.argv[1]))
    print(json.dumps({"data": [features_from_game(g) for g in games]}, indent=1))
