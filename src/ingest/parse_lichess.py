#!/usr/bin/env python3
"""Stream a Lichess .pgn.zst -> Parquet parts (headers + first 20 plies + clocks)."""
import sys, re, io, json, argparse, pathlib
import pyarrow as pa, pyarrow.parquet as pq
HDR = re.compile(r'^\[(\w+) "(.*)"\]\s*$')
TOK = re.compile(r'([^\s{}]+)\s*(\{[^}]*\})?')
CLK = re.compile(r'\[%clk (\d+):(\d\d):(\d\d)')
NUM = re.compile(r'\d+\.+')
RES = {"1-0", "0-1", "1/2-1/2", "*"}
N = 20
SCHEMA = pa.schema([("game_id", pa.string()), ("event", pa.string()), ("result", pa.string()),
    ("utc_date", pa.string()), ("utc_time", pa.string()), ("white_elo", pa.int32()),
    ("black_elo", pa.int32()), ("white_diff", pa.int32()), ("black_diff", pa.int32()),
    ("eco", pa.string()), ("opening", pa.string()), ("time_control", pa.string()),
    ("termination", pa.string()), ("n_plies", pa.int32()),
    ("moves20", pa.list_(pa.string())), ("clocks20", pa.list_(pa.int32()))])

def to_int(x):
    try: return int(x)
    except (TypeError, ValueError): return None

def parse_game(lines):
    h, mt = {}, []
    for ln in lines:
        m = HDR.match(ln)
        if m: h[m[1]] = m[2]
        elif ln.strip(): mt.append(ln.strip())
    moves, clocks, n = [], [], 0
    for tok, com in TOK.findall(" ".join(mt)):
        if tok in RES: continue
        tok = NUM.sub("", tok).rstrip("?!")
        if not tok or tok[0] in "[$(": continue
        n += 1
        if n <= N:
            moves.append(tok)
            m = CLK.search(com or "")
            clocks.append(int(m[1])*3600 + int(m[2])*60 + int(m[3]) if m else None)
    return {"game_id": h.get("Site", "").rsplit("/", 1)[-1] or None, "event": h.get("Event"),
        "result": h.get("Result"), "utc_date": h.get("UTCDate"), "utc_time": h.get("UTCTime"),
        "white_elo": to_int(h.get("WhiteElo")), "black_elo": to_int(h.get("BlackElo")),
        "white_diff": to_int(h.get("WhiteRatingDiff")), "black_diff": to_int(h.get("BlackRatingDiff")),
        "eco": h.get("ECO"), "opening": h.get("Opening"), "time_control": h.get("TimeControl"),
        "termination": h.get("Termination"), "n_plies": n, "moves20": moves, "clocks20": clocks}

def open_text(p):
    if p.endswith(".zst"):
        import zstandard
        r = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(open(p, "rb"))
        return io.TextIOWrapper(r, encoding="utf-8", errors="replace")
    return open(p, encoding="utf-8", errors="replace")

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("out")
ap.add_argument("--max-games", type=int, default=0)
a = ap.parse_args()
out = pathlib.Path(a.out)
out.mkdir(parents=True, exist_ok=False)          # refuses to overwrite an earlier run
rows, part, total, bad, buf = [], 0, 0, 0, []

def flush():
    global rows, part
    if rows:
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA),
                       out / f"part-{part:05d}.parquet", compression="zstd")
        part += 1; rows = []

def done(b):
    global total, bad
    r = parse_game(b)
    if not r["game_id"] or not r["result"]:
        bad += 1; return
    rows.append(r); total += 1
    if total % 500_000 == 0: print("parsed", total, file=sys.stderr, flush=True)
    if len(rows) >= 200_000: flush()

for ln in open_text(a.src):
    if ln.startswith("[Event ") and buf:
        done(buf); buf = []
        if a.max_games and total >= a.max_games: break
    buf.append(ln)
else:
    if buf: done(buf)
flush()
json.dump({"source": a.src, "games_parsed": total, "unparseable": bad,
           "parts": part, "plies_kept": N}, open(out / "_bronze-manifest.json", "w"), indent=2)
print("DONE", total, "games,", bad, "unparseable,", part, "parts")
