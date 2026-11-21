"""make_fixed_inputs.py - pick 200 FIXED games from the DEPLOY days of the retained snapshot (not the test days) and write them as a
request file. The same file is later sent to the endpoint and scored offline, so both sides see identical inputs (PDF III).

    python3 project/make_fixed_inputs.py [--n 200]
Rows are chosen by the smallest game-id hash, so the choice is the same on every machine. Writes evidence/fixed_inputs.json.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ml_contract as mc

FIELDS = ["white_elo", "black_elo", "elo_diff", "elo_mean", "tc_base", "tc_inc", "speed_class", "eco", "eco_family", "opening_family",
          "w_clk_1", "b_clk_1", "w_used_10", "b_used_10", "used_diff_10", "unequal_start"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--local", help="folder with the parquet parts (for tests)")
    a = ap.parse_args()
    tmp = None
    folder = a.local
    if not folder:
        src = json.loads((HERE.parent / ".data_source.json").read_text())
        from adlfs import AzureBlobFileSystem
        from azure.identity import AzureCliCredential
        account = src["path"].split("@")[1].split(".")[0]
        remote = src["path"].split(".dfs.core.windows.net/")[1].rstrip("/")
        fs = AzureBlobFileSystem(account_name=account, credential=AzureCliCredential())
        tmp = tempfile.TemporaryDirectory(prefix="p2-fixed-")
        folder = tmp.name
        print("downloading the retained snapshot ...")
        fs.get("curated/" + remote, folder, recursive=True)
        if not (Path(folder) / "_manifest.json").exists():
            folder = str(next(Path(folder).rglob("_manifest.json")).parent)
    digest = mc.snapshot_fingerprint(folder)
    ref = json.loads((HERE / "split.json").read_text()) if (HERE / "split.json").exists() else json.loads((HERE.parent / "split.json").read_text())
    bounds = ref["boundaries_epoch"]
    pool = []
    for f in mc.part_files(folder):
        df = pd.read_parquet(f, columns=[mc.KEY_COL, mc.TIME_COL] + FIELDS)
        d = df[mc.partition_of(mc.epoch_seconds(df[mc.TIME_COL]), bounds) == 3].dropna(subset=FIELDS)
        if len(d):
            d = d.assign(_h=mc.id_hash(d[mc.KEY_COL]))
            pool.append(d.nsmallest(a.n, "_h"))
    pick = pd.concat(pool).nsmallest(a.n, "_h")
    rows = []
    for _, r in pick.iterrows():
        row = {"id": str(r[mc.KEY_COL])}
        for c in FIELDS:
            v = r[c]
            row[c] = v if isinstance(v, str) else float(v) if c == "elo_mean" else int(v)
        rows.append(row)
    out = HERE.parent / "evidence" / "fixed_inputs.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"data": rows}))
    meta = {"rows": len(rows), "from_partition": "deploy", "selection": "smallest game-id hash", "snapshot_sha256": digest,
            "split_id": ref["split_id"], "first_ids": [r["id"] for r in rows[:5]]}
    (HERE.parent / "evidence" / "fixed_inputs_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out} ({len(rows)} games from the deploy days; snapshot {digest[:12]}...)")


if __name__ == "__main__":
    main()
