"""Check the chronological split on the retained Lichess snapshot (no training, no model).

python3 project/check_split.py              # reads the retained snapshot from your lake (needs az login)
python3 project/check_split.py --local DIR  # a folder of parquet files + _manifest.json (for tests)
Writes split.json: the exact day boundaries, sizes, class counts and membership hashes.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ml_contract as mc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", help="folder with the retained parquet parts and _manifest.json")
    ap.add_argument("--val", type=float, default=.15)
    ap.add_argument("--test", type=float, default=.15)
    ap.add_argument("--deploy", type=float, default=.10)
    a = ap.parse_args()
    tmp = None
    if a.local:
        folder = a.local
    else:
        src = json.loads((HERE.parent / ".data_source.json").read_text())
        from adlfs import AzureBlobFileSystem
        from azure.identity import AzureCliCredential
        account = src["path"].split("@")[1].split(".")[0]
        remote = src["path"].split(".dfs.core.windows.net/")[1].rstrip("/")
        fs = AzureBlobFileSystem(account_name=account, credential=AzureCliCredential())
        tmp = tempfile.TemporaryDirectory(prefix="p2-split-")
        folder = tmp.name
        print("downloading the retained snapshot to check its bytes ...")
        fs.get("curated/" + remote, folder, recursive=True)
        if not (Path(folder) / "_manifest.json").exists():   # fs.get may nest one level
            folder = str(next(Path(folder).rglob("_manifest.json")).parent)
    digest = mc.snapshot_fingerprint(folder)
    print("snapshot bytes verified:", digest)
    counts = mc.day_counts(folder)
    bounds = mc.split_boundaries(counts, a.val, a.test, a.deploy)
    scan = mc.scan_split(folder, bounds)
    manifest = mc.split_manifest(scan, bounds, seed=42, data_sha256=digest)
    total = sum(scan["rows"])
    print(f"{len(counts)} UTC days, {total:,} games")
    print(f"{'part':7s}{'games':>12s}{'share':>8s}  first game (UTC)      last game (UTC)       White/draw/Black")
    for n in ["train", "val", "test", "deploy"]:
        p = manifest[n]; c = p["class_counts_white_draw_black"]; s = sum(c)
        print(f"{n:7s}{p['rows']:>12,}{100*p['rows']/total:>7.1f}%  {p['first_game_utc']}   {p['last_game_utc']}   "
              + " / ".join(f"{100*x/s:.1f}%" for x in c))
    assert min(scan["rows"]) > 0
    assert sum(scan["rows"]) == sum(p["rows"] for p in json.loads((Path(folder)/"_manifest.json").read_text())["parts"])
    out = HERE.parent / "split.json"
    out.write_text(json.dumps(manifest, indent=2))
    print("split_id:", manifest["split_id"])
    print("wrote", out)


if __name__ == "__main__":
    main()
