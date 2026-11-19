"""Retain the exact Phase 1 feature bytes before registering the Phase 2 data asset (adapted from Lab 8).

python3 project/retain_data.py <lake-account> [phase1-run-id]
Reads curated/lichess_runs/<run>/features (the published Phase 1 run, default 1022222200504518),
hashes and copies the Parquet parts into a content-addressed folder, and writes project/data-asset.yml.
Never overwrites an existing snapshot. A rerun verifies it instead of replacing it.
"""
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

WEEK = Path(__file__).resolve().parent.parent
DEFAULT_RUN = "1022222200504518"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def remote_sha(fs, remote):
    h = hashlib.sha256()
    with fs.open(remote, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: python3 project/retain_data.py <lake-account> [phase1-run-id]")
    from adlfs import AzureBlobFileSystem
    from azure.identity import AzureCliCredential
    import pyarrow.parquet as pq
    lake = sys.argv[1]
    run = sys.argv[2] if len(sys.argv) == 3 else DEFAULT_RUN
    fs = AzureBlobFileSystem(account_name=lake, credential=AzureCliCredential())
    success_path = f"curated/quality/lichess_runs/{run}/success.json"
    if not fs.exists(success_path):
        raise SystemExit(f"{success_path} missing: use the job run ID of a PUBLISHED Phase 1 run")
    success = json.loads(fs.cat_file(success_path))
    folder = f"curated/lichess_runs/{run}/features"
    files = sorted(p for p in fs.find(folder) if p.endswith(".parquet"))
    if not files:
        raise SystemExit("the Phase 1 features folder is missing or empty")
    with tempfile.TemporaryDirectory(prefix="p2-retain-") as tmp:
        cached, parts = [], []
        for i, remote in enumerate(files):
            rel = remote.removeprefix(folder + "/")
            local = Path(tmp) / rel
            local.parent.mkdir(parents=True, exist_ok=True)
            fs.get_file(remote, str(local))
            rows = pq.ParquetFile(local).metadata.num_rows
            parts.append({"path": rel, "sha256": sha(local), "rows": rows, "bytes": local.stat().st_size})
            cached.append(local)
            print(f"hashed {i + 1}/{len(files)}: {rel} ({rows:,} rows)")
        rows = sum(p["rows"] for p in parts)
        if rows != int(success["features_rows"]):
            raise SystemExit("row count differs from the Phase 1 success.json")
        provenance = {"format": 1, "phase1_run_id": run, "phase1_features_path": folder,
                      "phase1_source_sha256": success.get("source_sha256"),
                      "phase1_silver_version": success.get("silver_version"), "parts": parts, "rows": rows}
        digest = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
        dst = f"curated/lichess_snapshots/{digest}/features"
        marker = dst + "/_manifest.json"
        if fs.exists(marker):
            old = json.loads(fs.cat_file(marker))
            if old != {**provenance, "snapshot_sha256": digest}:
                raise SystemExit("existing snapshot manifest differs; refusing to overwrite it")
            actual = {p.removeprefix(dst + "/") for p in fs.find(dst) if p.endswith(".parquet")}
            if actual != {p["path"] for p in parts}:
                raise SystemExit("retained snapshot part list differs; refusing to reuse it")
            for part in parts:
                if remote_sha(fs, dst + "/" + part["path"]) != part["sha256"]:
                    raise SystemExit("retained snapshot was modified; do not reuse its asset version")
            print("verified existing retained snapshot")
        else:
            existing = {p.removeprefix(dst + "/") for p in fs.find(dst)} if fs.exists(dst) else set()
            if existing - {p["path"] for p in parts}:
                raise SystemExit("incomplete destination holds unexpected files; refusing to modify it")
            for local, part in zip(cached, parts):
                remote = dst + "/" + part["path"]
                if fs.exists(remote):
                    if remote_sha(fs, remote) != part["sha256"]:
                        raise SystemExit("incomplete destination has different bytes; refusing to overwrite it")
                else:
                    fs.put_file(str(local), remote)
            fs.pipe_file(marker, json.dumps({**provenance, "snapshot_sha256": digest}, indent=2).encode())
        uri = f"abfss://curated@{lake}.dfs.core.windows.net/lichess_snapshots/{digest}/features/"
        version = os.environ.get("DATA_VERSION", "1")
        yaml = (f"$schema: https://azuremlschemas.azureedge.net/latest/data.schema.json\n"
                f"name: lichess-features\nversion: {version}\ntype: uri_folder\n"
                f"description: Retained Phase 1 features (Lichess 2017-05) from job run {run}\n"
                f"path: {uri}\ntags:\n  snapshot_sha256: {digest}\n  phase1_run_id: \"{run}\"\n"
                f"  source_sha256: {success.get('source_sha256')}\n")
        (WEEK / "project/data-asset.yml").write_text(yaml)
        (WEEK / ".data_source.json").write_text(json.dumps({**provenance, "snapshot_sha256": digest, "path": uri}, indent=2) + "\n")
        print(f"retained {rows:,} rows; snapshot SHA-256 {digest}")
        print("wrote project/data-asset.yml; next: az ml data create -f project/data-asset.yml")


if __name__ == "__main__":
    main()
