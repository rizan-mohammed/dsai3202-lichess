"""python3 project/fetch_artifacts.py <run-id> <artifact-folder> <destination>   e.g.  ... holdout evidence/final"""
import sys
from pathlib import Path
from aml_common import connect

run_id, art, dest = sys.argv[1:4]
_, client = connect()
Path(dest).mkdir(parents=True, exist_ok=True)
print(client.download_artifacts(run_id, art, dest))
