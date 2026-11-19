"""Point MLflow at your Azure ML workspace (adapted from Lab-9 aml_common.py).
Order: env MLFLOW_TRACKING_URI, then Lab-P2/.tracking_uri, then `az ml workspace show`."""
import os
import pathlib
import subprocess
import sys

WEEK = pathlib.Path(__file__).resolve().parent.parent


def tracking_uri():
    uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if uri:
        return uri
    f = WEEK / ".tracking_uri"
    if f.is_file() and f.read_text().strip():
        return f.read_text().strip()
    try:
        uri = subprocess.run(["az", "ml", "workspace", "show", "--query", "mlflow_tracking_uri", "-o", "tsv"],
                             capture_output=True, text=True, timeout=60, check=True).stdout.strip()
    except Exception as e:  # noqa: BLE001
        sys.exit(f"cannot find the workspace tracking URI ({e}). Run: bash scripts/setup.sh")
    if not uri:
        sys.exit("no mlflow_tracking_uri from az ml workspace show. Run: bash scripts/setup.sh")
    f.write_text(uri + "\n")
    return uri


def connect():
    import mlflow
    from mlflow.tracking import MlflowClient
    uri = tracking_uri()
    if uri.startswith("azureml"):
        try:
            import azureml.mlflow  # noqa: F401  registers the azureml:// scheme
        except ImportError:
            sys.exit("pip3 install --user azureml-mlflow==1.58.0 mlflow==2.17.2   then run this again")
    mlflow.set_tracking_uri(uri)
    return mlflow, MlflowClient()


def read_file(name):
    f = WEEK / name
    return f.read_text().strip() if f.is_file() else ""
