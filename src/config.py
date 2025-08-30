from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
FIG_DIR = OUTPUTS_DIR / "figures"
REPORT_DIR = OUTPUTS_DIR / "reports"
ARTIFACTS_DIR = OUTPUTS_DIR / "artifacts"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file:{(ROOT / 'mlruns').as_posix()}")
