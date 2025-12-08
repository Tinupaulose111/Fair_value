import os
import json
import dill
import subprocess
import logging
import requests
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# =========================================================
# Project imports
# =========================================================
from core.custom_transformers import RowFilter, FeatureEngineer, Preprocessor
from core.metrics_logger import log_metrics


# =========================================================
# PATH SETUP (repo / Docker safe)
# =========================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))      # .../drift
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))  # repo root (/app in Docker)

RETRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "trainer", "retrain_model.py")

# =========================================================
# CONFIG
# =========================================================
OLD_FILE = os.path.join(PROJECT_ROOT, "data", "Marketdata_newdata.csv")
NEW_FILE = os.path.join(PROJECT_ROOT, "data", "Marketdata_newdata_update.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "final_model_fairvalue.pkl")

# Artifacts are written inside the repo workspace (allowed in GitHub runner)
ARTIFACT_DIR = Path(PROJECT_ROOT) / "drift_artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

DRIFT_REPORT_PATH = ARTIFACT_DIR / "drift_report.json"
DRIFT_LOG = ARTIFACT_DIR / "drift_detection.log"

# Thresholds
DRIFT_P_THRESHOLD = 0.05
DRIFT_KS_THRESHOLD = 0.10
MIN_ROWS_REQUIRED = 30
OVERALL_SEVERITY_TRIGGER = 20.0


# =========================================================
# LOGGING
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(DRIFT_LOG),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("drift_monitor")


# =========================================================
# GITHUB WORKFLOW TRIGGER
# =========================================================
def trigger_github_trainer(overall_severity: float) -> Tuple[bool, dict]:
    """
    Trigger the retrain GitHub Actions workflow (retrain.yml)
    via workflow_dispatch.
    """

    owner = os.environ.get("REPO_OWNER")
    repo = os.environ.get("REPO_NAME")
    workflow_file = os.environ.get("TRAINER_WORKFLOW")
    ref = os.environ.get("TARGET_BRANCH", "main")
    token = os.environ.get("GHCR_PAT")  # your secret name

    if not all([owner, repo, workflow_file, token]):
        logger.error("❌ Missing required GitHub env vars (REPO_OWNER / REPO_NAME / TRAINER_WORKFLOW / GHCR_PAT)")
        return False, {"error": "missing_env"}

    url = (
        f"https://api.github.com/repos/"
        f"{owner}/{repo}/actions/workflows/{workflow_file}/dispatches"
    )

    payload = {
        "ref": ref,
        "inputs": {
            "reason": "data_drift_detected",
            "overall_severity": f"{overall_severity:.2f}",
        },
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=15)

        if resp.status_code in (204, 201):
            logger.info("✅ Retrain workflow triggered successfully")
            return True, {"status": resp.status_code}

        logger.error(f"❌ Trigger failed: {resp.status_code} | {resp.text}")
        return False, {"status": resp.status_code, "text": resp.text}

    except Exception as e:
        logger.exception("❌ GitHub API call failed")
        return False, {"error": str(e)}


# =========================================================
# HELPERS
# =========================================================
def now_utc() -> str:
    return datetime.utcnow().isoformat()


def safe_load_pipeline(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at: {path}")

    with open(path, "rb") as f:
        obj = dill.load(f)

    # In your project you save {"pipeline": ..., "smearing_factor": ...}
    if isinstance(obj, dict) and "pipeline" in obj:
        return obj["pipeline"]

    return obj


def clean_like_training(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[(df["EPS"] > 0) & (df["Book Value"] > 0)]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def get_preprocessor(pipeline):
    # Your training pipeline is [RowFilter -> FeatureEngineer -> Preprocessor -> Model]
    try:
        return pipeline[:-1]
    except Exception:
        return pipeline


def transform(preproc, df: pd.DataFrame) -> pd.DataFrame:
    arr = preproc.transform(df)
    try:
        cols = preproc.get_feature_names_out()
    except Exception:
        cols = [f"f{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=cols)


def feature_severity(old: pd.Series, new: pd.Series) -> float:
    ks_stat, _ = ks_2samp(old, new)
    mean_shift = abs(new.mean() - old.mean()) / (old.std() + 1e-9)
    score = (0.7 * ks_stat + 0.3 * min(mean_shift / 3, 1)) * 100.0
    return float(score)


# =========================================================
# MAIN
# =========================================================
def main():
    logger.info("🚀 Drift detection started")

    # Check required files
    for p in [OLD_FILE, NEW_FILE, MODEL_PATH]:
        if not os.path.exists(p):
            logger.error(f"❌ Missing required file: {p}")
            return

    # Load data
    df_old = clean_like_training(pd.read_csv(OLD_FILE))
    df_new = clean_like_training(pd.read_csv(NEW_FILE))

    # Load model pipeline and preprocessor
    pipeline = safe_load_pipeline(MODEL_PATH)
    preproc = get_preprocessor(pipeline)

    df_old_t = transform(preproc, df_old)
    df_new_t = transform(preproc, df_new)

    results = {}
    severities = []

    for col in df_old_t.columns:
        old = df_old_t[col].dropna()
        new = df_new_t[col].dropna()

        if len(old) < MIN_ROWS_REQUIRED or len(new) < MIN_ROWS_REQUIRED:
            continue

        ks_stat, p_val = ks_2samp(old, new)
        sev = feature_severity(old, new)

        drifted = (p_val < DRIFT_P_THRESHOLD) and (ks_stat > DRIFT_KS_THRESHOLD)

        results[col] = {
            "p_value": float(p_val),
            "ks_stat": float(ks_stat),
            "severity": float(sev),
            "drift": drifted,
        }

        severities.append(sev)

    overall_sev = float(np.mean(severities)) if severities else 0.0

    drift_detected = (
        overall_sev >= OVERALL_SEVERITY_TRIGGER
        or any(v["drift"] for v in results.values())
    )

    report = {
        "timestamp": now_utc(),
        "overall_severity": overall_sev,
        "drift_detected": drift_detected,
        "features": results,
    }

    with open(DRIFT_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"📊 Overall severity: {overall_sev:.2f}")

    if drift_detected:
        logger.warning("🚨 Drift detected — triggering retrain workflow")
        ok, info = trigger_github_trainer(overall_sev)

        if not ok:
            logger.warning("⚠ GitHub trigger failed — running local retrain")
            subprocess.run(["python", RETRAIN_SCRIPT], check=False)

    else:
        logger.info("✅ No drift detected")

    return report


if __name__ == "__main__":
    main()

