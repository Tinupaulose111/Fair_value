import os
import json
import shutil
import dill
import subprocess
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# =========================================================
# FIX: Correct import of custom transformers
# =========================================================
from core.custom_transformers import RowFilter, FeatureEngineer, Preprocessor
from core.metrics_logger import log_metrics


# =========================================================
# PATH FIX — Get correct project root
# =========================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))      # /app
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))  # /app/..

# Fix retraining command
RETRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "trainer", "retrain_model.py")
RETRAIN_CMD = ["python", RETRAIN_SCRIPT]


# =========================================================
# CONFIG
# =========================================================
OLD_FILE = "/app/data/Marketdata_newdata.csv"
NEW_FILE = "/app/data/Marketdata_newdata_update.csv"

MODEL_PATH = "/app/model/final_model_fairvalue.pkl"


# =========================================================
# ARTIFACT DIRECTORY (WRITEABLE BY appuser)
# =========================================================
ARTIFACT_DIR = Path("/app/drift_artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_PATH = ARTIFACT_DIR / "baseline_transformed.parquet"
DRIFT_REPORT_PATH = ARTIFACT_DIR / "drift_report.json"
TRANSFORMED_OLD_PATH = ARTIFACT_DIR / "transformed_old.parquet"
TRANSFORMED_NEW_PATH = ARTIFACT_DIR / "transformed_new.parquet"

# =========================================================
# LOG FILE MUST ALSO BE STORED INSIDE /app
# =========================================================
DRIFT_LOG = ARTIFACT_DIR / "drift_detection.log"

# thresholds
DRIFT_P_THRESHOLD = 0.05
DRIFT_KS_STAT_THRESHOLD = 0.10
MIN_ROWS_REQUIRED = 30
SEVERITY_WEIGHT_KS = 0.7
SEVERITY_WEIGHT_MEAN = 0.3
OVERALL_SEVERITY_TRIGGER = 20
BASELINE_UPDATE_ON_NO_DRIFT = True
BASELINE_MAX_AGE_DAYS = 30

SLACK_WEBHOOK = os.environ.get("DRIFT_SLACK_WEBHOOK")
EMAIL_NOTIFY = os.environ.get("DRIFT_ALERT_EMAIL")


# =========================================================
# Logging setup — write inside /app/drift_artifacts
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(DRIFT_LOG),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("drift_detector")


# =========================================================
# Helper functions
# =========================================================

def now_str():
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def safe_load_pipeline(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as f:
        saved = dill.load(f)
    pipeline = saved.get("pipeline")
    if pipeline is None:
        raise ValueError("No 'pipeline' key found in saved model.")
    return pipeline, saved


def clean_like_training(df):
    """Cleaning step identical to training."""
    if not {'EPS', 'Book Value'}.issubset(df.columns):
        return df.replace([np.inf, -np.inf], np.nan).dropna()

    df2 = df[(df['EPS'] > 0) & (df['Book Value'] > 0)].copy()
    df2 = df2.replace([np.inf, -np.inf], np.nan).dropna()
    return df2


def get_preprocess_pipeline(pipeline):
    try:
        return pipeline[:-1]  # remove final model
    except:
        return pipeline       # fallback


def transform_to_df(preproc, df_raw):
    transformed = preproc.transform(df_raw)

    try:
        feature_names = preproc.get_feature_names_out()
    except:
        feature_names = [f"f{i}" for i in range(transformed.shape[1])]

    return pd.DataFrame(transformed, columns=feature_names, index=df_raw.index)


def schema_validation(df_old, df_new):
    old_cols = set(df_old.columns)
    new_cols = set(df_new.columns)

    return {
        "added_columns": list(new_cols - old_cols),
        "dropped_columns": list(old_cols - new_cols),
        "dtype_mismatches": {
            c: {"old": str(df_old[c].dtype), "new": str(df_new[c].dtype)}
            for c in old_cols & new_cols if df_old[c].dtype != df_new[c].dtype
        }
    }


def data_quality_checks(df):
    checks = {}
    for c in df.columns:
        s = df[c]
        null_pct = float(s.isna().mean() * 100)
        zeros_pct = float((s == 0).mean() * 100) if np.issubdtype(s.dtype, np.number) else None
        unique = int(s.nunique(dropna=True))

        checks[c] = {
            "null_pct": round(null_pct, 3),
            "zeros_pct": round(zeros_pct, 3) if zeros_pct is not None else None,
            "unique": unique
        }
    return checks


def compute_feature_severity(old_s, new_s):
    stat, _ = ks_2samp(old_s, new_s)
    ks_score = float(stat)

    old_mean, new_mean = old_s.mean(), new_s.mean()
    old_std = old_s.std() or 1e-9
    mean_shift = abs(new_mean - old_mean) / old_std
    mean_score = min(mean_shift / 3, 1.0)

    combined = (SEVERITY_WEIGHT_KS * ks_score) + (SEVERITY_WEIGHT_MEAN * mean_score)
    return float(min(max(combined * 100, 0.0), 100.0))


def compute_overall_severity(scores):
    if not scores:
        return 0.0
    return float(np.mean(list(scores.values())))


def save_artifact(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".json":
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
    elif path.suffix in (".parquet", ".pq"):
        obj.to_parquet(path)
    else:
        with open(path, "wb") as f:
            dill.dump(obj, f)


# =========================================================
# MAIN
# =========================================================

def main():
    logger.info("=== Drift Detection Started ===")

    # ensure files exist
    for p in [OLD_FILE, NEW_FILE, MODEL_PATH]:
        if not os.path.exists(p):
            logger.error(f"Required file missing: {p}")
            return

    # load data
    df_old_raw = pd.read_csv(OLD_FILE)
    df_new_raw = pd.read_csv(NEW_FILE)

    # schema checks
    schema = schema_validation(df_old_raw, df_new_raw)
    save_artifact(schema, DRIFT_REPORT_PATH)

    # clean like training
    df_old_clean = clean_like_training(df_old_raw)
    df_new_clean = clean_like_training(df_new_raw)

    # load model
    pipeline, saved = safe_load_pipeline(MODEL_PATH)
    preproc = get_preprocess_pipeline(pipeline)

    # transform
    df_old_trans = transform_to_df(preproc, df_old_clean)
    df_new_trans = transform_to_df(preproc, df_new_clean)

    # feature drift
    features = df_old_trans.columns.tolist()
    drift_results = {}
    severities = {}

    for feat in features:
        s_old = df_old_trans[feat].dropna()
        s_new = df_new_trans[feat].dropna()

        if len(s_old) < MIN_ROWS_REQUIRED or len(s_new) < MIN_ROWS_REQUIRED:
            continue

        stat, p = ks_2samp(s_old, s_new)
        severity = compute_feature_severity(s_old, s_new)

        drift_results[feat] = {
            "p_value": float(p),
            "ks_stat": float(stat),
            "severity": float(severity),
            "drift": p < DRIFT_P_THRESHOLD and stat > DRIFT_KS_STAT_THRESHOLD
        }

        severities[feat] = severity

    # calculate overall
    overall_sev = compute_overall_severity(severities)
    logger.info(f"Overall severity = {overall_sev}")

    # Save drift report
    full_report = {
        "timestamp": now_str(),
        "overall_severity": overall_sev,
        "features": drift_results
    }
    save_artifact(full_report, DRIFT_REPORT_PATH)

    # retrain trigger
    drift_triggered = overall_sev >= OVERALL_SEVERITY_TRIGGER or any(v["drift"] for v in drift_results.values())

    if drift_triggered:
        logger.warning("🚨 Drift detected — triggering retrain")
        logger.info(f"Running: {RETRAIN_CMD}")

        result = subprocess.run(RETRAIN_CMD, capture_output=True, text=True)
        logger.info(result.stdout)
        logger.error(result.stderr)

    else:
        logger.info("No drift detected")

    return full_report


if __name__ == "__main__":
    main()
