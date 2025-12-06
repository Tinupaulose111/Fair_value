# metrics_logger.py
"""
Enterprise-ready metrics logger.
- Writes a human-readable latest snapshot (metrics/latest.json)
- Appends every run to a JSONL history file (metrics/history.jsonl)
- Thread-safe, UTC-aware, numpy-safe.
"""

import os
import json
import threading
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
METRICS_DIR = Path("metrics")
METRICS_SNAPSHOT = METRICS_DIR / "latest.json"
METRICS_LOG = METRICS_DIR / "history.jsonl"

# Ensure the folder exists
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# Thread-safe file access
_file_lock = threading.Lock()

# ----------------------------------------------------------------------
# Logging (optional – you can also import this logger elsewhere)
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("metrics_logger")

# ----------------------------------------------------------------------
# Helper: safe JSON default serializer
# ----------------------------------------------------------------------
def _json_default(obj: Any) -> Any:
    """Convert numpy / pandas scalars to native Python types."""
    if hasattr(obj, "item"):          # numpy scalar
        return obj.item()
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)                   # fallback


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def log_metrics(metrics: Dict[str, Any], notes: str = "") -> None:
    """
    Save the latest metrics snapshot and append the entry to the history log.

    Args:
        metrics: Dictionary with performance numbers (R², RMSE, …).
        notes:   Optional free-text note (e.g. "Drift detected in EPS").
    """
    # ------------------------------------------------------------------
    # 1. Build the enriched entry
    # ------------------------------------------------------------------
    entry = metrics.copy()
    entry["timestamp_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entry["timestamp_local"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry["notes"] = notes.strip()

    # ------------------------------------------------------------------
    # 2. Serialise once (re-used for both files)
    # ------------------------------------------------------------------
    json_line = json.dumps(entry, default=_json_default)

    with _file_lock:
        # ---- latest snapshot (pretty-printed) ----
        try:
            with open(METRICS_SNAPSHOT, "w", encoding="utf-8") as f:
                json.dump(entry, f, indent=2, default=_json_default)
            log.info(f"Metrics snapshot → {METRICS_SNAPSHOT}")
        except Exception as exc:
            log.error(f"Failed to write snapshot {METRICS_SNAPSHOT}: {exc}")

        # ---- append to history (JSONL) ----
        try:
            with open(METRICS_LOG, "a", encoding="utf-8") as f:
                f.write(json_line + "\n")
            log.info(f"Metrics appended → {METRICS_LOG}")
        except Exception as exc:
            log.error(f"Failed to append to history {METRICS_LOG}: {exc}")