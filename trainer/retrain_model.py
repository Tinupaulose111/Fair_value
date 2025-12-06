import sys
import os
import json
import shutil
import dill
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_percentage_error
)

# =========================================================
# 1️⃣ Add project root to Python path
# =========================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))          # /trainer
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))   # /app

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# =========================================================
# 2️⃣ Import custom modules
# =========================================================
from core.custom_transformers import RowFilter, FeatureEngineer, Preprocessor
from core.metrics_logger import log_metrics

print("\n🔧 Starting Model RETRAIN Process...")
print("-" * 60)

# =========================================================
# 3️⃣ Paths (✔ FIXED for Docker)
# =========================================================
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATA_PATH = os.path.join(DATA_DIR, "Marketdata_newdata_update.csv")

MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "final_model_fairvalue.pkl")
BACKUP_DIR = os.path.join(PROJECT_ROOT, "model_backups")

os.makedirs(BACKUP_DIR, exist_ok=True)

# =========================================================
# 4️⃣ Safety Checks
# =========================================================
if not os.path.exists(DATA_PATH):
    print(f"❌ Data file not found: {DATA_PATH}")
    exit()

if not os.path.exists(MODEL_PATH):
    print(f"❌ Original model not found: {MODEL_PATH}")
    exit()

# =========================================================
# 5️⃣ Load Existing Model
# =========================================================
try:
    with open(MODEL_PATH, "rb") as f:
        saved = dill.load(f)

    pipeline = saved["pipeline"]
    old_smearing = float(saved["smearing_factor"])

    print("✅ Base model loaded successfully.")

except Exception as e:
    print(f"❌ Error loading existing model: {e}")
    exit()

# =========================================================
# 6️⃣ Load & Preprocess New Data
# =========================================================
print("\n📥 Loading new training data...")
df_new = pd.read_csv(DATA_PATH)

# Financial target calculation
df_new["Intrinsic_value"] = np.sqrt(22.5 * df_new["EPS"] * df_new["Book Value"])
df_new["Intrinsic_value_log"] = np.log1p(df_new["Intrinsic_value"])

# Clean data
df_new = (
    df_new[(df_new["EPS"] > 0) & (df_new["Book Value"] > 0)]
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
)

FEATURES = [
    "EPS", "Book Value", "Price", "Net Income", "Shareholder Equity",
    "Operating Income", "Revenue", "Total Assets", "Total Liabilities",
    "Free Cash Flow", "Dividend per Share"
]

if not all(col in df_new.columns for col in FEATURES):
    missing = [c for c in FEATURES if c not in df_new.columns]
    print(f"⚠ Missing columns: {missing}")
    exit()

X = df_new[FEATURES]
y_log = df_new["Intrinsic_value_log"]
y = df_new["Intrinsic_value"]

# =========================================================
# 7️⃣ Train/Test Split
# =========================================================
X_train, X_test, y_train_log, y_test_log, y_train, y_test = train_test_split(
    X, y_log, y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# 8️⃣ Retrain Model
# =========================================================
print("\n🔁 Retraining model with new data...")
pipeline.fit(X_train, y_train_log)

# New smearing factor
new_smearing = float(np.mean(np.exp(y_train_log - pipeline.predict(X_train))))
print(f"📊 New Smearing Factor: {new_smearing:.5f}")

# =========================================================
# 9️⃣ Evaluate on Test Split
# =========================================================
y_pred_test = np.expm1(pipeline.predict(X_test)) * new_smearing

r2 = r2_score(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100

print("\n📊 Evaluation on Test Data")
print(f"   R²   : {r2:.4f}")
print(f"   RMSE : {rmse:.2f}")
print(f"   MAPE : {mape:.2f}%")

# =========================================================
# 🔟 Backup the Old Model
# =========================================================
timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
backup_path = os.path.join(BACKUP_DIR, f"model_backup_{timestamp}.pkl")

shutil.copy(MODEL_PATH, backup_path)
print(f"\n🗂 Old model backed up at: {backup_path}")

# =========================================================
# 1️⃣1️⃣ Save New Model
# =========================================================
with open(MODEL_PATH, "wb") as f:
    dill.dump({"pipeline": pipeline, "smearing_factor": new_smearing}, f)

# Versioned copy
version_count = len([f for f in os.listdir(BACKUP_DIR) if f.endswith(".pkl")])
versioned_path = os.path.join(BACKUP_DIR, f"model_v{version_count}.pkl")
shutil.copy(MODEL_PATH, versioned_path)

print(f"💾 New model saved to: {MODEL_PATH}")
print(f"🧩 Versioned copy saved as: {versioned_path}")

# =========================================================
# 1️⃣2️⃣ Log Metrics
# =========================================================
metrics = {
    "r2": r2,
    "rmse": rmse,
    "mape": mape,
    "new_smearing": new_smearing,
    "old_smearing": old_smearing,
    "data_rows_used": len(df_new),
    "timestamp": timestamp,
    "version": f"v{version_count}"
}

with open(os.path.join(PROJECT_ROOT, "model_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

log_metrics(metrics)

print("\n✅ Retraining Complete!")
print("-" * 60)
