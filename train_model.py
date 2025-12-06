import sys
import os

# -----------------------------
# 1. Add project root to PYTHONPATH
# -----------------------------
if "__file__" in globals():
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    CURRENT_DIR = os.getcwd()

PROJECT_ROOT = os.path.abspath(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# -----------------------------
# 2. Import transformers
# -----------------------------
from core.custom_transformers import RowFilter, FeatureEngineer, Preprocessor

import pandas as pd
import numpy as np
import dill   # <--- use dill, NOT pickle!
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

# -----------------------------
# 3. Load dataset
# -----------------------------
df = pd.read_csv('Marketdata_newdata.csv')
df['Intrinsic_value'] = np.sqrt(22.5 * df['EPS'] * df['Book Value'])
df['Intrinsic_value_log'] = np.log1p(df['Intrinsic_value'])

df_filtered = df[(df['EPS'] > 0) & (df['Book Value'] > 0)]
df_filtered = df_filtered.replace([np.inf, -np.inf], np.nan).dropna()

initial_features = [
    'EPS', 'Book Value', 'Price', 'Net Income', 'Shareholder Equity',
    'Operating Income', 'Revenue', 'Total Assets', 'Total Liabilities',
    'Free Cash Flow', 'Dividend per Share'
]

X = df_filtered[initial_features]
y_log = df_filtered['Intrinsic_value_log']
y = df_filtered['Intrinsic_value']

selected_features = [
    'Dividend per Share_power', 'ROE_power', 'Operating_Margin_power', 'Net_Profit_Margin_power',
    'FCF_Torevenue_power', 'Cashflow_pershare_power', 'EV_to_sales_power', 'BP_x_EarningsYield_log',
    'Book Value_log', 'ROA_power', 'Debt_to_Equity_log'
]

# -----------------------------
# 4. Build pipeline
# -----------------------------
pipeline = Pipeline([
    ('row_filter', RowFilter()),
    ('feature_engineer', FeatureEngineer()),
    ('preprocessor', Preprocessor(selected_features)),
    ('model', XGBRegressor(
        n_estimators=400,
        learning_rate=0.04,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=1.0,
        random_state=42
    ))
])

# -----------------------------
# 5. K-fold cross validation
# -----------------------------
kf = KFold(n_splits=9, shuffle=True, random_state=42)

train_r2_scores, val_r2_scores = [], []
val_rmse_scores, val_mape_scores = [], []
smearing_factors = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train_log, y_val_log = y_log.iloc[train_idx], y_log.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    pipeline.fit(X_train, y_train_log)

    y_train_pred_log = pipeline.predict(X_train)
    y_val_pred_log = pipeline.predict(X_val)

    smearing_factor = float(np.mean(np.exp(y_train_log - y_train_pred_log)))
    smearing_factors.append(smearing_factor)

    y_train_pred = np.expm1(y_train_pred_log) * smearing_factor
    y_val_pred = np.expm1(y_val_pred_log) * smearing_factor

    train_r2_scores.append(r2_score(y_train, y_train_pred))
    val_r2_scores.append(r2_score(y_val, y_val_pred))
    val_rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_val_pred)))
    val_mape_scores.append(mean_absolute_percentage_error(y_val, y_val_pred) * 100)

    print(f"\n===== Fold {fold} Results =====")
    print(f"Train R²: {train_r2_scores[-1]:.4f} | Validation R²: {val_r2_scores[-1]:.4f}")
    print(f"Validation RMSE: {val_rmse_scores[-1]:.2f}")
    print(f"Validation MAPE: {val_mape_scores[-1]:.2f}%")
    print(f"Smearing Factor: {smearing_factor:.4f}")
    print("===================================")

# -----------------------------
# 6. Fit final model
# -----------------------------
pipeline.fit(X, y_log)
final_smearing_factor = float(np.mean(np.exp(y_log - pipeline.predict(X))))

# -----------------------------
# 7. Save final model
# -----------------------------
model_path = os.path.join("model", "final_model_fairvalue.pkl")
os.makedirs("model", exist_ok=True)

with open(model_path, "wb") as f:
    dill.dump({'pipeline': pipeline, 'smearing_factor': final_smearing_factor}, f)

print("\n✅ Final model trained & saved successfully!")
print(f"Saved at: {model_path}")
print(f"Final Smearing Factor: {final_smearing_factor:.4f}")
