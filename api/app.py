import sys
import os
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import dill
import yfinance as yf

# =========================================================
# 1️⃣ Add project root to PYTHONPATH (Works locally + Docker)
# =========================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # local: /.../api , docker: /app

# Detect docker environment
if CURRENT_DIR == "/app":
    PROJECT_ROOT = "/app"
else:
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# =========================================================
# 2️⃣ Import custom modules
# =========================================================
from core.custom_transformers import RowFilter, FeatureEngineer, Preprocessor
from core.metrics_logger import log_metrics

# =========================================================
# Flask Setup
# =========================================================
app = Flask(__name__, template_folder="templates")
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================================================
# Model Path
# =========================================================
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "final_model_fairvalue.pkl")

print("MODEL_PATH =", MODEL_PATH)
print("MODEL EXISTS? =", os.path.exists(MODEL_PATH))

pipeline = None
smearing_factor = 1.05

# =========================================================
# Load Model
# =========================================================
try:
    with open(MODEL_PATH, "rb") as f:
        loaded = dill.load(f)

    if isinstance(loaded, dict):
        pipeline = loaded["pipeline"]
        smearing_factor = float(loaded.get("smearing_factor", 1.05))
    else:
        pipeline = loaded

    logger.info("✅ Model loaded successfully.")

except Exception as e:
    logger.error(f"❌ Model load failed: {e}")

# =========================================================
# Helper Functions
# =========================================================

RAW_FEATURES = [
    'Price', 'EPS', 'Book Value', 'Dividend per Share',
    'Net Income', 'Shareholder Equity', 'Operating Income',
    'Revenue', 'Total Assets', 'Total Liabilities', 'Free Cash Flow'
]

def safe_get(obj, attr, default=None):
    try:
        val = getattr(obj, attr, None)
        return val() if callable(val) else val
    except Exception:
        return default

def get_stock_data(ticker: str) -> dict:
    """Pull financial data from Yahoo Finance."""
    stock = yf.Ticker(ticker)

    try:
        info = safe_get(stock, "info", {}) or {}
        fast_info = safe_get(stock, "fast_info", {}) or {}
        financials = safe_get(stock, "financials", pd.DataFrame())
        balance = safe_get(stock, "balance_sheet", pd.DataFrame())
        cashflow = safe_get(stock, "cashflow", pd.DataFrame())

        price = fast_info.get("lastPrice") or info.get("currentPrice") or 0
        dividend = info.get("dividendRate") or 0

        shareholder_equity = 0
        for k in [
            "Stockholders Equity", "Common Stock Equity",
            "Total Equity Gross Minority Interest"
        ]:
            if k in balance.index:
                shareholder_equity = balance.loc[k].iloc[0]
                break

        return {
            "Price": float(price),
            "EPS": float(info.get("trailingEps") or 0),
            "Book Value": float(info.get("bookValue") or 0),
            "Dividend per Share": float(dividend),
            "Net Income": float(financials.loc["Net Income"].iloc[0]) if "Net Income" in financials.index else 0,
            "Shareholder Equity": float(shareholder_equity),
            "Operating Income": float(financials.loc["Operating Income"].iloc[0]) if "Operating Income" in financials.index else 0,
            "Revenue": float(financials.loc["Total Revenue"].iloc[0]) if "Total Revenue" in financials.index else 0,
            "Total Assets": float(balance.loc["Total Assets"].iloc[0]) if "Total Assets" in balance.index else 0,
            "Total Liabilities": float(balance.loc["Total Liabilities Net Minority Interest"].iloc[0]) if "Total Liabilities Net Minority Interest" in balance.index else 0,
            "Free Cash Flow": float(cashflow.loc["Free Cash Flow"].iloc[0]) if "Free Cash Flow" in cashflow.index else 0,
        }

    except Exception as e:
        logger.warning(f"⚠️ Yahoo Finance Error: {e}")
        return {k: 0.0 for k in RAW_FEATURES}

def compute_ratios(data: dict) -> dict:
    """Financial ratio calculations."""
    try:
        P = data["Price"]
        E = data["EPS"]
        BV = data["Book Value"]
        NI = data["Net Income"]
        RE = data["Revenue"]
        SE = data["Shareholder Equity"]
        OI = data["Operating Income"]
        TA = data["Total Assets"]
        TL = data["Total Liabilities"]
        FCF = data["Free Cash Flow"]

        return {
            "P/E Ratio": P / E if E else 0,
            "P/B Ratio": P / BV if BV else 0,
            "ROE": NI / SE if SE else 0,
            "ROA": NI / TA if TA else 0,
            "Debt/Equity": TL / SE if SE else 0,
            "Operating Margin": OI / RE if RE else 0,
            "FCF Yield": FCF / (P * 1e6) if P else 0,
        }

    except Exception:
        return {}

def predict_back_transformed(df_row: pd.DataFrame) -> float:
    if pipeline is None:
        raise RuntimeError("Model not loaded")

    pred_log = pipeline.predict(df_row)[0]
    return float(np.expm1(pred_log) * smearing_factor)

# =========================================================
# Routes
# =========================================================

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    ticker = None
    data = {}
    ratios = {}
    comparison = {}

    if request.method == "POST":
        try:
            ticker = request.form.get("Ticker", "").strip().upper()

            if ticker:
                data = get_stock_data(ticker)
                df = pd.DataFrame([data])[RAW_FEATURES]
                current_price = data["Price"]

                fair_value = predict_back_transformed(df)
                prediction = round(fair_value, 2)
                ratios = compute_ratios(data)

                diff_percent = ((current_price - prediction) / prediction) * 100

                if diff_percent < 0:
                    comparison = {"status": "Undervalued (Buy)", "color": "green"}
                elif diff_percent <= 20:
                    comparison = {"status": "Fairly Priced (Hold)", "color": "orange"}
                else:
                    comparison = {"status": "Overvalued (Sell)", "color": "red"}

        except Exception as e:
            logger.exception("Prediction error:")
            error = str(e)

    return render_template(
        "index.html",
        features=RAW_FEATURES,
        prediction=prediction,
        error=error,
        ticker=ticker,
        data=data,
        ratios=ratios,
        comparison=comparison,
        current_price=data.get("Price")
    )

@app.route("/health")
def health():
    return jsonify({"status": "healthy", "model_loaded": pipeline is not None})

# =========================================================
# Run App
# =========================================================
if __name__ == "__main__":
    app.run(debug=True)
