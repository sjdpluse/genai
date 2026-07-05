"""
config.py — تنظیمات ApexTrade Pro v3.0
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ─── Supabase ───────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# ─── Binance ──────────────────────────────────────────────
BINANCE_BASE = "https://api.binance.com/api/v3"
BINANCE_FAPI = "https://fapi.binance.com/fapi/v1"

# ─── Yahoo Finance (Macro) ─────────────────────────────────
YAHOO_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"

# ─── امنیت ────────────────────────────────────────────────
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

# ─── تنظیمات سیگنال ───────────────────────────────────────
SYMBOL = "ETHUSDT"
INTERVALS = ["1h", "4h", "1d"]
CANDLE_LIMIT = 1000
TRAIN_LIMIT = 8000

# Triple Barrier Labeling
LABEL_FORWARD_CANDLES = 12
LABEL_LONG_THRESHOLD = 0.018
LABEL_SHORT_THRESHOLD = -0.018
LABEL_TIME_BARRIER = 24
LABEL_VOL_BARRIER_MULT = 2.0

# حداقل confidence
MIN_CONFIDENCE = 65

# ─── Risk Management ──────────────────────────────────────
ATR_SL_MULTIPLIER = 2.0
ATR_TP1_MULTIPLIER = 3.5
ATR_TP2_MULTIPLIER = 6.0
MIN_RISK_REWARD = 1.5

MAX_PENDING_SIGNALS = 1

# ─── فیلترهای پیشرفته ─────────────────────────────────────
VOLATILITY_FILTER_MAX = 0.05
VOLUME_FILTER_MIN = 0.7
TREND_FILTER_ADX_MIN = 25.0
TREND_FILTER_EMA_ALIGN = True
SESSION_FILTER_NIGHT_START = 2
SESSION_FILTER_NIGHT_END = 6
FUNDING_RATE_THRESHOLD = 0.0001
MIN_CONFLUENCE_INDICATORS = 3

# ─── مسیر مدل ─────────────────────────────────────────────
MODEL_PATH = f"model/signal_model_{SYMBOL}.joblib"
SCALER_PATH = f"model/scaler_{SYMBOL}.joblib"

# Sanity Check
_max_possible_rr = ATR_TP1_MULTIPLIER / ATR_SL_MULTIPLIER
assert MIN_RISK_REWARD <= _max_possible_rr, (
    f"پیکربندی نامعتبر: MIN_RISK_REWARD={MIN_RISK_REWARD} > {_max_possible_rr:.3f}"
)
