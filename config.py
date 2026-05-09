"""
config.py — تنظیمات مرکزی پروژه
تمام مقادیر حساس از .env خوانده می‌شوند
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ─── Supabase ───────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")   # service_role key — نه anon key

# ─── Binance (برای داده‌های عمومی نیازی به API key نیست) ─────
BINANCE_BASE = "https://api.binance.com/api/v3"

# ─── تنظیمات سیگنال ─────────────────────────────────────────
SYMBOL         = "BTCUSDT"
INTERVALS      = ["1h", "4h"]          # تایم‌فریم‌ها
CANDLE_LIMIT   = 1000                  # تعداد کندل برای fetch
TRAIN_LIMIT    = 5000                  # کندل برای آموزش مدل (حدود ۷ ماه 1h)

# آستانه بازده آینده برای برچسب‌گذاری (label)
LABEL_FORWARD_CANDLES = 8             # ۸ کندل جلوتر = ۸ ساعت
LABEL_LONG_THRESHOLD  = 0.015         # +۱.۵٪ → LONG
LABEL_SHORT_THRESHOLD = -0.015        # -۱.۵٪ → SHORT

# حداقل confidence برای صدور سیگنال
MIN_CONFIDENCE = 55                    # درصد — زیر این → WAIT

# ─── تنظیمات Risk Management ─────────────────────────────────
ATR_SL_MULTIPLIER  = 1.8
ATR_TP1_MULTIPLIER = 3.0
ATR_TP2_MULTIPLIER = 5.0

# ─── مسیر ذخیره مدل ─────────────────────────────────────────
MODEL_PATH  = "model/signal_model.joblib"
SCALER_PATH = "model/scaler.joblib"
