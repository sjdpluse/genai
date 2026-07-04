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

# ─── امنیت Endpoint های حساس (Admin) ─────────────────────────
# اگر تنظیم شود، endpoint های /train و /train-sync به هدر
# X-Admin-Token نیاز خواهند داشت (جلوگیری از DoS/سوءاستفادهٔ عمومی،
# چون آموزش مدل عملیات سنگین CPU است و روی Railway هزینه دارد).
# ⚠️ در Railway حتماً این متغیر محیطی را ست کنید؛ اگر خالی بماند،
# این endpoint ها برای همه باز می‌مانند (سازگاری با نسخهٔ قبلی).
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

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

# حداقل نسبت ریسک‌به‌ریوارد قابل‌قبول برای صدور سیگنال LONG/SHORT
#
# ⚠️ نکته مهم: این مقدار باید <= (ATR_TP1_MULTIPLIER / ATR_SL_MULTIPLIER) باشد،
# وگرنه هیچ سیگنالی هرگز از فیلتر ریسک/ریوارد عبور نمی‌کند (این دقیقاً همان
# باگی بود که باعث می‌شد سیستم همیشه WAIT برگرداند: قبلاً min_rr=1.8 هاردکد
# شده بود در حالی‌که TP1/SL = 3.0/1.8 = 1.667 است، یعنی همیشه کمتر از حد مجاز).
#
# نسبت واقعی فعلی برای TP1 = 3.0/1.8 ≈ 1.667  → این مقدار باید کمی زیر آن باشد.
MIN_RISK_REWARD = 1.5

# ─── جلوگیری از Overtrading ───────────────────────────────────
# حداکثر تعداد سیگنال «pending» هم‌زمان که مجاز به ذخیره در دیتابیس است.
# با ۱، سیستم تا وقتی معاملهٔ باز قبلی به نتیجه (win/loss/expired) نرسیده،
# سیگنال جدید LONG/SHORT صادر نمی‌کند — مدیریت ریسک روی تعداد معاملات
# اولویت بالاتری از فرکانس سیگنال دارد.
MAX_PENDING_SIGNALS = 1

# ─── مسیر ذخیره مدل ─────────────────────────────────────────
MODEL_PATH  = "model/signal_model.joblib"
SCALER_PATH = "model/scaler.joblib"

# ─── قفل ایمنی پیکربندی (Config Sanity Guard) ────────────────
# این باگ قبلاً یک‌بار واقعاً رخ داده بود: اگر MIN_RISK_REWARD به اشتباه
# بالاتر از نسبت واقعی TP1/SL تنظیم شود، هیچ سیگنال LONG/SHORT ای هرگز
# از فیلتر _risk_reward_ok عبور نمی‌کند و سیستم برای همیشه فقط WAIT
# برمی‌گرداند — بدون هیچ خطای صریحی، فقط به‌صورت خاموش. به‌جای این‌که
# منتظر بمانیم این باگ دوباره در آینده (مثلاً بعد از تغییر ضرایب ATR)
# به‌صورت ساکت برگردد، همین‌جا در زمان import، صراحتاً fail می‌کنیم.
_max_possible_rr = ATR_TP1_MULTIPLIER / ATR_SL_MULTIPLIER
assert MIN_RISK_REWARD <= _max_possible_rr, (
    f"پیکربندی نامعتبر: MIN_RISK_REWARD={MIN_RISK_REWARD} بزرگ‌تر از حداکثر "
    f"نسبت ممکن ATR_TP1_MULTIPLIER/ATR_SL_MULTIPLIER={_max_possible_rr:.3f} است. "
    f"با این تنظیمات هیچ سیگنال LONG/SHORT ای هرگز صادر نخواهد شد. "
    f"یا MIN_RISK_REWARD را کاهش دهید یا ATR_TP1_MULTIPLIER را افزایش دهید."
)
