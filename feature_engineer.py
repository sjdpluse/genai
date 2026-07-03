"""
feature_engineer.py — محاسبه اندیکاتورهای تکنیکال و ساخت Feature Matrix
از کتابخانه `ta` استفاده می‌کند — دقیق‌تر از محاسبات JS دستی
"""
import pandas as pd
import numpy as np
import ta
import logging

logger = logging.getLogger(__name__)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    محاسبه تمام اندیکاتورها روی DataFrame کندل‌ها

    Input:  DataFrame با ستون‌های open, high, low, close, volume
    Output: همان DataFrame به علاوه ستون‌های اندیکاتور

    ⚠️ این تابع کاملاً causal است: هر مقدار در ردیف t فقط از داده‌های
    ردیف‌های <= t استفاده می‌کند (rolling/ewm در pandas و `ta` به‌صورت
    پیش‌فرض trailing هستند، نه centered) — پس lookahead bias در سطح
    تک‌تایم‌فریم وجود ندارد.
    """
    df = df.copy()
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    # ─── روند (Trend) ────────────────────────────────────────
    df["ema_20"]  = ta.trend.EMAIndicator(c, 20).ema_indicator()
    df["ema_50"]  = ta.trend.EMAIndicator(c, 50).ema_indicator()
    df["ema_200"] = ta.trend.EMAIndicator(c, 200).ema_indicator()
    df["sma_20"]  = ta.trend.SMAIndicator(c, 20).sma_indicator()

    # MACD
    macd_obj = ta.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
    df["macd"]        = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()
    df["macd_hist"]   = macd_obj.macd_diff()

    # ADX
    adx_obj = ta.trend.ADXIndicator(h, l, c, 14)
    df["adx"]      = adx_obj.adx()
    df["adx_pos"]  = adx_obj.adx_pos()
    df["adx_neg"]  = adx_obj.adx_neg()

    # Ichimoku — به‌جای این‌که فقط محاسبه و دور ریخته شود، به یک ویژگی
    # نرمال (موقعیت قیمت نسبت به ابر ایچیموکو) تبدیل می‌شود؛ پایین‌تر
    # در price_vs_ichi_a / ichi_cloud_diff استفاده می‌شود.
    ichi = ta.trend.IchimokuIndicator(h, l)
    df["ichi_a"] = ichi.ichimoku_a()
    df["ichi_b"] = ichi.ichimoku_b()

    # ─── نوسان (Momentum) ───────────────────────────────────
    df["rsi"]     = ta.momentum.RSIIndicator(c, 14).rsi()
    df["rsi_3"]   = ta.momentum.RSIIndicator(c, 3).rsi()   # RSI کوتاه برای سیگنال‌های سریع

    stoch = ta.momentum.StochasticOscillator(h, l, c, 14, 3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    df["cci"] = ta.trend.CCIIndicator(h, l, c, 20).cci()
    df["mfi"] = ta.volume.MFIIndicator(h, l, c, v, 14).money_flow_index()
    df["williams_r"] = ta.momentum.WilliamsRIndicator(h, l, c, 14).williams_r()

    # ROC — نرخ تغییر قیمت
    df["roc_5"]  = ta.momentum.ROCIndicator(c, 5).roc()
    df["roc_20"] = ta.momentum.ROCIndicator(c, 20).roc()

    # ─── نوسان‌پذیری (Volatility) ────────────────────────────
    bb = ta.volatility.BollingerBands(c, 20, 2)
    df["bb_upper"]  = bb.bollinger_hband()
    df["bb_lower"]  = bb.bollinger_lband()
    df["bb_mid"]    = bb.bollinger_mavg()
    df["bb_width"]  = bb.bollinger_wband()   # پهنای باند — نشانه volatility
    df["bb_pct"]    = bb.bollinger_pband()   # موقعیت قیمت در باند (۰=کف، ۱=سقف)

    df["atr"]  = ta.volatility.AverageTrueRange(h, l, c, 14).average_true_range()
    df["atr_pct"] = df["atr"] / c            # ATR به‌صورت درصد قیمت

    # ─── حجم (Volume) ───────────────────────────────────────
    # ⚠️ OBV به‌صورت ذاتی یک مجموع تجمعی (cumulative sum) از ابتدای
    # سری داده است. این یعنی مقدار خامش کاملاً به تعداد کندل‌های
    # fetch‌شده وابسته است: در آموزش با ۵۰۰۰ کندل مقیاسش ~۸ برابر
    # حالت inference (۶۰۰ کندل) است. اگر مقدار خام OBV مستقیماً به
    # مدل داده شود، مدل روی مقیاسی آموزش می‌بیند که در inference هرگز
    # تکرار نمی‌شود (train/serve skew) — این یک باگ واقعی صحت داده بود.
    # راه‌حل: از z-score غلتان OBV استفاده می‌کنیم که ایستا (stationary)
    # و مستقل از طول تاریخچهٔ دریافت‌شده است.
    obv_raw = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
    obv_mean = obv_raw.rolling(50).mean()
    obv_std  = obv_raw.rolling(50).std()
    df["obv_z"] = (obv_raw - obv_mean) / obv_std.replace(0, np.nan)

    df["vwap"] = ta.volume.VolumeWeightedAveragePrice(h, l, c, v).volume_weighted_average_price()

    # نسبت حجم نسبت به میانگین ۲۰ کندل
    df["vol_ratio"] = v / v.rolling(20).mean()

    # ─── ویژگی‌های ترکیبی (engineered features) ─────────────
    # فاصله قیمت از EMAها (به‌صورت درصد)
    df["price_vs_ema20"]  = (c - df["ema_20"])  / df["ema_20"]
    df["price_vs_ema50"]  = (c - df["ema_50"])  / df["ema_50"]
    df["price_vs_ema200"] = (c - df["ema_200"]) / df["ema_200"]
    df["ema20_vs_ema50"]  = (df["ema_20"] - df["ema_50"]) / df["ema_50"]

    # شیب EMA (momentum روند)
    df["ema20_slope"]  = df["ema_20"].diff(3)  / df["ema_20"].shift(3)
    df["ema50_slope"]  = df["ema_50"].diff(5)  / df["ema_50"].shift(5)

    # کندل‌شناسی ساده
    df["candle_body"]  = (c - df["open"]).abs() / df["atr"]
    df["candle_range"] = (h - l) / df["atr"]
    df["upper_wick"]   = (h - pd.concat([c, df["open"]], axis=1).max(axis=1)) / df["atr"]
    df["lower_wick"]   = (pd.concat([c, df["open"]], axis=1).min(axis=1) - l) / df["atr"]

    # High/Low نسبی
    df["dist_to_high20"] = (h.rolling(20).max() - c) / c
    df["dist_to_low20"]  = (c - l.rolling(20).min()) / c

    # موقعیت قیمت نسبت به ابر ایچیموکو (فیلتر روند) + قیمت نسبت به VWAP
    df["price_vs_ichi_a"] = (c - df["ichi_a"]) / c
    df["ichi_cloud_diff"] = (df["ichi_a"] - df["ichi_b"]) / c   # + یعنی ابر صعودی
    df["price_vs_vwap"]   = (c - df["vwap"]) / df["vwap"]

    return df


def build_feature_matrix(df_1h: pd.DataFrame,
                          df_4h: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    ساخت ماتریس ویژگی نهایی با ادغام چند تایم‌فریم

    Args:
        df_1h: داده ۱ ساعته (خام یا از قبل با compute_indicators پردازش‌شده)
        df_4h: داده ۴ ساعته (خام یا از قبل پردازش‌شده) — اختیاری

    Returns:
        DataFrame آماده برای ورود به مدل ML
    """
    # اگر ورودی از قبل توسط compute_indicators پردازش شده (ستون rsi موجود
    # است)، دوباره محاسبه نمی‌کنیم — از محاسبات تکراری/هدررفت CPU
    # جلوگیری می‌کند (این تابع هم از مسیر آموزش و هم مسیر inference
    # صدا زده می‌شود؛ حالا هر دو می‌توانند یک df از قبل محاسبه‌شده را
    # به اشتراک بگذارند).
    if "rsi" not in df_1h.columns:
        df_1h = compute_indicators(df_1h)
    else:
        df_1h = df_1h.copy()

    # ستون‌های ویژگی نهایی (۱H)
    feature_cols = [
        # 1H indicators
        "rsi", "rsi_3", "stoch_k", "stoch_d", "cci", "mfi", "williams_r",
        "macd_hist", "adx", "adx_pos", "adx_neg",
        "bb_pct", "bb_width", "atr_pct",
        "obv_z", "vol_ratio",
        "price_vs_ema20", "price_vs_ema50", "price_vs_ema200",
        "ema20_vs_ema50", "ema20_slope", "ema50_slope",
        "candle_body", "candle_range", "upper_wick", "lower_wick",
        "dist_to_high20", "dist_to_low20",
        "roc_5", "roc_20",
        "price_vs_ichi_a", "ichi_cloud_diff", "price_vs_vwap",
    ]

    # اگر ۴H داریم، اندیکاتورهای کلیدی آن را اضافه می‌کنیم
    if df_4h is not None:
        if "rsi" not in df_4h.columns:
            df_4h = compute_indicators(df_4h)
        else:
            df_4h = df_4h.copy()

        # ─── رفع باگ Look-Ahead Bias در ادغام چند تایم‌فریم ─────
        # اندیس هر ردیف df_4h زمان «بازشدن» کندل ۴H است، اما مقادیر آن
        # ردیف (rsi، macd_hist و ...) بر اساس close همان کندل محاسبه
        # شده‌اند که فقط در «زمان بسته‌شدن» (۴ ساعت بعد) مشخص می‌شود.
        # قبلاً این ردیف مستقیماً روی ردیف‌های ۱H هم‌بازه (که هنوز به
        # زمان بسته‌شدنش نرسیده بودند) forward-fill می‌شد → یعنی مدل در
        # آموزش به دادهٔ آینده دسترسی داشت (classic multi-timeframe
        # leakage). با shift(1) مقدار هر کندل ۴H را به زمان *بسته‌شدنش*
        # منتقل می‌کنیم، بعد ffill می‌کنیم — یعنی هر ردیف ۱H فقط به
        # آخرین کندل ۴H که واقعاً تا آن لحظه بسته شده دسترسی دارد.
        cols_4h = ["rsi", "adx", "macd_hist", "ema20_vs_ema50",
                   "bb_pct", "atr_pct", "price_vs_ema200", "obv_z"]
        df_4h_shifted    = df_4h[cols_4h].shift(1)
        df_4h_resampled  = df_4h_shifted.reindex(df_1h.index, method="ffill")
        df_4h_resampled.columns = [f"4h_{c}" for c in cols_4h]

        df_1h = pd.concat([df_1h, df_4h_resampled], axis=1)
        feature_cols += [c for c in df_1h.columns if c.startswith("4h_")]

    # اطمینان از وجود تمام ستون‌ها — ستون‌های گمشده با NaN پر می‌شوند
    for c in feature_cols:
        if c not in df_1h.columns:
            df_1h[c] = np.nan

    return df_1h[feature_cols].copy()


def create_labels(df_1h: pd.DataFrame,
                  forward_candles: int = 8,
                  long_threshold: float = 0.015,
                  short_threshold: float = -0.015) -> pd.Series:
    """
    برچسب‌گذاری: نگاه به آینده
    اگر قیمت در N کندل آینده بیش از long_threshold رشد کرد → 1 (LONG)
    اگر افت کرد → -1 (SHORT)
    وگرنه → 0 (WAIT)

    ⚠️ رفع باگ: نسخهٔ قبلی برچسب را با pd.Series(0, ...) مقداردهی اولیه
    می‌کرد و هرگز NaN تولید نمی‌کرد — یعنی N کندل آخر که آیندهٔ واقعی‌شان
    اصلاً مشخص نیست (به‌خاطر shift(-forward_candles))، به‌اشتباه با
    برچسب WAIT وارد آموزش می‌شدند (چون NaN >= threshold همیشه False
    ارزیابی می‌شود). این باعث می‌شد مدل با برچسب‌های نادرست/جعلی برای
    جدیدترین کندل‌های هر دورهٔ آموزش تغذیه شود. حالا این ردیف‌ها صراحتاً
    NaN می‌شوند تا با y.dropna() به‌درستی از آموزش حذف شوند.
    """
    future_return = df_1h["close"].shift(-forward_candles) / df_1h["close"] - 1

    labels = pd.Series(0.0, index=df_1h.index, name="label")
    labels[future_return >= long_threshold]  = 1.0   # LONG
    labels[future_return <= short_threshold] = -1.0  # SHORT
    labels[future_return.isna()] = np.nan             # آیندهٔ نامعلوم → حذف از آموزش

    return labels
