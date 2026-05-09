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

    # Ichimoku
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
    df["obv"]  = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
    df["vwap"] = ta.volume.VolumeWeightedAveragePriceIndicator(h, l, c, v).volume_weighted_average_price()

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

    return df


def build_feature_matrix(df_1h: pd.DataFrame,
                          df_4h: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    ساخت ماتریس ویژگی نهایی با ادغام چند تایم‌فریم

    Args:
        df_1h: داده ۱ ساعته با اندیکاتورها
        df_4h: داده ۴ ساعته با اندیکاتورها (اختیاری — context روند بزرگ‌تر)

    Returns:
        DataFrame آماده برای ورود به مدل ML
    """
    df_1h = compute_indicators(df_1h)

    # اگر ۴H داریم، اندیکاتورهای کلیدی آن را اضافه می‌کنیم
    if df_4h is not None:
        df_4h = compute_indicators(df_4h)

        # ریسمپل ۴H به ۱H (forward-fill) تا بتوانیم merge کنیم
        cols_4h = ["rsi", "adx", "macd_hist", "ema20_vs_ema50",
                   "bb_pct", "atr_pct", "price_vs_ema200"]
        df_4h_resampled = df_4h[cols_4h].reindex(df_1h.index, method="ffill")
        df_4h_resampled.columns = [f"4h_{c}" for c in cols_4h]

        df_1h = pd.concat([df_1h, df_4h_resampled], axis=1)

    # ستون‌های ویژگی نهایی
    feature_cols = [
        # 1H indicators
        "rsi", "rsi_3", "stoch_k", "stoch_d", "cci", "mfi", "williams_r",
        "macd_hist", "adx", "adx_pos", "adx_neg",
        "bb_pct", "bb_width", "atr_pct",
        "obv", "vol_ratio",
        "price_vs_ema20", "price_vs_ema50", "price_vs_ema200",
        "ema20_vs_ema50", "ema20_slope", "ema50_slope",
        "candle_body", "candle_range", "upper_wick", "lower_wick",
        "dist_to_high20", "dist_to_low20",
        "roc_5", "roc_20",
    ]

    # اضافه کردن ویژگی‌های ۴H اگر وجود داشتند
    if df_4h is not None:
        feature_cols += [c for c in df_1h.columns if c.startswith("4h_")]

    available = [c for c in feature_cols if c in df_1h.columns]
    return df_1h[available].copy()


def create_labels(df_1h: pd.DataFrame,
                  forward_candles: int = 8,
                  long_threshold: float = 0.015,
                  short_threshold: float = -0.015) -> pd.Series:
    """
    برچسب‌گذاری: نگاه به آینده
    اگر قیمت در N کندل آینده بیش از long_threshold رشد کرد → 1 (LONG)
    اگر افت کرد → -1 (SHORT)
    وگرنه → 0 (WAIT)

    ⚠️ این برچسب‌ها فقط برای آموزش مدل استفاده می‌شوند.
       در پیش‌بینی واقعی، مدل روی آخرین کندل‌ها predict می‌کند.
    """
    future_return = df_1h["close"].shift(-forward_candles) / df_1h["close"] - 1

    labels = pd.Series(0, index=df_1h.index, name="label")
    labels[future_return >= long_threshold]  = 1   # LONG
    labels[future_return <= short_threshold] = -1  # SHORT

    return labels
