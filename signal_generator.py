"""
signal_generator.py — تولید سیگنال نهایی
ادغام خروجی مدل ML با مدیریت ریسک واقعی
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timezone

from config import (SYMBOL, ATR_SL_MULTIPLIER,
                    ATR_TP1_MULTIPLIER, ATR_TP2_MULTIPLIER,
                    MIN_CONFIDENCE, LABEL_LONG_THRESHOLD,
                    LABEL_SHORT_THRESHOLD, LABEL_FORWARD_CANDLES)
from data_fetcher import fetch_ohlcv, fetch_multi_timeframe, get_current_price
from feature_engineer import compute_indicators, build_feature_matrix, create_labels
from ml_model import load_model, predict, train_model, save_model

logger = logging.getLogger(__name__)


def _calculate_sl_tp(price: float, atr: float,
                     signal_type: str) -> tuple:
    """محاسبه Stop Loss و Take Profit بر اساس ATR"""
    if signal_type == "LONG":
        sl  = price - atr * ATR_SL_MULTIPLIER
        tp1 = price + atr * ATR_TP1_MULTIPLIER
        tp2 = price + atr * ATR_TP2_MULTIPLIER
    elif signal_type == "SHORT":
        sl  = price + atr * ATR_SL_MULTIPLIER
        tp1 = price - atr * ATR_TP1_MULTIPLIER
        tp2 = price - atr * ATR_TP2_MULTIPLIER
    else:
        sl = tp1 = tp2 = None

    return (
        round(sl,  2) if sl  is not None else None,
        round(tp1, 2) if tp1 is not None else None,
        round(tp2, 2) if tp2 is not None else None
    )


def _risk_reward_ok(price: float, sl: float,
                    tp1: float, min_rr: float = 1.8) -> bool:
    """بررسی نسبت ریسک به ریوارد"""
    if sl is None or tp1 is None:
        return False
    risk   = abs(price - sl)
    reward = abs(tp1 - price)
    if risk == 0:
        return False
    return (reward / risk) >= min_rr


def run_training(limit: int = 5000) -> dict:
    """
    آموزش کامل مدل:
    1. دریافت داده تاریخی
    2. محاسبه اندیکاتورها
    3. ساخت برچسب‌ها
    4. آموزش RandomForest
    5. ذخیره مدل
    """
    logger.info("شروع آموزش مدل...")

    # دریافت داده
    from config import TRAIN_LIMIT
    data = fetch_multi_timeframe(limit_1h=limit)
    df_1h = data["1h"]
    df_4h = data["4h"]

    # محاسبه ویژگی‌ها
    X = build_feature_matrix(df_1h, df_4h)

    # ساخت برچسب‌ها
    y = create_labels(
        df_1h,
        forward_candles=LABEL_FORWARD_CANDLES,
        long_threshold=LABEL_LONG_THRESHOLD,
        short_threshold=LABEL_SHORT_THRESHOLD
    )

    feature_cols = list(X.columns)

    # آموزش
    pipeline, metrics = train_model(X, y)

    # ذخیره
    save_model(pipeline, feature_cols)

    logger.info(f"آموزش تمام شد. CV Accuracy: {metrics['cv_accuracy']}")
    return metrics


def generate_signal() -> dict:
    """
    تولید سیگنال لحظه‌ای:
    1. دریافت آخرین کندل‌ها
    2. محاسبه اندیکاتورها
    3. پیش‌بینی با مدل ML
    4. محاسبه SL/TP
    5. بازگشت دیکشنری سیگنال

    Returns:
        dict آماده برای ذخیره در Supabase
    """
    logger.info("تولید سیگنال...")

    # بارگذاری مدل
    pipeline, feature_cols = load_model()

    # دریافت داده تازه
    df_1h = fetch_ohlcv(SYMBOL, "1h", limit=500)
    df_4h = fetch_ohlcv(SYMBOL, "4h", limit=300)

    # محاسبه ویژگی‌ها
    X = build_feature_matrix(df_1h, df_4h)

    # هم‌راستا کردن ستون‌های داده جدید با ستون‌های زمان آموزش
    # این کار از خطا جلوگیری می‌کند اگر برخی ویژگی‌ها در داده جدید محاسبه نشوند
    X_aligned = X.reindex(columns=feature_cols, fill_value=np.nan)

    # پیش‌بینی
    prediction = predict(pipeline, feature_cols, X_aligned, min_confidence=MIN_CONFIDENCE)

    signal_type = prediction["type"]
    confidence  = prediction["confidence"]
    probabilities = prediction["probabilities"]

    # قیمت فعلی و ATR
    price = float(df_1h["close"].iloc[-1])
    atr   = float(df_1h["close"].diff().abs().rolling(14).mean().iloc[-1])

    # SL / TP
    sl, tp1, tp2 = _calculate_sl_tp(price, atr, signal_type)

    # بررسی Risk/Reward
    if signal_type != "WAIT" and not _risk_reward_ok(price, sl, tp1):
        reason = f"ریسک به ریوارد کافی نیست (min 1.8). ML پیش‌بینی {prediction['raw_prediction']} داد."
        signal_type = "WAIT"
        sl = tp1 = tp2 = None
    else:
        reason = _build_reason(prediction, df_1h)

    signal = {
        "type":          signal_type,
        "entry_price":   price if signal_type != "WAIT" else None,
        "stop_loss":     sl,
        "take_profit1":  tp1,
        "take_profit2":  tp2,
        "confidence":    int(confidence),
        "reasons":       reason,
        "probabilities": probabilities,
        "status":        "pending",
        "created_at":    datetime.now(timezone.utc).isoformat()
    }

    logger.info(f"سیگنال: {signal_type} | confidence: {confidence}% | price: {price}")
    return signal


def _build_reason(prediction: dict, df: pd.DataFrame) -> str:
    """ساخت توضیح فارسی برای سیگنال"""
    df_ind = compute_indicators(df)
    last = df_ind.iloc[-1]
    parts = []

    rsi = last.get("rsi", None)
    if rsi is not None:
        parts.append(f"RSI: {rsi:.1f}")

    adx = last.get("adx", None)
    if adx is not None:
        parts.append(f"ADX: {adx:.1f}")

    macd = last.get("macd_hist", None)
    if macd is not None:
        trend = "صعودی" if macd > 0 else "نزولی"
        parts.append(f"MACD {trend}")

    bb_pct = last.get("bb_pct", None)
    if bb_pct is not None:
        if bb_pct < 0.2:
            parts.append("قیمت نزدیک کف بولینگر")
        elif bb_pct > 0.8:
            parts.append("قیمت نزدیک سقف بولینگر")

    proba_str = " | ".join(
        [f"{k}: {v}%" for k, v in prediction["probabilities"].items()]
    )
    parts.append(f"احتمالات ML: [{proba_str}]")

    return " • ".join(parts)
