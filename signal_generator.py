"""
signal_generator.py — 7-Layer Filter Signal Generation
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timezone

from config import (
    SYMBOL, ATR_SL_MULTIPLIER, ATR_TP1_MULTIPLIER, ATR_TP2_MULTIPLIER,
    MIN_CONFIDENCE, MIN_RISK_REWARD, LABEL_LONG_THRESHOLD, LABEL_SHORT_THRESHOLD,
    LABEL_FORWARD_CANDLES, VOLATILITY_FILTER_MAX, VOLUME_FILTER_MIN,
    TREND_FILTER_ADX_MIN, TREND_FILTER_EMA_ALIGN,
    SESSION_FILTER_NIGHT_START, SESSION_FILTER_NIGHT_END,
    FUNDING_RATE_THRESHOLD, MIN_CONFLUENCE_INDICATORS,
)
from data_fetcher import (
    fetch_ohlcv, fetch_multi_timeframe, get_current_price,
    fetch_funding_rate, fetch_order_book, fetch_macro_data,
)
from feature_engineer import compute_indicators, build_feature_matrix, create_labels
from ml_model import load_model, predict, train_model, save_model

logger = logging.getLogger(__name__)


def _calculate_sl_tp(price: float, atr: float, signal_type: str) -> tuple:
    if signal_type == "LONG":
        sl = price - atr * ATR_SL_MULTIPLIER
        tp1 = price + atr * ATR_TP1_MULTIPLIER
        tp2 = price + atr * ATR_TP2_MULTIPLIER
    elif signal_type == "SHORT":
        sl = price + atr * ATR_SL_MULTIPLIER
        tp1 = price - atr * ATR_TP1_MULTIPLIER
        tp2 = price - atr * ATR_TP2_MULTIPLIER
    else:
        sl = tp1 = tp2 = None
    return (round(sl, 2) if sl else None, round(tp1, 2) if tp1 else None, round(tp2, 2) if tp2 else None)


def _risk_reward_ok(price: float, sl: float, tp1: float) -> bool:
    if sl is None or tp1 is None:
        return False
    risk = abs(price - sl)
    reward = abs(tp1 - price)
    if risk == 0:
        return False
    return (reward / risk) >= MIN_RISK_REWARD


def _check_trend_filter(df_ind: pd.DataFrame, signal_type: str) -> bool:
    if not TREND_FILTER_EMA_ALIGN:
        return True
    last = df_ind.iloc[-1]
    adx = last.get("adx", 0)
    price = last.get("close", 0)
    ema200 = last.get("ema_200", 0)

    if adx < TREND_FILTER_ADX_MIN:
        return False
    if signal_type == "LONG" and price < ema200:
        return False
    if signal_type == "SHORT" and price > ema200:
        return False
    return True


def _check_volatility_filter(df_ind: pd.DataFrame) -> bool:
    last = df_ind.iloc[-1]
    atr_pct = last.get("atr_pct", 0)
    return np.isfinite(atr_pct) and atr_pct <= VOLATILITY_FILTER_MAX


def _check_volume_filter(df_ind: pd.DataFrame) -> bool:
    last = df_ind.iloc[-1]
    vol_ratio = last.get("vol_ratio", 1)
    return np.isfinite(vol_ratio) and vol_ratio >= VOLUME_FILTER_MIN


def _check_session_filter() -> bool:
    now = datetime.now(timezone.utc)
    hour = now.hour
    return not (SESSION_FILTER_NIGHT_START <= hour < SESSION_FILTER_NIGHT_END)


def _check_funding_filter(funding_rate: float, signal_type: str) -> bool:
    if signal_type == "LONG" and funding_rate > FUNDING_RATE_THRESHOLD:
        return False
    if signal_type == "SHORT" and funding_rate < -FUNDING_RATE_THRESHOLD:
        return False
    return True


def _count_confluence(df_ind: pd.DataFrame, signal_type: str) -> int:
    last = df_ind.iloc[-1]
    count = 0

    rsi = last.get("rsi", 50)
    if signal_type == "LONG" and rsi < 40:
        count += 1
    elif signal_type == "SHORT" and rsi > 60:
        count += 1

    macd_hist = last.get("macd_hist", 0)
    if signal_type == "LONG" and macd_hist > 0:
        count += 1
    elif signal_type == "SHORT" and macd_hist < 0:
        count += 1

    price_vs_ema20 = last.get("price_vs_ema20", 0)
    if signal_type == "LONG" and price_vs_ema20 > 0:
        count += 1
    elif signal_type == "SHORT" and price_vs_ema20 < 0:
        count += 1

    bb_pct = last.get("bb_pct", 0.5)
    if signal_type == "LONG" and bb_pct < 0.3:
        count += 1
    elif signal_type == "SHORT" and bb_pct > 0.7:
        count += 1

    stoch_k = last.get("stoch_k", 50)
    if signal_type == "LONG" and stoch_k < 30:
        count += 1
    elif signal_type == "SHORT" and stoch_k > 70:
        count += 1

    adx_pos = last.get("adx_pos", 0)
    adx_neg = last.get("adx_neg", 0)
    if signal_type == "LONG" and adx_pos > adx_neg:
        count += 1
    elif signal_type == "SHORT" and adx_neg > adx_pos:
        count += 1

    if signal_type == "LONG" and last.get("rsi_bull_div", 0) == 1:
        count += 1
    elif signal_type == "SHORT" and last.get("rsi_bear_div", 0) == 1:
        count += 1

    return count


def run_training(limit: int = 8000) -> dict:
    logger.info("شروع آموزش ApexTrade Pro...")

    data = fetch_multi_timeframe(limit_1h=limit, limit_4h=2000, limit_1d=500)
    df_1h_ind = compute_indicators(data["1h"])
    funding_df = fetch_funding_rate(SYMBOL, limit=500)

    X = build_feature_matrix(df_1h_ind, data["4h"], data["1d"], funding_df)
    y = create_labels(df_1h_ind, forward_candles=LABEL_FORWARD_CANDLES,
                      long_threshold=LABEL_LONG_THRESHOLD,
                      short_threshold=LABEL_SHORT_THRESHOLD)

    feature_cols = list(X.columns)
    pipeline, metrics = train_model(X, y)
    save_model(pipeline, feature_cols)

    logger.info(f"آموزش تمام. CV Acc: {metrics['cv_accuracy']} | F1: {metrics.get('cv_f1')}")
    return metrics


def generate_signal() -> dict:
    logger.info("تولید سیگنال ApexTrade Pro...")

    pipeline, feature_cols = load_model()

    df_1h = fetch_ohlcv(SYMBOL, "1h", limit=800)
    df_4h = fetch_ohlcv(SYMBOL, "4h", limit=600)
    df_1d = fetch_ohlcv(SYMBOL, "1d", limit=300)
    df_1h_ind = compute_indicators(df_1h)

    funding_df = fetch_funding_rate(SYMBOL, limit=100)
    macro = fetch_macro_data()
    order_book = fetch_order_book(SYMBOL)

    X = build_feature_matrix(df_1h_ind, df_4h, df_1d, funding_df, macro)
    prediction = predict(pipeline, feature_cols, X, min_confidence=MIN_CONFIDENCE)

    signal_type = prediction["type"]
    confidence = prediction["confidence"]
    probabilities = prediction["probabilities"]

    price = float(df_1h_ind["close"].iloc[-1])
    atr = float(df_1h_ind["atr"].iloc[-1])
    if not np.isfinite(atr) or atr <= 0:
        atr = price * 0.01

    sl, tp1, tp2 = _calculate_sl_tp(price, atr, signal_type)

    # ─── 7-Layer Filter System ─────────────────────────────
    filters_passed = []
    filters_failed = []

    # Layer 1: Risk/Reward
    if signal_type != "WAIT":
        if _risk_reward_ok(price, sl, tp1):
            filters_passed.append("R/R OK")
        else:
            filters_failed.append("R/R FAIL")
            signal_type = "WAIT"

    # Layer 2: Trend
    if signal_type != "WAIT":
        if _check_trend_filter(df_1h_ind, prediction["raw_prediction"]):
            filters_passed.append("Trend OK")
        else:
            filters_failed.append("Trend FAIL")
            signal_type = "WAIT"

    # Layer 3: Volatility
    if signal_type != "WAIT":
        if _check_volatility_filter(df_1h_ind):
            filters_passed.append("Vol OK")
        else:
            filters_failed.append("Vol FAIL")
            signal_type = "WAIT"

    # Layer 4: Volume
    if signal_type != "WAIT":
        if _check_volume_filter(df_1h_ind):
            filters_passed.append("Volume OK")
        else:
            filters_failed.append("Volume FAIL")
            signal_type = "WAIT"

    # Layer 5: Session
    if signal_type != "WAIT":
        if _check_session_filter():
            filters_passed.append("Session OK")
        else:
            filters_failed.append("Session FAIL")
            signal_type = "WAIT"

    # Layer 6: Funding
    if signal_type != "WAIT":
        funding_rate = funding_df["funding_rate"].iloc[-1] if not funding_df.empty else 0
        if _check_funding_filter(funding_rate, prediction["raw_prediction"]):
            filters_passed.append("Funding OK")
        else:
            filters_failed.append("Funding FAIL")
            signal_type = "WAIT"

    # Layer 7: Confluence
    if signal_type != "WAIT":
        confluence = _count_confluence(df_1h_ind, prediction["raw_prediction"])
        if confluence >= MIN_CONFLUENCE_INDICATORS:
            filters_passed.append(f"Confluence {confluence}/{MIN_CONFLUENCE_INDICATORS}")
        else:
            filters_failed.append(f"Confluence {confluence}/{MIN_CONFLUENCE_INDICATORS}")
            signal_type = "WAIT"

    if signal_type == "WAIT":
        sl = tp1 = tp2 = None

    reason = _build_reason(prediction, df_1h_ind, filters_passed, filters_failed, order_book, funding_df)

    signal = {
        "type": signal_type,
        "entry_price": price if signal_type != "WAIT" else None,
        "stop_loss": sl,
        "take_profit1": tp1,
        "take_profit2": tp2,
        "confidence": int(confidence),
        "reasons": reason,
        "probabilities": probabilities,
        "filters": {"passed": filters_passed, "failed": filters_failed},
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    logger.info(f"سیگنال: {signal_type} | conf: {confidence}% | filters: {filters_passed}")
    return signal


def _build_reason(prediction: dict, df_ind: pd.DataFrame,
                  filters_passed: list, filters_failed: list,
                  order_book: dict, funding_df: pd.DataFrame) -> str:
    last = df_ind.iloc[-1]
    parts = []

    rsi = last.get("rsi")
    if rsi is not None and pd.notna(rsi):
        parts.append(f"RSI: {rsi:.1f}")

    adx = last.get("adx")
    if adx is not None and pd.notna(adx):
        parts.append(f"ADX: {adx:.1f}")

    macd = last.get("macd_hist")
    if macd is not None and pd.notna(macd):
        parts.append(f"MACD {'صعودی' if macd > 0 else 'نزولی'}")

    bb_pct = last.get("bb_pct")
    if bb_pct is not None and pd.notna(bb_pct):
        if bb_pct < 0.2:
            parts.append("قیمت نزدیک کف BB")
        elif bb_pct > 0.8:
            parts.append("قیمت نزدیک سقف BB")

    ichi_diff = last.get("ichi_cloud_diff")
    if ichi_diff is not None and pd.notna(ichi_diff):
        parts.append("ابر صعودی" if ichi_diff > 0 else "ابر نزولی")

    if last.get("rsi_bull_div", 0) == 1:
        parts.append(" divergence صعودی RSI")
    if last.get("rsi_bear_div", 0) == 1:
        parts.append(" divergence نزولی RSI")

    spread = order_book.get("spread", 0)
    if spread > 0:
        parts.append(f"Spread: {spread*100:.3f}%")

    depth = order_book.get("depth_imbalance", 0)
    if abs(depth) > 0.1:
        parts.append(f"{'خریدار' if depth > 0 else 'فروشنده'} قوی ({depth:.2f})")

    if not funding_df.empty:
        fr = funding_df["funding_rate"].iloc[-1]
        parts.append(f"Funding: {fr*100:.4f}%")

    if filters_passed:
        parts.append(f"✓ {', '.join(filters_passed)}")
    if filters_failed:
        parts.append(f"✗ {', '.join(filters_failed)}")

    proba_str = " | ".join([f"{k}: {v}%" for k, v in prediction["probabilities"].items()])
    parts.append(f"ML: [{proba_str}]")

    return " • ".join([p for p in parts if p])
