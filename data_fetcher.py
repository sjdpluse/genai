"""
data_fetcher.py — دریافت داده OHLCV از Binance
هیچ API key نیاز ندارد (داده عمومی)
"""
import requests
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timezone
from config import BINANCE_BASE, SYMBOL, CANDLE_LIMIT, TRAIN_LIMIT

logger = logging.getLogger(__name__)


def fetch_ohlcv(symbol: str = SYMBOL,
                interval: str = "1h",
                limit: int = CANDLE_LIMIT,
                drop_unclosed: bool = True) -> pd.DataFrame:
    """
    دریافت کندل‌ها از Binance و تبدیل به DataFrame

    ⚠️ نکته مهم: آخرین کندلی که Binance برمی‌گرداند معمولاً کندل
    «درحال‌شکل‌گیری» (هنوز بسته نشده) است. استفاده از این کندل برای
    محاسبه اندیکاتورها و قیمت باعث می‌شود مقادیر هر ثانیه عوض شوند و
    سیگنال‌ها ناپایدار/غیرقابل‌اعتماد به نظر برسند («دیتای ناقص»).
    با drop_unclosed=True این کندل باز، قبل از بازگشت حذف می‌شود.

    Returns:
        DataFrame با ستون‌های: open, high, low, close, volume, timestamp
    """
    url = f"{BINANCE_BASE}/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        raw = resp.json()
    except Exception as e:
        logger.error(f"Binance fetch error: {e}")
        raise RuntimeError(f"خطا در دریافت داده از Binance: {e}")

    if not raw:
        raise RuntimeError(f"Binance داده‌ای برای {symbol}/{interval} برنگرداند.")

    df = pd.DataFrame(raw, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])

    # تبدیل نوع داده
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["timestamp"]  = pd.to_datetime(df["timestamp"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    df = df[["timestamp", "open", "high", "low", "close", "volume", "close_time"]].copy()
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    # ─── حذف کندل بازِ (بسته‌نشده) انتهایی ──────────────────
    if drop_unclosed and len(df) > 0:
        now_utc = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))
        if df["close_time"].iloc[-1] > now_utc:
            df = df.iloc[:-1]
            logger.info(f"کندل بازِ {interval} حذف شد (هنوز بسته نشده بود).")

    df = df.drop(columns=["close_time"])

    if df.empty:
        raise RuntimeError(f"پس از حذف کندل باز، داده‌ای برای {symbol}/{interval} باقی نماند.")

    logger.info(f"دریافت {len(df)} کندل بسته‌شدهٔ {interval} برای {symbol}")
    return df


def fetch_multi_timeframe(symbol: str = SYMBOL,
                          limit_1h: int = TRAIN_LIMIT,
                          limit_4h: int = 1500) -> dict:
    """
    دریافت همزمان چند تایم‌فریم

    Returns:
        dict با کلیدهای '1h' و '4h'
    """
    return {
        "1h": fetch_ohlcv(symbol, "1h", limit_1h),
        "4h": fetch_ohlcv(symbol, "4h", limit_4h),
    }


def get_current_price(symbol: str = SYMBOL) -> float:
    """قیمت لحظه‌ای"""
    url = f"{BINANCE_BASE}/ticker/price"
    resp = requests.get(url, params={"symbol": symbol}, timeout=10)
    resp.raise_for_status()
    return float(resp.json()["price"])
