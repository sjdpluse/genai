"""
data_fetcher.py — دریافت داده OHLCV از Binance
هیچ API key نیاز ندارد (داده عمومی)
"""
import requests
import pandas as pd
import numpy as np
import logging
from config import BINANCE_BASE, SYMBOL, CANDLE_LIMIT, TRAIN_LIMIT

logger = logging.getLogger(__name__)


def fetch_ohlcv(symbol: str = SYMBOL,
                interval: str = "1h",
                limit: int = CANDLE_LIMIT) -> pd.DataFrame:
    """
    دریافت کندل‌ها از Binance و تبدیل به DataFrame
    
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
    
    df = pd.DataFrame(raw, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    
    # تبدیل نوع داده
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    
    logger.info(f"دریافت {len(df)} کندل {interval} برای {symbol}")
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
