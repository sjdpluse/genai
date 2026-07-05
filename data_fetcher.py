"""
data_fetcher.py — دریافت OHLCV + Funding Rate + Order Book + Macro
"""
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timezone
from config import (
    BINANCE_BASE, BINANCE_FAPI, YAHOO_BASE,
    SYMBOL, CANDLE_LIMIT, TRAIN_LIMIT
)

logger = logging.getLogger(__name__)

_session = requests.Session()
_retry_strategy = Retry(
    total=3, backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)
_adapter = HTTPAdapter(max_retries=_retry_strategy)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)


def _get_json(url: str, params: dict = None, timeout: int = 15):
    resp = _session.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def fetch_ohlcv(symbol: str = SYMBOL, interval: str = "1h",
                limit: int = CANDLE_LIMIT, drop_unclosed: bool = True) -> pd.DataFrame:
    url = f"{BINANCE_BASE}/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    
    raw = _get_json(url, params)
    if not raw:
        raise RuntimeError(f"Binance داده‌ای برنگرداند {symbol}/{interval}")
    
    df = pd.DataFrame(raw, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume", "close_time"]].copy()
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    
    if drop_unclosed and len(df) > 0:
        now_utc = pd.Timestamp(datetime.now(timezone.utc))
        if df["close_time"].iloc[-1] > now_utc:
            df = df.iloc[:-1]
            logger.info(f"کندل باز {interval} حذف شد")
    
    df = df.drop(columns=["close_time"])
    logger.info(f"دریافت {len(df)} کندل {interval} برای {symbol}")
    return df


def fetch_ohlcv_since(symbol: str = SYMBOL, interval: str = "5m",
                      start_time: datetime = None, limit: int = 1000) -> pd.DataFrame:
    if start_time is None:
        raise ValueError("start_time الزامی است")
    
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    
    url = f"{BINANCE_BASE}/klines"
    params = {
        "symbol": symbol, "interval": interval,
        "startTime": int(start_time.timestamp() * 1000),
        "limit": limit,
    }
    
    raw = _get_json(url, params)
    if not raw:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    
    df = pd.DataFrame(raw, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    return df


def fetch_multi_timeframe(symbol: str = SYMBOL,
                          limit_1h: int = TRAIN_LIMIT,
                          limit_4h: int = 2000,
                          limit_1d: int = 500) -> dict:
    return {
        "1h": fetch_ohlcv(symbol, "1h", limit_1h),
        "4h": fetch_ohlcv(symbol, "4h", limit_4h),
        "1d": fetch_ohlcv(symbol, "1d", limit_1d),
    }


def get_current_price(symbol: str = SYMBOL) -> float:
    url = f"{BINANCE_BASE}/ticker/price"
    data = _get_json(url, {"symbol": symbol})
    return float(data["price"])


def fetch_funding_rate(symbol: str = SYMBOL, limit: int = 500) -> pd.DataFrame:
    url = f"{BINANCE_FAPI}/fundingRate"
    params = {"symbol": symbol, "limit": limit}
    
    try:
        raw = _get_json(url, params, timeout=10)
        if not raw:
            return pd.DataFrame(columns=["funding_rate"])
        
        df = pd.DataFrame(raw)
        df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        df["funding_rate"] = df["fundingRate"].astype(float)
        df = df[["fundingTime", "funding_rate"]].copy()
        df.set_index("fundingTime", inplace=True)
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        logger.warning(f"خطا در funding rate: {e}")
        return pd.DataFrame(columns=["funding_rate"])


def fetch_order_book(symbol: str = SYMBOL, limit: int = 100) -> dict:
    url = f"{BINANCE_BASE}/depth"
    params = {"symbol": symbol, "limit": limit}
    
    try:
        data = _get_json(url, params, timeout=10)
        bids = pd.DataFrame(data["bids"], columns=["price", "qty"], dtype=float)
        asks = pd.DataFrame(data["asks"], columns=["price", "qty"], dtype=float)
        
        best_bid = bids["price"].iloc[0]
        best_ask = asks["price"].iloc[0]
        spread = (best_ask - best_bid) / best_bid
        
        bid_depth = bids["qty"].sum()
        ask_depth = asks["qty"].sum()
        depth_imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)
        
        return {
            "spread": spread,
            "depth_imbalance": depth_imbalance,
            "best_bid": best_bid,
            "best_ask": best_ask,
        }
    except Exception as e:
        logger.warning(f"خطا در order book: {e}")
        return {"spread": 0, "depth_imbalance": 0, "best_bid": 0, "best_ask": 0}


def fetch_macro_data() -> dict:
    macro = {}
    
    try:
        url = f"{YAHOO_BASE}/DX-Y.NYB"
        params = {"interval": "1d", "range": "1mo"}
        data = _get_json(url, params, timeout=10)
        result = data["chart"]["result"][0]
        close = result["meta"]["regularMarketPrice"]
        prev_close = result["meta"]["previousClose"]
        macro["dxy_change"] = (close - prev_close) / prev_close if prev_close else 0
    except Exception as e:
        logger.warning(f"DXY error: {e}")
        macro["dxy_change"] = 0
    
    try:
        url = f"{YAHOO_BASE}/^VIX"
        params = {"interval": "1d", "range": "1mo"}
        data = _get_json(url, params, timeout=10)
        result = data["chart"]["result"][0]
        macro["vix_level"] = result["meta"]["regularMarketPrice"]
    except Exception as e:
        logger.warning(f"VIX error: {e}")
        macro["vix_level"] = 20
    
    try:
        url = f"{YAHOO_BASE}/BTC.D"
        params = {"interval": "1d", "range": "1mo"}
        data = _get_json(url, params, timeout=10)
        result = data["chart"]["result"][0]
        macro["btc_dominance"] = result["meta"]["regularMarketPrice"]
    except Exception as e:
        logger.warning(f"BTC dom error: {e}")
        macro["btc_dominance"] = 50
    
    return macro
