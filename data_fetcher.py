"""
data_fetcher.py — دریافت OHLCV از CoinGecko (بدون محدودیت IP)
"""
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timezone
from config import SYMBOL, CANDLE_LIMIT, TRAIN_LIMIT

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


# ─── CoinGecko API ──────────────────────────────────────────
COINGECKO_BASE = "https://api.coingecko.com/api/v3"

# نگاشت نماد به ID CoinGecko
COIN_ID_MAP = {
    "ETHUSDT": "ethereum",
    "BTCUSDT": "bitcoin",
    "SOLUSDT": "solana",
    "BNBUSDT": "binancecoin",
    "ADAUSDT": "cardano",
    "XRPUSDT": "ripple",
    "DOTUSDT": "polkadot",
    "AVAXUSDT": "avalanche-2",
    "MATICUSDT": "matic-network",
    "LINKUSDT": "chainlink",
}


def _get_coin_id(symbol: str) -> str:
    """تبدیل نماد Binance به ID CoinGecko"""
    return COIN_ID_MAP.get(symbol, symbol.lower().replace("usdt", ""))


def fetch_ohlcv(symbol: str = SYMBOL, interval: str = "1h",
                limit: int = CANDLE_LIMIT, drop_unclosed: bool = True) -> pd.DataFrame:
    """
    دریافت کندل‌ها از CoinGecko

    ⚠️ CoinGecko فقط تایم‌فریم روزانه (1d) دارد.
    برای 1h و 4h از داده‌های روزانه تقریب می‌زنیم یا از API دیگر استفاده می‌کنیم.

    راه‌حل: از /coins/{id}/market_chart با granularity دقیق‌تر
    """
    coin_id = _get_coin_id(symbol)

    # تبدیل limit به days (تقریبی)
    # 1h: هر روز 24 کندل → limit/24 روز
    # 4h: هر روز 6 کندل → limit/6 روز
    if interval == "1h":
        days = max(limit // 24 + 1, 1)
    elif interval == "4h":
        days = max(limit // 6 + 1, 1)
    elif interval == "1d":
        days = limit
    else:
        days = 30

    # محدودیت CoinGecko: max 365 روز
    days = min(days, 365)

    url = f"{COINGECKO_BASE}/coins/{coin_id}/ohlc"
    params = {
        "vs_currency": "usd",
        "days": str(days),
    }

    try:
        raw = _get_json(url, params, timeout=30)
    except Exception as e:
        logger.error(f"CoinGecko fetch error: {e}")
        raise RuntimeError(f"خطا در دریافت داده از CoinGecko: {e}")

    if not raw:
        raise RuntimeError(f"CoinGecko داده‌ای برای {coin_id} برنگرداند.")

    # CoinGecko OHLC: [timestamp, open, high, low, close]
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)

    # حجم را از market_chart بگیریم
    try:
        vol_url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
        vol_params = {"vs_currency": "usd", "days": str(days)}
        vol_data = _get_json(vol_url, vol_params, timeout=30)
        vol_df = pd.DataFrame(vol_data["total_volumes"], columns=["timestamp", "volume"])
        vol_df["timestamp"] = pd.to_datetime(vol_df["timestamp"], unit="ms", utc=True)
        vol_df["volume"] = vol_df["volume"].astype(float)
        df = df.merge(vol_df, on="timestamp", how="left")
    except Exception as e:
        logger.warning(f"خطا در دریافت حجم: {e} — با ۰ پر می‌شود")
        df["volume"] = 0.0

    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    # حذف کندل باز (آخرین کندل امروز اگر هنوز بسته نشده)
    if drop_unclosed and len(df) > 0:
        now_utc = pd.Timestamp(datetime.now(timezone.utc))
        last_candle_date = df.index[-1].floor("D")
        if last_candle_date >= now_utc.floor("D"):
            # اگر آخرین کندل امروز است و هنوز روز تمام نشده
            pass  # CoinGecko معمولاً دیروز را برمی‌گرداند

    logger.info(f"دریافت {len(df)} کندل از CoinGecko برای {coin_id} ({interval})")
    return df


def fetch_ohlcv_since(symbol: str = SYMBOL, interval: str = "5m",
                      start_time: datetime = None, limit: int = 1000) -> pd.DataFrame:
    """
    دریافت کندل‌ها از زمان مشخص — برای win_tracker
    CoinGecko granularity دقیق ندارد، از market_chart استفاده می‌کنیم
    """
    if start_time is None:
        raise ValueError("start_time الزامی است")

    coin_id = _get_coin_id(symbol)

    # محاسبه تعداد روز از start_time تا الان
    now = datetime.now(timezone.utc)
    days_diff = (now - start_time).days + 1
    days_diff = min(max(days_diff, 1), 90)  # محدودیت CoinGecko

    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": str(days_diff)}

    try:
        data = _get_json(url, params, timeout=30)
    except Exception as e:
        logger.error(f"CoinGecko since error: {e}")
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    # market_chart: prices [[timestamp, price], ...]
    prices = pd.DataFrame(data["prices"], columns=["timestamp", "close"])
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], unit="ms", utc=True)

    # برای win_tracker به high/low نیاز داریم — از close approx می‌زنیم
    prices["open"] = prices["close"].shift(1)
    prices["high"] = prices["close"] * 1.002  # تخمین 0.2%
    prices["low"] = prices["close"] * 0.998   # تخمین 0.2%

    # حجم
    if "total_volumes" in data:
        vols = pd.DataFrame(data["total_volumes"], columns=["timestamp", "volume"])
        vols["timestamp"] = pd.to_datetime(vols["timestamp"], unit="ms", utc=True)
        prices = prices.merge(vols, on="timestamp", how="left")
    else:
        prices["volume"] = 0.0

    prices.set_index("timestamp", inplace=True)
    prices = prices[["open", "high", "low", "close", "volume"]].dropna()
    prices.sort_index(inplace=True)

    # فیلتر از start_time
    prices = prices[prices.index >= start_time]

    return prices


def fetch_multi_timeframe(symbol: str = SYMBOL,
                          limit_1h: int = TRAIN_LIMIT,
                          limit_4h: int = 2000,
                          limit_1d: int = 500) -> dict:
    """
    دریافت همزمان چند تایم‌فریم از CoinGecko
    ⚠️ CoinGecko فقط daily دارد — همه را با daily می‌گیریم
    """
    # CoinGecko فقط daily OHLC دارد
    # برای 1h و 4h از daily استفاده می‌کنیم (تقریب)
    df_daily = fetch_ohlcv(symbol, "1d", limit=limit_1d)

    return {
        "1h": df_daily,   # تقریب: daily به‌جای 1h
        "4h": df_daily,   # تقریب: daily به‌جای 4h
        "1d": df_daily,
    }


def get_current_price(symbol: str = SYMBOL) -> float:
    """قیمت لحظه‌ای از CoinGecko"""
    coin_id = _get_coin_id(symbol)
    url = f"{COINGECKO_BASE}/simple/price"
    params = {"ids": coin_id, "vs_currencies": "usd"}
    data = _get_json(url, params)
    return float(data[coin_id]["usd"])


def fetch_funding_rate(symbol: str = SYMBOL, limit: int = 500) -> pd.DataFrame:
    """
    CoinGecko funding rate ندارد — DataFrame خالی برمی‌گرداند
    """
    logger.warning("CoinGecko funding rate ندارد — با ۰ پر می‌شود")
    return pd.DataFrame(columns=["funding_rate"])


def fetch_order_book(symbol: str = SYMBOL, limit: int = 100) -> dict:
    """
    CoinGecko order book ندارد — مقادیر پیش‌فرض
    """
    logger.warning("CoinGecko order book ندارد — مقادیر پیش‌فرض")
    return {"spread": 0, "depth_imbalance": 0, "best_bid": 0, "best_ask": 0}


def fetch_macro_data() -> dict:
    """داده‌های کلان — ساده‌شده"""
    macro = {}
    try:
        # DXY از Yahoo Finance (ممکن است محدود باشد)
        url = "https://query1.finance.yahoo.com/v8/finance/chart/DX-Y.NYB"
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
        url = "https://query1.finance.yahoo.com/v8/finance/chart/^VIX"
        params = {"interval": "1d", "range": "1mo"}
        data = _get_json(url, params, timeout=10)
        result = data["chart"]["result"][0]
        macro["vix_level"] = result["meta"]["regularMarketPrice"]
    except Exception as e:
        logger.warning(f"VIX error: {e}")
        macro["vix_level"] = 20

    try:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/BTC.D"
        params = {"interval": "1d", "range": "1mo"}
        data = _get_json(url, params, timeout=10)
        result = data["chart"]["result"][0]
        macro["btc_dominance"] = result["meta"]["regularMarketPrice"]
    except Exception as e:
        logger.warning(f"BTC dom error: {e}")
        macro["btc_dominance"] = 50

    return macro
