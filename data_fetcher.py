"""
data_fetcher.py — دریافت داده OHLCV از Binance
هیچ API key نیاز ندارد (داده عمومی)
"""
import requests
from requests.adapters import HTTPAdapter
try:
    from urllib3.util.retry import Retry
except ImportError:  # سازگاری با نسخه‌های قدیمی‌تر urllib3
    from urllib3.util import Retry
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timezone
from config import BINANCE_BASE, SYMBOL, CANDLE_LIMIT, TRAIN_LIMIT

logger = logging.getLogger(__name__)

# ─── Session با Retry خودکار ─────────────────────────────────
# قبلاً هر فراخوانی requests.get مستقیم و بدون retry بود — یک خطای
# موقت شبکه یا rate-limit لحظه‌ای بایننس (HTTP 429/5xx) باعث می‌شد
# کل تولید سیگنال یا کل آموزش مدل با یک Exception شکست بخورد. حالا
# با retry نمایی (exponential backoff) این خطاهای گذرا به‌صورت
# خودکار ۳ بار دوباره تلاش می‌شوند قبل از این‌که واقعاً fail کنند.
_session = requests.Session()
_retry_strategy = Retry(
    total=3,
    backoff_factor=0.5,             # 0.5s, 1s, 2s
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)
_adapter = HTTPAdapter(max_retries=_retry_strategy)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)


def _get_json(url: str, params: dict, timeout: int = 15):
    resp = _session.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


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
        raw = _get_json(url, params)
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


def fetch_ohlcv_since(symbol: str = SYMBOL,
                      interval: str = "15m",
                      start_time: datetime | None = None,
                      limit: int = 1000) -> pd.DataFrame:
    """
    دریافت کندل‌ها از یک لحظهٔ مشخص (start_time) به بعد.

    برخلاف fetch_ohlcv، این تابع کندل بازِ انتهایی را حذف نمی‌کند و
    close_time را برنمی‌گرداند — چون هدفش ردیابی دقیق برخورد قیمت به
    TP/SL (win_tracker) است و برای این کار به بالاترین/پایین‌ترین
    قیمتِ لحظه‌ای (حتی کندل هنوز بازِ فعلی) هم نیاز داریم؛ کندل باز
    هم بخشی واقعی از حرکت قیمت است و نباید نادیده گرفته شود.

    Returns:
        DataFrame با ستون‌های open, high, low, close, volume — ایندکس UTC
    """
    if start_time is None:
        raise ValueError("پارامتر start_time الزامی است.")

    # اطمینان از timezone-aware بودن ورودی
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)

    url = f"{BINANCE_BASE}/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": int(start_time.timestamp() * 1000),
        "limit": limit,
    }

    try:
        raw = _get_json(url, params)
    except Exception as e:
        logger.error(f"Binance fetch_ohlcv_since error: {e}")
        raise RuntimeError(f"خطا در دریافت کندل‌های ردیابی از Binance: {e}")

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
    data = _get_json(url, {"symbol": symbol})
    return float(data["price"])
