"""
supabase_client.py — اتصال به Supabase
"""
from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_KEY
import logging

logger = logging.getLogger(__name__)

_client: Client | None = None


def get_supabase() -> Client:
    """Singleton client"""
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError(
                "SUPABASE_URL و SUPABASE_KEY در .env تنظیم نشده‌اند!"
            )
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("اتصال به Supabase برقرار شد.")
    return _client


def save_signal_to_db(signal: dict) -> dict | None:
    """
    ذخیره سیگنال در جدول signal_history
    فقط اگر سیگنال LONG یا SHORT باشد ذخیره می‌کنیم
    """
    if signal.get("type") == "WAIT":
        logger.info("سیگنال WAIT است — در دیتابیس ذخیره نمی‌شود.")
        return None

    sb = get_supabase()

    row = {
        "type":          signal["type"],
        "entry_price":   signal.get("entry_price"),
        "stop_loss":     signal.get("stop_loss"),
        "take_profit1":  signal.get("take_profit1"),
        "take_profit2":  signal.get("take_profit2"),
        "confidence":    signal.get("confidence"),
        "reasons":       signal.get("reasons"),
        "status":        "pending",
    }

    try:
        result = sb.table("signal_history").insert(row).execute()
        inserted = result.data[0] if result.data else None
        logger.info(f"سیگنال ذخیره شد: ID={inserted.get('id', '?') if inserted else '?'}")
        return inserted
    except Exception as e:
        logger.error(f"خطا در ذخیره سیگنال: {e}")
        return None
