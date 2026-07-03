"""
supabase_client.py — اتصال به Supabase
"""
from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_KEY, MAX_PENDING_SIGNALS
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


def count_pending_signals() -> int:
    """تعداد سیگنال‌های هنوز باز (pending) در دیتابیس"""
    sb = get_supabase()
    try:
        result = (sb.table("signal_history")
                    .select("id", count="exact")
                    .eq("status", "pending")
                    .execute())
        return result.count if result.count is not None else len(result.data)
    except Exception as e:
        logger.error(f"خطا در شمارش سیگنال‌های pending: {e}")
        # در حالت خطا محافظه‌کارانه عمل می‌کنیم: طوری فرض می‌کنیم که
        # انگار ظرفیت پر است، تا به‌جای صدور سیگنال ناخواسته، فقط
        # یک سیگنال را رد کنیم (ریسک را بر تجارت اولویت می‌دهیم)
        return MAX_PENDING_SIGNALS


def save_signal_to_db(signal: dict) -> dict | None:
    """
    ذخیره سیگنال در جدول signal_history
    فقط اگر سیگنال LONG یا SHORT باشد ذخیره می‌کنیم

    ⚠️ جلوگیری از Overtrading: اگر به‌اندازهٔ MAX_PENDING_SIGNALS معاملهٔ
    pending باز داریم، سیگنال جدید ذخیره نمی‌شود — مدیریت ریسک روی تعداد
    معاملات هم‌زمان، اولویت بالاتری از فرکانس سیگنال دارد. این از انباشته
    شدن چند سیگنال هم‌جهت/متناقض روی هم جلوگیری می‌کند.
    """
    if signal.get("type") == "WAIT":
        logger.info("سیگنال WAIT است — در دیتابیس ذخیره نمی‌شود.")
        return None

    pending_count = count_pending_signals()
    if pending_count >= MAX_PENDING_SIGNALS:
        logger.info(
            f"سیگنال {signal.get('type')} رد شد — در حال حاضر "
            f"{pending_count} معاملهٔ pending باز است (سقف: {MAX_PENDING_SIGNALS})."
        )
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
