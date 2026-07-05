"""
model_store.py — ذخیره و بارگذاری مدل از Supabase Storage
"""
import io
import os
import joblib
import logging

from supabase_client import get_supabase
from config import MODEL_PATH, SYMBOL

logger = logging.getLogger(__name__)

BUCKET  = "ml-models"
STORAGE_KEY = f"signal_model_{SYMBOL}.joblib"


def upload_model_to_supabase() -> bool:
    """مدل محلی را به Supabase Storage آپلود کن"""
    if not os.path.exists(MODEL_PATH):
        logger.error("فایل محلی مدل پیدا نشد برای آپلود")
        return False
    try:
        sb = get_supabase()
        with open(MODEL_PATH, "rb") as f:
            data = f.read()

        try:
            sb.storage.from_(BUCKET).remove([STORAGE_KEY])
        except Exception:
            pass

        sb.storage.from_(BUCKET).upload(
            path=STORAGE_KEY,
            file=data,
            file_options={"content-type": "application/octet-stream"}
        )
        logger.info(f"مدل با موفقیت در Supabase Storage آپلود شد ({len(data)/1024:.1f} KB)")
        return True
    except Exception as e:
        logger.error(f"خطا در آپلود مدل: {e}")
        return False


def download_model_from_supabase() -> bool:
    """مدل را از Supabase Storage دانلود و محلی ذخیره کن"""
    try:
        sb = get_supabase()
        data = sb.storage.from_(BUCKET).download(STORAGE_KEY)

        if not data:
            logger.warning("مدلی در Supabase Storage پیدا نشد")
            return False

        os.makedirs("model", exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            f.write(data)

        logger.info(f"مدل از Supabase Storage دانلود شد ({len(data)/1024:.1f} KB)")
        return True
    except Exception as e:
        logger.warning(f"دانلود مدل از Supabase Storage ناموفق: {e}")
        return False


def ensure_model_available() -> bool:
    """
    اطمینان از وجود مدل:
    ۱. اگر محلی دارد → خوب
    ۲. اگر ندارد → از Supabase دانلود کن
    ۳. اگر آن هم نداشت → False (نیاز به train)
    """
    if os.path.exists(MODEL_PATH):
        logger.info("مدل محلی موجود است")
        return True

    logger.info("مدل محلی پیدا نشد — دانلود از Supabase Storage...")
    return download_model_from_supabase()
