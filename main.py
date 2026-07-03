import logging
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, BackgroundTasks, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler

from signal_generator import generate_signal, run_training
from win_tracker import track_pending_signals, get_performance_stats
from supabase_client import save_signal_to_db, get_supabase
from model_store import ensure_model_available, upload_model_to_supabase
from config import MODEL_PATH, ADMIN_TOKEN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

_training_state = {
    "is_training": False,
    "last_trained": None,
    "last_error":   None,
    "cv_accuracy":  None,
}
_training_lock = threading.Lock()
scheduler = BackgroundScheduler()


def _model_ready_or_download() -> bool:
    """
    نقطهٔ واحد برای «آیا مدل آماده است؟» — یا محلی موجود است یا از
    Supabase Storage دانلود می‌شود. قبلاً این منطق در /signal تکرار شده
    بود و در hourly_job اصلاً چک نمی‌شد (اگر Railway ری‌استارت شود و
    فایل‌سیستم ephemeral مدل را پاک کند، hourly_job سایلنت شکست
    می‌خورد). حالا هر دو مسیر از همین یک تابع استفاده می‌کنند (DRY).
    """
    return ensure_model_available()


def hourly_job():
    logger.info("=== کار ساعتی ===")
    try:
        track_pending_signals()
        if not _model_ready_or_download():
            logger.warning("مدل آماده نیست — کار ساعتی این دور را رد می‌کند.")
            return
        signal = generate_signal()
        save_signal_to_db(signal)
    except Exception as e:
        logger.error(f"خطا در کار ساعتی: {e}")


def _do_training():
    """
    ⚠️ رفع Race Condition: قبلاً چک `is_training` و ست‌کردن آن به True
    دو عملیات جدا بودند (TOCTOU) — اگر دو درخواست هم‌زمان به /train و
    /train-sync می‌آمدند، هر دو می‌توانستند از چک عبور کنند و هم‌زمان
    run_training() را اجرا کنند (رقابت روی نوشتن فایل مدل + هدررفت CPU
    مضاعف). حالا چک + ست شدن atomically زیر یک lock انجام می‌شود.
    """
    with _training_lock:
        if _training_state["is_training"]:
            raise RuntimeError("آموزش از قبل در حال اجراست.")
        _training_state["is_training"] = True
        _training_state["last_error"]   = None

    try:
        logger.info("آموزش مدل شروع شد...")
        metrics = run_training()
        upload_model_to_supabase()
        with _training_lock:
            _training_state["is_training"] = False
            _training_state["last_trained"] = datetime.now(timezone.utc).isoformat()
            _training_state["cv_accuracy"]  = metrics.get("cv_accuracy")
        logger.info(f"آموزش موفق — CV Accuracy: {metrics.get('cv_accuracy')}")
        return metrics
    except Exception as e:
        with _training_lock:
            _training_state["is_training"] = False
            _training_state["last_error"]  = str(e)
        logger.error(f"خطا در آموزش: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("سرور در حال راه‌اندازی...")
    model_ready = ensure_model_available()

    if not model_ready:
        # ⚠️ رفع مشکل Deploy روی Railway: قبلاً اینجا _do_training() به‌صورت
        # synchronous و بلاک‌کننده صدا زده می‌شد — یعنی سرور تا ۵-۱۰ دقیقه
        # اصلاً به هیچ HTTP request (حتی /health) جواب نمی‌داد، چون
        # lifespan startup قبل از yield کامل نمی‌شد. این می‌تواند باعث
        # Timeout در health-check دیپلوی Railway و شکست کل دیپلوی شود.
        # حالا آموزش اولیه در یک Thread پس‌زمینه اجرا می‌شود؛ سرور فوراً
        # بالا می‌آید و کاربر می‌تواند وضعیت را از /train-status پیگیری کند.
        logger.info("مدلی پیدا نشد — آموزش اولیه در پس‌زمینه شروع شد...")
        threading.Thread(target=_do_training, daemon=True).start()
    else:
        try:
            hourly_job()
        except Exception as e:
            logger.warning(f"اولین سیگنال: {e}")

    if not ADMIN_TOKEN:
        logger.warning(
            "⚠️ ADMIN_TOKEN تنظیم نشده — endpoint های /train و /train-sync "
            "برای همه باز هستند. برای Production حتماً این متغیر محیطی را ست کنید."
        )

    scheduler.add_job(hourly_job, "interval", hours=1, id="hourly_signal")
    scheduler.start()
    logger.info("سرور آماده")
    yield
    scheduler.shutdown()


app = FastAPI(title="ApexTrade ML Signal API", version="2.2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def verify_admin(x_admin_token: str = Header(default="")):
    """
    محافظت از endpoint های سنگین/حساس (آموزش مدل). اگر ADMIN_TOKEN در
    محیط ست نشده باشد، به‌صورت پیش‌فرض باز می‌ماند (سازگاری با نسخهٔ
    قبلی) اما هشدار در لاگ startup داده می‌شود.
    """
    if ADMIN_TOKEN and x_admin_token != ADMIN_TOKEN:
        raise HTTPException(401, "دسترسی غیرمجاز — هدر X-Admin-Token معتبر لازم است.")
    return True


@app.get("/")
def root():
    return {"name": "ApexTrade ML Signal API v2.2", "status": "running",
            "model": os.path.exists(MODEL_PATH), "time": datetime.now(timezone.utc).isoformat()}


@app.get("/signal")
def get_signal():
    if not _model_ready_or_download():
        if _training_state["is_training"]:
            raise HTTPException(503, "مدل در حال آموزش است — /train-status را چک کنید.")
        raise HTTPException(503, "مدل آماده نیست. POST /train-sync را اجرا کنید.")
    try:
        signal = generate_signal()
        return {"success": True, "signal": signal}
    except Exception as e:
        logger.error(f"خطا در get_signal: {e}")
        raise HTTPException(500, str(e))


@app.post("/train", dependencies=[Depends(verify_admin)])
def train_background(background_tasks: BackgroundTasks):
    if _training_state["is_training"]:
        return {"success": False, "message": "آموزش در حال اجرا است."}
    background_tasks.add_task(_do_training)
    return {"success": True, "message": "آموزش شروع شد — /train-status را چک کنید."}


@app.post("/train-sync", dependencies=[Depends(verify_admin)])
def train_sync():
    """آموزش همزمان — منتظر می‌ماند تا تمام شود (۵-۱۰ دقیقه)"""
    try:
        metrics = _do_training()
        return {
            "success": True,
            "message": "آموزش تمام شد. /signal حالا کار می‌کند.",
            "cv_accuracy": metrics.get("cv_accuracy"),
            "n_samples": metrics.get("n_samples"),
            "calibrated": metrics.get("calibrated"),
        }
    except RuntimeError as e:
        # چک atomic داخل _do_training رد شد (آموزش از قبل در حال اجراست)
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"خطا: {e}")


@app.get("/train-status")
def train_status():
    state = _training_state.copy()
    return {
        "is_training":  state["is_training"],
        "model_exists": os.path.exists(MODEL_PATH),
        "last_trained": state["last_trained"],
        "cv_accuracy":  state["cv_accuracy"],
        "last_error":   state["last_error"],
    }


@app.get("/track")
def track_signals():
    try:
        return {"success": True, "result": track_pending_signals()}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/performance")
def performance():
    try:
        return {"success": True, "stats": get_performance_stats()}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/history")
def history(limit: int = 20):
    try:
        sb = get_supabase()
        result = sb.table("signal_history").select("*").order("created_at", desc=True).limit(limit).execute()
        return {"success": True, "signals": result.data}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/health")
def health():
    state = _training_state.copy()
    return {
        "status": "ok",
        "model_trained": os.path.exists(MODEL_PATH),
        "is_training":   state["is_training"],
        "last_trained":  state["last_trained"],
        "cv_accuracy":   state["cv_accuracy"],
        "scheduler":     scheduler.running,
        "time":          datetime.now(timezone.utc).isoformat()
    }
