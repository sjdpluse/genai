"""
main.py — FastAPI Application (نسخه اصلاح‌شده)
تغییرات:
  - /train-sync  : آموزش همزمان (بلاکینگ) — مطمئن‌ترین روش
  - /train-status: وضعیت آموزش در لحظه
  - مدل در Supabase Storage ذخیره می‌شود (پایدار روی Railway)
  - startup: ابتدا از Supabase دانلود، اگر نبود train می‌کند
"""
import logging
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler

from signal_generator import generate_signal, run_training
from win_tracker import track_pending_signals, get_performance_stats
from supabase_client import save_signal_to_db, get_supabase
from model_store import ensure_model_available, upload_model_to_supabase
from config import MODEL_PATH

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


def hourly_job():
    logger.info("=== کار ساعتی ===")
    try:
        track_pending_signals()
        signal = generate_signal()
        save_signal_to_db(signal)
    except Exception as e:
        logger.error(f"خطا در کار ساعتی: {e}")


def _do_training():
    with _training_lock:
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
        logger.info("مدلی پیدا نشد — آموزش اولیه...")
        try:
            _do_training()
        except Exception as e:
            logger.error(f"آموزش اولیه ناموفق: {e}")
    try:
        hourly_job()
    except Exception as e:
        logger.warning(f"اولین سیگنال: {e}")
    scheduler.add_job(hourly_job, "interval", hours=1, id="hourly_signal")
    scheduler.start()
    logger.info("سرور آماده")
    yield
    scheduler.shutdown()


app = FastAPI(title="ApexTrade ML Signal API", version="2.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
def root():
    return {"name": "ApexTrade ML Signal API v2.1", "status": "running",
            "model": os.path.exists(MODEL_PATH), "time": datetime.now(timezone.utc).isoformat()}


@app.get("/signal")
def get_signal():
    if not os.path.exists(MODEL_PATH):
        from model_store import download_model_from_supabase
        if not download_model_from_supabase():
            if _training_state["is_training"]:
                raise HTTPException(503, "مدل در حال آموزش است — /train-status را چک کنید.")
            raise HTTPException(503, "مدل آماده نیست. POST /train-sync را اجرا کنید.")
    try:
        signal = generate_signal()
        return {"success": True, "signal": signal}
    except Exception as e:
        logger.error(f"خطا در get_signal: {e}")
        raise HTTPException(500, str(e))


@app.post("/train")
def train_background(background_tasks: BackgroundTasks):
    if _training_state["is_training"]:
        return {"success": False, "message": "آموزش در حال اجرا است."}
    background_tasks.add_task(_do_training)
    return {"success": True, "message": "آموزش شروع شد — /train-status را چک کنید."}


@app.post("/train-sync")
def train_sync():
    """آموزش همزمان — منتظر می‌ماند تا تمام شود (۵-۱۰ دقیقه)"""
    if _training_state["is_training"]:
        raise HTTPException(400, "آموزش در حال اجرا است.")
    try:
        metrics = _do_training()
        return {
            "success": True,
            "message": "آموزش تمام شد. /signal حالا کار می‌کند.",
            "cv_accuracy": metrics.get("cv_accuracy"),
            "n_samples": metrics.get("n_samples"),
        }
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
