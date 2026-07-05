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
    "last_error": None,
    "cv_accuracy": None,
    "cv_f1": None,
}
_training_lock = threading.Lock()
scheduler = BackgroundScheduler()


def _model_ready_or_download() -> bool:
    return ensure_model_available()


def hourly_job():
    logger.info("=== کار ساعتی ApexTrade Pro ===")
    try:
        track_pending_signals()
        if not _model_ready_or_download():
            logger.warning("مدل آماده نیست")
            return
        signal = generate_signal()
        save_signal_to_db(signal)
    except Exception as e:
        logger.error(f"خطا در کار ساعتی: {e}")


def _do_training():
    with _training_lock:
        if _training_state["is_training"]:
            raise RuntimeError("آموزش در حال اجراست")
        _training_state["is_training"] = True
        _training_state["last_error"] = None
    
    try:
        logger.info("آموزش مدل شروع شد...")
        metrics = run_training()
        upload_model_to_supabase()
        with _training_lock:
            _training_state["is_training"] = False
            _training_state["last_trained"] = datetime.now(timezone.utc).isoformat()
            _training_state["cv_accuracy"] = metrics.get("cv_accuracy")
            _training_state["cv_f1"] = metrics.get("cv_f1")
        logger.info(f"آموزش موفق — Acc: {metrics.get('cv_accuracy')} | F1: {metrics.get('cv_f1')}")
        return metrics
    except Exception as e:
        with _training_lock:
            _training_state["is_training"] = False
            _training_state["last_error"] = str(e)
        logger.error(f"خطا در آموزش: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("راه‌اندازی ApexTrade Pro v3.0...")
    model_ready = ensure_model_available()
    
    if not model_ready:
        logger.info("مدل پیدا نشد — آموزش اولیه...")
        threading.Thread(target=_do_training, daemon=True).start()
    else:
        try:
            hourly_job()
        except Exception as e:
            logger.warning(f"اولین سیگنال: {e}")
    
    if not ADMIN_TOKEN:
        logger.warning("ADMIN_TOKEN تنظیم نشده")
    
    scheduler.add_job(hourly_job, "interval", hours=1, id="hourly_signal")
    scheduler.start()
    logger.info("سرور آماده")
    yield
    scheduler.shutdown()


app = FastAPI(title="ApexTrade Pro ML Signal API", version="3.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def verify_admin(x_admin_token: str = Header(default="")):
    if ADMIN_TOKEN and x_admin_token != ADMIN_TOKEN:
        raise HTTPException(401, "دسترسی غیرمجاز")
    return True


@app.get("/")
def root():
    return {
        "name": "ApexTrade Pro ML Signal API v3.0",
        "status": "running",
        "model": os.path.exists(MODEL_PATH),
        "time": datetime.now(timezone.utc).isoformat()
    }


@app.get("/signal")
def get_signal():
    if not _model_ready_or_download():
        if _training_state["is_training"]:
            raise HTTPException(503, "مدل در حال آموزش است")
        raise HTTPException(503, "مدل آماده نیست")
    try:
        signal = generate_signal()
        return {"success": True, "signal": signal}
    except Exception as e:
        logger.error(f"خطا در get_signal: {e}")
        raise HTTPException(500, str(e))


@app.post("/train", dependencies=[Depends(verify_admin)])
def train_background(background_tasks: BackgroundTasks):
    if _training_state["is_training"]:
        return {"success": False, "message": "آموزش در حال اجراست"}
    background_tasks.add_task(_do_training)
    return {"success": True, "message": "آموزش شروع شد"}


@app.post("/train-sync", dependencies=[Depends(verify_admin)])
def train_sync():
    try:
        metrics = _do_training()
        return {
            "success": True,
            "cv_accuracy": metrics.get("cv_accuracy"),
            "cv_f1": metrics.get("cv_f1"),
            "n_samples": metrics.get("n_samples"),
            "calibrated": metrics.get("calibrated"),
        }
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"خطا: {e}")


@app.get("/train-status")
def train_status():
    state = _training_state.copy()
    return {
        "is_training": state["is_training"],
        "model_exists": os.path.exists(MODEL_PATH),
        "last_trained": state["last_trained"],
        "cv_accuracy": state["cv_accuracy"],
        "cv_f1": state["cv_f1"],
        "last_error": state["last_error"],
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
        "is_training": state["is_training"],
        "last_trained": state["last_trained"],
        "cv_accuracy": state["cv_accuracy"],
        "cv_f1": state["cv_f1"],
        "scheduler": scheduler.running,
        "time": datetime.now(timezone.utc).isoformat()
    }
