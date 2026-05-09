"""
main.py — FastAPI Application
تمام endpoint‌های API اینجا تعریف می‌شوند
"""
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler

from signal_generator import generate_signal, run_training
from win_tracker import track_pending_signals, get_performance_stats
from supabase_client import save_signal_to_db, get_supabase

# ─── Logging ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ─── Scheduler ───────────────────────────────────────────────
scheduler = BackgroundScheduler()


def hourly_job():
    """کار ساعتی: تولید سیگنال + ردیابی win/loss"""
    logger.info("=== اجرای کار ساعتی ===")
    try:
        # ۱. ردیابی سیگنال‌های قبلی
        track_pending_signals()

        # ۲. تولید سیگنال جدید
        signal = generate_signal()

        # ۳. ذخیره در Supabase
        save_signal_to_db(signal)

    except Exception as e:
        logger.error(f"خطا در کار ساعتی: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """راه‌اندازی scheduler هنگام شروع سرور"""
    # اگر مدل وجود ندارد، آن را آموزش بده
    model_path = "model/signal_model.joblib"
    if not os.path.exists(model_path):
        logger.info("مدل آموزش‌دیده‌ای پیدا نشد. آموزش اولیه شروع می‌شود...")
        try:
            run_training()
        except Exception as e:
            logger.error(f"خطا در آموزش اولیه: {e}")

    # اجرای فوری اولین سیگنال
    try:
        hourly_job()
    except Exception as e:
        logger.warning(f"اجرای اولیه: {e}")

    # Scheduler هر ۱ ساعت یک بار
    scheduler.add_job(hourly_job, "interval", hours=1, id="hourly_signal")
    scheduler.start()
    logger.info("Scheduler فعال شد — هر ۱ ساعت سیگنال تولید می‌شود")

    yield

    scheduler.shutdown()


# ─── App ─────────────────────────────────────────────────────
app = FastAPI(
    title="ApexTrade ML Signal API",
    description="سیستم تولید سیگنال با یادگیری ماشین برای ApexTrade",
    version="2.0.0",
    lifespan=lifespan
)

# CORS — اجازه دسترسی از وب‌سایت شما
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # در production: URL وب‌سایتتان را بگذارید
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Endpoints ───────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "ApexTrade ML Signal API",
        "status": "running",
        "time": datetime.now(timezone.utc).isoformat()
    }


@app.get("/signal")
def get_signal():
    """
    تولید سیگنال لحظه‌ای با مدل ML
    این endpoint را می‌توانید از وب‌سایت‌تان صدا بزنید
    """
    try:
        signal = generate_signal()
        return {"success": True, "signal": signal}
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="مدل هنوز آموزش ندیده. از /train استفاده کنید."
        )
    except Exception as e:
        logger.error(f"خطا در get_signal: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
def train(background_tasks: BackgroundTasks):
    """
    آموزش یا آموزش مجدد مدل ML
    در background اجرا می‌شود (۵-۱۰ دقیقه طول می‌کشد)
    """
    background_tasks.add_task(_run_training_task)
    return {
        "success": True,
        "message": "آموزش مدل در background شروع شد. چند دقیقه صبر کنید."
    }


def _run_training_task():
    try:
        metrics = run_training()
        logger.info(f"آموزش تمام شد: {metrics}")
    except Exception as e:
        logger.error(f"خطا در آموزش: {e}")


@app.get("/track")
def track_signals():
    """ردیابی دستی نتیجه سیگنال‌های pending"""
    try:
        result = track_pending_signals()
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/performance")
def performance():
    """آمار عملکرد کلی سیگنال‌ها"""
    try:
        stats = get_performance_stats()
        return {"success": True, "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history")
def history(limit: int = 20):
    """
    تاریخچه سیگنال‌های اخیر از Supabase
    """
    try:
        sb = get_supabase()
        result = sb.table("signal_history")\
                   .select("*")\
                   .order("created_at", desc=True)\
                   .limit(limit)\
                   .execute()
        return {"success": True, "signals": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    """بررسی سلامت سرویس"""
    model_exists = os.path.exists("model/signal_model.joblib")
    return {
        "status":        "ok",
        "model_trained": model_exists,
        "scheduler":     scheduler.running,
        "time":          datetime.now(timezone.utc).isoformat()
    }
