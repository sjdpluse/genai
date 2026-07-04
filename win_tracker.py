"""
win_tracker.py — ردیاب خودکار نتیجه سیگنال‌ها
هر ساعت اجرا می‌شود و سیگنال‌های pending را بررسی می‌کند

⚠️ رفع باگ مهم (نسخهٔ قبلی):
نسخهٔ قبلی فقط قیمت لحظه‌ای (spot price) را در زمان اجرای job با TP/SL
مقایسه می‌کرد. یعنی اگر بین دو اجرا (هر ۱ ساعت) قیمت به‌طور موقت به TP
یا SL برخورد می‌کرد و برمی‌گشت، این رویداد کاملاً از دست می‌رفت و
سیگنال به‌اشتباه همچنان pending می‌ماند یا با تأخیر/غلط بسته می‌شد.
این باعث می‌شد آمار /performance (که تنها معیار سنجش دقت واقعی سیستم
است) غیرقابل‌اعتماد باشد.

راه‌حل: برای هر سیگنال pending، کندل‌های ۱۵ دقیقه‌ای از لحظهٔ صدور
سیگنال تا الان از Binance گرفته می‌شود و به‌ترتیب زمانی بررسی می‌شود
که آیا high/low هرکدام واقعاً به TP1 یا SL رسیده — و مهم‌تر، کدام
زودتر رسیده.
"""
import logging
from datetime import datetime, timezone
from data_fetcher import get_current_price, fetch_ohlcv_since
from supabase_client import get_supabase

logger = logging.getLogger(__name__)

# تایم‌فریم ردیابی — هرچه ریزتر، تشخیص ترتیب برخورد TP/SL دقیق‌تر.
# ۱۵ دقیقه تعادل خوبی بین دقت و تعداد کندل/درخواست است.
TRACK_INTERVAL = "15m"
EXPIRY_HOURS = 72


def _resolve_from_candles(sig_type: str, tp1: float, sl: float,
                          candles) -> tuple:
    """
    بررسی کندل‌به‌کندل (به‌ترتیب زمانی صعودی) برای پیدا کردن اولین
    برخورد واقعی به TP1 یا SL.

    ⚠️ ابهام درون‌کندلی: اگر هم high و هم low یک کندل ۱۵ دقیقه‌ای
    هم‌زمان TP و SL را پوشش دهند (کندل بسیار پرنوسان)، از روی داده
    OHLC نمی‌شود فهمید کدام واقعاً زودتر رخ داده. در این حالت به‌صورت
    محافظه‌کارانه فرض می‌کنیم SL زودتر لمس شده — در مدیریت ریسک، فرض
    بدترین سناریوی محتمل همیشه امن‌تر از خوش‌بینی است.

    Returns:
        (outcome, closed_at) یا (None, None) اگر هنوز به هیچ‌کدام نرسیده
    """
    for ts, row in candles.iterrows():
        hi, lo = float(row["high"]), float(row["low"])

        if sig_type == "LONG":
            hit_tp = hi >= tp1
            hit_sl = lo <= sl
        elif sig_type == "SHORT":
            hit_tp = lo <= tp1
            hit_sl = hi >= sl
        else:
            return None, None

        if hit_tp and hit_sl:
            return "loss", ts
        if hit_sl:
            return "loss", ts
        if hit_tp:
            return "win", ts

    return None, None


def _fallback_spot_check(sig_type: str, tp1: float, sl: float) -> str | None:
    """
    fallback محافظه‌کارانه وقتی دریافت کندل‌های تاریخی ناموفق بود:
    حداقل با قیمت لحظه‌ای چک می‌کنیم که آیا *همین الان* در وضعیت
    win/loss هستیم یا نه. دقت کمتری نسبت به بررسی کندل‌به‌کندل دارد
    (ممکن است یک برخورد میانی را از دست بدهد) اما بهتر از رها کردن
    کامل سیگنال در حالت pending است.
    """
    try:
        price = get_current_price()
    except Exception as e:
        logger.error(f"fallback قیمت لحظه‌ای هم ناموفق بود: {e}")
        return None

    if sig_type == "LONG":
        if price >= tp1:
            return "win"
        if price <= sl:
            return "loss"
    elif sig_type == "SHORT":
        if price <= tp1:
            return "win"
        if price >= sl:
            return "loss"
    return None


def track_pending_signals() -> dict:
    """
    بررسی تمام سیگنال‌های pending در Supabase با استفاده از تاریخچهٔ
    واقعی high/low قیمت (نه فقط قیمت لحظه‌ای).

    Returns:
        خلاصه تعداد تغییرات
    """
    sb = get_supabase()

    result = sb.table("signal_history")\
               .select("*")\
               .eq("status", "pending")\
               .execute()

    signals = result.data
    if not signals:
        logger.info("هیچ سیگنال pending‌ای وجود ندارد.")
        return {"checked": 0, "wins": 0, "losses": 0, "expired": 0}

    stats = {"checked": len(signals), "wins": 0, "losses": 0, "expired": 0, "errors": 0}
    now = datetime.now(timezone.utc)

    for sig in signals:
        sig_id   = sig["id"]
        sig_type = sig.get("type")
        tp1      = sig.get("take_profit1")
        sl       = sig.get("stop_loss")
        created  = sig.get("created_at")

        if not created or tp1 is None or sl is None or sig_type not in ("LONG", "SHORT"):
            logger.warning(f"سیگنال {sig_id} فیلدهای لازم را ندارد — رد شد.")
            continue

        try:
            created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
        except Exception:
            logger.error(f"created_at نامعتبر برای سیگنال {sig_id}: {created}")
            continue

        hours_passed = (now - created_dt).total_seconds() / 3600
        tp1_f, sl_f = float(tp1), float(sl)

        new_status = None
        closed_at = now

        try:
            candles = fetch_ohlcv_since(interval=TRACK_INTERVAL, start_time=created_dt)
            if not candles.empty:
                outcome, ts = _resolve_from_candles(sig_type, tp1_f, sl_f, candles)
                if outcome:
                    new_status = outcome
                    closed_at = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else now
        except Exception as e:
            logger.error(f"خطا در دریافت کندل‌های ردیابی برای سیگنال {sig_id}: {e}")
            stats["errors"] += 1
            fb = _fallback_spot_check(sig_type, tp1_f, sl_f)
            if fb:
                new_status = fb

        # انقضا فقط وقتی چک می‌شود که هنوز به نتیجه‌ای نرسیده باشیم
        if new_status is None and hours_passed > EXPIRY_HOURS:
            new_status = "expired"
            closed_at = now

        if new_status == "win":
            stats["wins"] += 1
        elif new_status == "loss":
            stats["losses"] += 1
        elif new_status == "expired":
            stats["expired"] += 1

        if new_status:
            try:
                sb.table("signal_history").update({
                    "status":    new_status,
                    "closed_at": closed_at.isoformat() if hasattr(closed_at, "isoformat") else now.isoformat()
                }).eq("id", sig_id).execute()

                logger.info(
                    f"سیگنال {sig_id[:8]}... → {new_status} "
                    f"(tp1={tp1_f}, sl={sl_f}, at={closed_at})"
                )
            except Exception as e:
                logger.error(f"خطا در آپدیت سیگنال {sig_id}: {e}")

    logger.info(f"ردیابی تمام شد: {stats}")
    return stats


def get_performance_stats() -> dict:
    """
    آمار عملکرد کلی سیگنال‌های تاریخی
    """
    sb = get_supabase()

    result = sb.table("signal_history")\
               .select("type, status, confidence, created_at")\
               .neq("status", "pending")\
               .execute()

    signals = result.data
    if not signals:
        return {"message": "هنوز سیگنال بسته‌شده‌ای وجود ندارد."}

    total  = len(signals)
    wins   = sum(1 for s in signals if s["status"] == "win")
    losses = sum(1 for s in signals if s["status"] == "loss")

    win_rate = round(wins / total * 100, 1) if total > 0 else 0

    # breakdown به تفکیک نوع
    by_type = {}
    for sig in signals:
        t = sig.get("type", "?")
        if t not in by_type:
            by_type[t] = {"total": 0, "wins": 0}
        by_type[t]["total"] += 1
        if sig["status"] == "win":
            by_type[t]["wins"] += 1

    for t in by_type:
        n = by_type[t]["total"]
        w = by_type[t]["wins"]
        by_type[t]["win_rate"] = round(w / n * 100, 1) if n > 0 else 0

    # میانگین confidence سیگنال‌های win در برابر loss
    win_conf  = [s["confidence"] for s in signals
                 if s["status"] == "win" and s.get("confidence")]
    loss_conf = [s["confidence"] for s in signals
                 if s["status"] == "loss" and s.get("confidence")]

    return {
        "total":          total,
        "wins":           wins,
        "losses":         losses,
        "expired":        total - wins - losses,
        "win_rate_pct":   win_rate,
        "by_type":        by_type,
        "avg_conf_wins":  round(sum(win_conf)  / len(win_conf),  1) if win_conf  else None,
        "avg_conf_losses":round(sum(loss_conf) / len(loss_conf), 1) if loss_conf else None,
    }
