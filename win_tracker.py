"""
win_tracker.py — ردیاب خودکار نتیجه سیگنال‌ها
هر ساعت اجرا می‌شود و سیگنال‌های pending را بررسی می‌کند
"""
import logging
from datetime import datetime, timezone
from data_fetcher import get_current_price
from supabase_client import get_supabase

logger = logging.getLogger(__name__)


def track_pending_signals() -> dict:
    """
    بررسی تمام سیگنال‌های pending در Supabase
    اگر قیمت به TP1 رسید → win
    اگر به SL رسید → loss
    اگر ۷۲ ساعت گذشت → expired

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

    try:
        current_price = get_current_price()
    except Exception as e:
        logger.error(f"خطا در دریافت قیمت: {e}")
        return {"error": str(e)}

    stats = {"checked": len(signals), "wins": 0, "losses": 0, "expired": 0}
    now = datetime.now(timezone.utc)

    for sig in signals:
        sig_id    = sig["id"]
        sig_type  = sig.get("type")
        tp1       = sig.get("take_profit1")
        sl        = sig.get("stop_loss")
        created   = sig.get("created_at")
        new_status = None

        # بررسی انقضا (۷۲ ساعت)
        if created:
            try:
                created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                hours_passed = (now - created_dt).total_seconds() / 3600
                if hours_passed > 72:
                    new_status = "expired"
                    stats["expired"] += 1
            except Exception:
                pass

        if new_status is None and tp1 is not None and sl is not None:
            if sig_type == "LONG":
                if current_price >= float(tp1):
                    new_status = "win"
                    stats["wins"] += 1
                elif current_price <= float(sl):
                    new_status = "loss"
                    stats["losses"] += 1

            elif sig_type == "SHORT":
                if current_price <= float(tp1):
                    new_status = "win"
                    stats["wins"] += 1
                elif current_price >= float(sl):
                    new_status = "loss"
                    stats["losses"] += 1

        if new_status:
            try:
                sb.table("signal_history").update({
                    "status":    new_status,
                    "closed_at": now.isoformat()
                }).eq("id", sig_id).execute()

                logger.info(
                    f"سیگنال {sig_id[:8]}... → {new_status} "
                    f"(price={current_price}, tp1={tp1}, sl={sl})"
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
