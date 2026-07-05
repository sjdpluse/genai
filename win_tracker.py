"""
win_tracker.py — 5m Candle Tracking
"""
import logging
from datetime import datetime, timezone
from data_fetcher import get_current_price, fetch_ohlcv_since
from supabase_client import get_supabase

logger = logging.getLogger(__name__)

TRACK_INTERVAL = "5m"
EXPIRY_HOURS = 72


def _resolve_from_candles(sig_type: str, tp1: float, sl: float, candles) -> tuple:
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
    try:
        price = get_current_price()
    except Exception as e:
        logger.error(f"fallback error: {e}")
        return None
    
    if sig_type == "LONG":
        if price >= tp1: return "win"
        if price <= sl: return "loss"
    elif sig_type == "SHORT":
        if price <= tp1: return "win"
        if price >= sl: return "loss"
    return None


def track_pending_signals() -> dict:
    sb = get_supabase()
    result = sb.table("signal_history").select("*").eq("status", "pending").execute()
    signals = result.data
    
    if not signals:
        return {"checked": 0, "wins": 0, "losses": 0, "expired": 0}
    
    stats = {"checked": len(signals), "wins": 0, "losses": 0, "expired": 0, "errors": 0}
    now = datetime.now(timezone.utc)
    
    for sig in signals:
        sig_id = sig["id"]
        sig_type = sig.get("type")
        tp1 = sig.get("take_profit1")
        sl = sig.get("stop_loss")
        created = sig.get("created_at")
        
        if not created or tp1 is None or sl is None or sig_type not in ("LONG", "SHORT"):
            continue
        
        try:
            created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
        except Exception:
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
            logger.error(f"خطا در ردیابی {sig_id}: {e}")
            stats["errors"] += 1
            fb = _fallback_spot_check(sig_type, tp1_f, sl_f)
            if fb:
                new_status = fb
        
        if new_status is None and hours_passed > EXPIRY_HOURS:
            new_status = "expired"
            closed_at = now
        
        if new_status == "win": stats["wins"] += 1
        elif new_status == "loss": stats["losses"] += 1
        elif new_status == "expired": stats["expired"] += 1
        
        if new_status:
            try:
                sb.table("signal_history").update({
                    "status": new_status,
                    "closed_at": closed_at.isoformat() if hasattr(closed_at, "isoformat") else now.isoformat()
                }).eq("id", sig_id).execute()
                logger.info(f"سیگنال {sig_id[:8]}... → {new_status}")
            except Exception as e:
                logger.error(f"update error {sig_id}: {e}")
    
    logger.info(f"ردیابی: {stats}")
    return stats


def get_performance_stats() -> dict:
    sb = get_supabase()
    result = sb.table("signal_history").select("type, status, confidence, created_at").neq("status", "pending").execute()
    signals = result.data
    
    if not signals:
        return {"message": "هنوز سیگنال بسته‌شده‌ای وجود ندارد."}
    
    total = len(signals)
    wins = sum(1 for s in signals if s["status"] == "win")
    losses = sum(1 for s in signals if s["status"] == "loss")
    win_rate = round(wins / total * 100, 1) if total > 0 else 0
    
    by_type = {}
    for sig in signals:
        t = sig.get("type", "?")
        by_type.setdefault(t, {"total": 0, "wins": 0})
        by_type[t]["total"] += 1
        if sig["status"] == "win":
            by_type[t]["wins"] += 1
    
    for t in by_type:
        n, w = by_type[t]["total"], by_type[t]["wins"]
        by_type[t]["win_rate"] = round(w / n * 100, 1) if n > 0 else 0
    
    win_conf = [s["confidence"] for s in signals if s["status"] == "win" and s.get("confidence")]
    loss_conf = [s["confidence"] for s in signals if s["status"] == "loss" and s.get("confidence")]
    
    avg_rr = 1.75
    
    return {
        "total": total,
        "wins": wins,
        "losses": losses,
        "expired": total - wins - losses,
        "win_rate_pct": win_rate,
        "by_type": by_type,
        "avg_conf_wins": round(sum(win_conf) / len(win_conf), 1) if win_conf else None,
        "avg_conf_losses": round(sum(loss_conf) / len(loss_conf), 1) if loss_conf else None,
        "estimated_expectancy": round((win_rate/100 * avg_rr) - ((100-win_rate)/100 * 1), 3),
        "sharpe_approx": round((win_rate/100 * avg_rr - (100-win_rate)/100) / 1.5, 3) if total > 20 else None,
    }
