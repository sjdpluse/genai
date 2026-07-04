"""
ml_model.py — آموزش، ارزیابی و پیش‌بینی مدل ML
از Random Forest استفاده می‌کنیم:
  - بدون overfitting زیاد
  - feature importance داخلی
  - بدون نیاز به normalization دقیق
  - سریع برای این حجم داده
"""
import os
import joblib
import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

from config import MODEL_PATH, SCALER_PATH, LABEL_FORWARD_CANDLES

logger = logging.getLogger(__name__)

# ─── نگاشت label عددی به نام ───────────────────────────────
LABEL_MAP = {-1: "SHORT", 0: "WAIT", 1: "LONG"}

# ─── Embargo برای جلوگیری از نشتی برچسب‌های آینده‌نگر ────────
# هر برچسب در ردیف t با استفاده از قیمت t+LABEL_FORWARD_CANDLES ساخته
# می‌شود (نگاه به آینده). این یعنی نزدیک‌ترین LABEL_FORWARD_CANDLES
# ردیف به انتهای هر پنجرهٔ آموزشی، برچسبی دارند که از داخل بازهٔ تست
# بعدی (که در TimeSeriesSplit بلافاصله بعد از آموزش می‌آید) گرفته شده
# است — یعنی مدل هنگام اعتبارسنجی به‌طور غیرمستقیم اطلاعاتی از خودِ
# دورهٔ تست را در برچسب‌های آموزشی دیده (leakage در مرز فولدها).
# این باعث می‌شد CV Accuracy گزارش‌شده خوش‌بینانه‌تر از دقت واقعی مدل
# در حالت زنده باشد. راه‌حل استاندارد: «Purged/Embargoed Cross-
# Validation» — حذف آخرین EMBARGO ردیف از هر پنجرهٔ آموزشی قبل از fit.
EMBARGO = LABEL_FORWARD_CANDLES


def _purge_train_indices(train_idx: np.ndarray, embargo: int = EMBARGO) -> np.ndarray:
    """حذف آخرین `embargo` نمونه از انتهای پنجرهٔ آموزشی (نزدیک‌ترین به تست)"""
    if embargo <= 0 or len(train_idx) <= embargo:
        return train_idx
    return train_idx[:-embargo]


def _build_pipeline() -> Pipeline:
    """
    فکتوری Pipeline — تا هایپرپارامترها فقط در یک‌جا تعریف شوند
    (هم برای مدل اصلی و هم برای کالیبراسیون استفاده می‌شود، DRY).
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=30,       # جلوگیری از overfitting
            min_samples_split=50,
            class_weight="balanced",   # جبران عدم توازن WAIT/LONG/SHORT
            random_state=42,
            n_jobs=-1                  # استفاده از همه هسته‌های CPU
        ))
    ])


def train_model(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    آموزش مدل با Time-Series Cross Validation + Embargo
    (از آینده برای گذشته استفاده نمی‌کنیم — نه در فیچرها، نه در مرز فولدها)

    Returns:
        (model, metrics_dict)  — model ممکن است Pipeline کالیبره‌شده باشد
    """
    # حذف NaN‌ها — شامل ردیف‌های ابتدایی (اندیکاتورهایی که هنوز lookback
    # کافی ندارند) و ردیف‌های انتهایی که برچسب‌شان به‌دلیل نیاز به
    # forward_candles کندل آینده هنوز NaN است (رجوع کنید به
    # feature_engineer.create_labels — دیگر مثل قبل به‌اشتباه WAIT نیستند)
    valid_idx = X.dropna().index.intersection(y.dropna().index)

    X_clean = X.loc[valid_idx]
    y_clean = y.loc[valid_idx]

    if len(X_clean) < 200:
        raise ValueError(
            f"داده تمیز کافی برای آموزش وجود ندارد ({len(X_clean)} نمونه). "
            "limit را در run_training افزایش دهید یا محدودیت‌های اندیکاتورها را بررسی کنید."
        )

    logger.info(f"آموزش روی {len(X_clean)} نمونه")
    logger.info(f"توزیع کلاس‌ها:\n{y_clean.value_counts()}")

    pipeline = _build_pipeline()

    # ─── Walk-Forward Validation با Embargo ──────────────────
    # تقسیم‌بندی زمانی — مدل هرگز آینده را نمی‌بیند، و مرز بین آموزش/تست
    # با EMBARGO ردیف خالی جدا می‌شود تا برچسب‌های آینده‌نگر نشت نکنند.
    tscv = TimeSeriesSplit(n_splits=5)
    fold_scores = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_clean)):
        train_idx = _purge_train_indices(train_idx)
        if len(train_idx) == 0:
            logger.warning(f"Fold {fold+1}: پس از embargo، دادهٔ آموزشی باقی نماند — رد شد.")
            continue

        X_tr, X_te = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
        y_tr, y_te = y_clean.iloc[train_idx], y_clean.iloc[test_idx]

        pipeline.fit(X_tr, y_tr)
        preds = pipeline.predict(X_te)
        score = accuracy_score(y_te, preds)
        fold_scores.append(score)
        logger.info(f"Fold {fold+1}: accuracy = {score:.3f} (embargo={EMBARGO})")

    if not fold_scores:
        raise ValueError("هیچ fold معتبری برای اعتبارسنجی باقی نماند — داده یا embargo را بررسی کنید.")

    avg_score = np.mean(fold_scores)
    logger.info(f"میانگین accuracy در cross-validation (با embargo): {avg_score:.3f}")

    # ─── آموزش نهایی روی تمام داده (برای feature importance و به‌عنوان fallback) ──
    pipeline.fit(X_clean, y_clean)

    # گزارش کامل روی داده آموزش
    final_preds = pipeline.predict(X_clean)
    report = classification_report(y_clean, final_preds,
                                   target_names=["SHORT", "WAIT", "LONG"])
    logger.info(f"گزارش نهایی:\n{report}")

    # Feature Importance
    clf = pipeline.named_steps["clf"]
    importance_df = pd.DataFrame({
        "feature": X_clean.columns,
        "importance": clf.feature_importances_
    }).sort_values("importance", ascending=False)
    logger.info(f"۱۰ ویژگی مهم‌تر:\n{importance_df.head(10).to_string()}")

    # ─── کالیبراسیون احتمالات (Confidence Calibration) ──────
    # predict_proba در RandomForest یعنی «سهم آرای درخت‌ها»، نه احتمال
    # واقعیِ کالیبره‌شده. برای این‌که عدد confidence که در سیگنال نهایی
    # نمایش داده می‌شود واقعاً قابل‌اتکا باشد (وقتی مدل ۷۰٪ می‌گوید،
    # واقعاً حدود ۷۰٪ از دفعات درست باشد)، یک کالیبراسیون Sigmoid
    # (Platt Scaling) با اعتبارسنجی walk-forward (+ همان embargo بالا،
    # چون همان مشکل نشتی مرزی اینجا هم صادق است) روی آن اعمال می‌کنیم.
    # اگر به هر دلیلی (مثلاً حجم دادهٔ ناکافی برای یکی از کلاس‌ها در
    # یکی از fold ها) کالیبراسیون شکست بخورد، بدون کرش، به مدل خام
    # (uncalibrated) برمی‌گردیم — سیستم هرگز نباید به‌خاطر این مرحله
    # از کار بیفتد.
    final_model = pipeline
    calibrated = False
    try:
        calib_cv_raw = TimeSeriesSplit(n_splits=3)
        calib_splits = [
            (_purge_train_indices(tr), te)
            for tr, te in calib_cv_raw.split(X_clean)
            if len(_purge_train_indices(tr)) > 0
        ]
        if not calib_splits:
            raise ValueError("هیچ split معتبری برای کالیبراسیون پس از embargo باقی نماند.")

        calibrated_pipeline = CalibratedClassifierCV(
            estimator=_build_pipeline(),
            method="sigmoid",
            cv=calib_splits,
        )
        calibrated_pipeline.fit(X_clean, y_clean)
        final_model = calibrated_pipeline
        calibrated = True
        logger.info("کالیبراسیون احتمالات (Platt Scaling، با embargo) با موفقیت انجام شد.")
    except Exception as e:
        logger.warning(f"کالیبراسیون احتمالات ناموفق بود — از مدل خام استفاده می‌شود: {e}")

    metrics = {
        "cv_accuracy": round(avg_score, 3),
        "fold_scores": [round(s, 3) for s in fold_scores],
        "n_samples": len(X_clean),
        "embargo_candles": EMBARGO,
        "feature_importance": importance_df.head(10).to_dict("records"),
        "calibrated": calibrated,
    }

    return final_model, metrics


def save_model(model, feature_cols: list) -> None:
    """ذخیره مدل آموزش‌دیده (Pipeline یا CalibratedClassifierCV) + لیست ویژگی‌ها"""
    os.makedirs("model", exist_ok=True)
    joblib.dump({"pipeline": model, "features": feature_cols}, MODEL_PATH)
    logger.info(f"مدل ذخیره شد: {MODEL_PATH}")


def load_model() -> tuple:
    """بارگذاری مدل ذخیره‌شده"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "مدل آموزش‌دیده‌ای پیدا نشد. ابتدا /train را صدا بزنید."
        )
    data = joblib.load(MODEL_PATH)
    return data["pipeline"], data["features"]


def predict(pipeline,
            feature_cols: list,
            X_latest: pd.DataFrame,
            min_confidence: int = 55) -> dict:
    """
    پیش‌بینی سیگنال برای آخرین کندل.

    ⚠️ مدل دقیقاً همان ستون‌هایی را می‌خواهد که در آموزش دیده.
       اگر ستونی وجود نداشت، با ۰ پر می‌کنیم — بهتر از crash، اما
       این وضعیت باید در لاگ هشدار داده شود چون معمولاً نشانهٔ
       ناهماهنگی بین نسخهٔ feature_engineer آموزش و inference است.
    """
    # آخرین ردیف
    row = X_latest.iloc[[-1]].copy()

    # اطمینان از وجود تمام ستون‌های آموزشی — ستون‌های گمشده با ۰ پر می‌شوند
    missing = [c for c in feature_cols if c not in row.columns]
    if missing:
        logger.warning(
            f"⚠️ ناسازگاری نسخهٔ ویژگی‌ها: {len(missing)} ستون در inference وجود ندارند "
            f"ولی مدل با آن‌ها آموزش دیده — با ۰ پر می‌شوند (توصیه: مدل را دوباره آموزش دهید): "
            f"{missing[:5]}..."
        )
        for c in missing:
            row[c] = 0.0

    # انتخاب دقیقاً همان ستون‌ها با همان ترتیب آموزش
    row = row[feature_cols]

    # NaN باقیمانده را با ۰ پر کن (برای اندیکاتورهایی که کندل کافی نداشتند)
    n_nan = int(row.isna().sum().sum())
    if n_nan > 0:
        logger.warning(f"⚠️ {n_nan} مقدار NaN در ویژگی‌های آخرین کندل با ۰ پر شد.")
    row = row.fillna(0.0)

    # پیش‌بینی
    proba = pipeline.predict_proba(row)[0]
    classes = pipeline.classes_   # [-1, 0, 1]

    proba_dict = {LABEL_MAP[int(c)]: round(float(p) * 100, 1)
                  for c, p in zip(classes, proba)}

    # کلاس با بیشترین احتمال
    best_class = classes[np.argmax(proba)]
    confidence = round(float(np.max(proba)) * 100, 1)

    # اگر confidence کافی نیست → WAIT
    signal_type = LABEL_MAP[int(best_class)] if confidence >= min_confidence else "WAIT"

    return {
        "type": signal_type,
        "confidence": confidence,
        "probabilities": proba_dict,
        "raw_prediction": LABEL_MAP[int(best_class)]
    }
