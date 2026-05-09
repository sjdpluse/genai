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

from config import MODEL_PATH, SCALER_PATH

logger = logging.getLogger(__name__)

# ─── نگاشت label عددی به نام ───────────────────────────────
LABEL_MAP = {-1: "SHORT", 0: "WAIT", 1: "LONG"}


def train_model(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    آموزش مدل با Time-Series Cross Validation
    (از آینده برای گذشته استفاده نمی‌کنیم — lookahead bias نداریم)

    Returns:
        (pipeline, metrics_dict)
    """
    # حذف NaN‌های اول (ناشی از محاسبه اندیکاتورها)
    valid_idx = X.dropna().index.intersection(y.dropna().index)

    # حذف N کندل آخر که برچسب‌شان قابل اعتماد نیست
    valid_idx = valid_idx[:-10]

    X_clean = X.loc[valid_idx]
    y_clean = y.loc[valid_idx]

    logger.info(f"آموزش روی {len(X_clean)} نمونه")
    logger.info(f"توزیع کلاس‌ها:\n{y_clean.value_counts()}")

    # Pipeline: Scaler + RandomForest
    pipeline = Pipeline([
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

    # ─── Walk-Forward Validation ─────────────────────────────
    # تقسیم‌بندی زمانی — مدل هرگز آینده را نمی‌بیند
    tscv = TimeSeriesSplit(n_splits=5)
    fold_scores = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_clean)):
        X_tr, X_te = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
        y_tr, y_te = y_clean.iloc[train_idx], y_clean.iloc[test_idx]

        pipeline.fit(X_tr, y_tr)
        preds = pipeline.predict(X_te)
        score = accuracy_score(y_te, preds)
        fold_scores.append(score)
        logger.info(f"Fold {fold+1}: accuracy = {score:.3f}")

    avg_score = np.mean(fold_scores)
    logger.info(f"میانگین accuracy در cross-validation: {avg_score:.3f}")

    # ─── آموزش نهایی روی تمام داده ──────────────────────────
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

    metrics = {
        "cv_accuracy": round(avg_score, 3),
        "fold_scores": [round(s, 3) for s in fold_scores],
        "n_samples": len(X_clean),
        "feature_importance": importance_df.head(10).to_dict("records")
    }

    return pipeline, metrics


def save_model(pipeline: Pipeline, feature_cols: list) -> None:
    """ذخیره مدل آموزش‌دیده + لیست ویژگی‌ها"""
    os.makedirs("model", exist_ok=True)
    joblib.dump({"pipeline": pipeline, "features": feature_cols}, MODEL_PATH)
    logger.info(f"مدل ذخیره شد: {MODEL_PATH}")


def load_model() -> tuple:
    """بارگذاری مدل ذخیره‌شده"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "مدل آموزش‌دیده‌ای پیدا نشد. ابتدا /train را صدا بزنید."
        )
    data = joblib.load(MODEL_PATH)
    return data["pipeline"], data["features"]


def predict(pipeline: Pipeline,
            feature_cols: list,
            X_latest: pd.DataFrame,
            min_confidence: int = 55) -> dict:
    """
    پیش‌بینی سیگنال برای آخرین کندل

    Returns:
        dict با کلیدهای: type, confidence, probabilities, features_used
    """
    # آماده‌سازی آخرین ردیف
    row = X_latest[feature_cols].dropna(axis=1).iloc[[-1]]

    # فقط ستون‌هایی که مدل می‌شناسد
    common_cols = [c for c in feature_cols if c in row.columns]
    row = row[common_cols]

    # پیش‌بینی
    proba = pipeline.predict_proba(row)[0]
    classes = pipeline.classes_   # [-1, 0, 1]

    proba_dict = {LABEL_MAP[c]: round(float(p) * 100, 1)
                  for c, p in zip(classes, proba)}

    # کلاس با بیشترین احتمال
    best_class = classes[np.argmax(proba)]
    confidence = round(float(np.max(proba)) * 100, 1)

    # اگر confidence کافی نیست → WAIT
    signal_type = LABEL_MAP[best_class] if confidence >= min_confidence else "WAIT"

    return {
        "type": signal_type,
        "confidence": confidence,
        "probabilities": proba_dict,
        "raw_prediction": LABEL_MAP[best_class]
    }
