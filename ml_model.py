"""
ml_model.py — Ensemble ML با sklearn-only (بدون xgboost/lightgbm)
RandomForest + ExtraTrees + GradientBoosting → VotingClassifier
"""
import os
import joblib
import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, 
    GradientBoostingClassifier, VotingClassifier
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

from config import MODEL_PATH, LABEL_FORWARD_CANDLES

logger = logging.getLogger(__name__)

LABEL_MAP = {-1: "SHORT", 0: "WAIT", 1: "LONG"}
EMBARGO = LABEL_FORWARD_CANDLES


def _purge_train_indices(train_idx: np.ndarray, embargo: int = EMBARGO) -> np.ndarray:
    """حذف آخرین embargo ردیف از پنجره آموزشی"""
    if embargo <= 0 or len(train_idx) <= embargo:
        return train_idx
    return train_idx[:-embargo]


def _build_pipeline() -> Pipeline:
    """Ensemble از ۳ مدل sklearn"""
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=30,
        min_samples_split=50,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    et = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=30,
        min_samples_split=50,
        class_weight="balanced",
        random_state=43,
        n_jobs=-1
    )

    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=44
    )

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("et", et), ("gb", gb)],
        voting="soft",
        n_jobs=-1
    )

    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", ensemble)
    ])


def train_model(X: pd.DataFrame, y: pd.Series) -> tuple:
    """آموزش مدل با Time-Series CV + Embargo"""

    # لاگ اولیه
    logger.info(f"train_model called with X.shape={X.shape}, y.shape={y.shape}")
    logger.info(f"X columns: {list(X.columns)[:5]}...")
    logger.info(f"y unique values: {y.unique()}")

    valid_idx = X.dropna().index.intersection(y.dropna().index)
    X_clean = X.loc[valid_idx]
    y_clean = y.loc[valid_idx]

    logger.info(f"After dropna: X_clean.shape={X_clean.shape}, y_clean.shape={y_clean.shape}")

    if len(X_clean) == 0:
        raise ValueError(
            f"داده تمیز صفر نمونه. "
            f"X has {X.isna().sum().sum()} NaN, y has {y.isna().sum()} NaN. "
            f"Original: X={X.shape}, y={y.shape}"
        )

    if len(X_clean) < 50:
        raise ValueError(
            f"داده ناکافی برای آموزش ({len(X_clean)} نمونه). "
            f"حداقل ۵۰ نمونه نیاز است."
        )

    logger.info(f"آموزش روی {len(X_clean)} نمونه")
    logger.info(f"توزیع کلاس‌ها:\n{y_clean.value_counts()}")

    pipeline = _build_pipeline()

    # Walk-Forward Validation
    n_splits = min(5, len(X_clean) // 100)  # کمترین split بر اساس داده
    if n_splits < 2:
        n_splits = 2
        logger.warning(f"داده کم — فقط {n_splits} split CV استفاده می‌شود")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_scores = []
    fold_f1s = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_clean)):
        train_idx = _purge_train_indices(train_idx)
        if len(train_idx) == 0:
            logger.warning(f"Fold {fold+1}: پس از embargo داده باقی نماند — رد شد.")
            continue

        X_tr, X_te = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
        y_tr, y_te = y_clean.iloc[train_idx], y_clean.iloc[test_idx]

        logger.info(f"Fold {fold+1}: train={len(X_tr)}, test={len(X_te)}")

        pipeline.fit(X_tr, y_tr)
        preds = pipeline.predict(X_te)
        score = accuracy_score(y_te, preds)
        f1 = f1_score(y_te, preds, average="weighted")
        fold_scores.append(score)
        fold_f1s.append(f1)
        logger.info(f"Fold {fold+1}: accuracy={score:.3f}, f1={f1:.3f}")

    if not fold_scores:
        raise ValueError("هیچ fold معتبری باقی نماند — داده خیلی کم است.")

    avg_score = np.mean(fold_scores)
    avg_f1 = np.mean(fold_f1s)
    logger.info(f"میانگین CV accuracy: {avg_score:.3f} | F1: {avg_f1:.3f}")

    # آموزش نهایی
    pipeline.fit(X_clean, y_clean)
    final_preds = pipeline.predict(X_clean)
    report = classification_report(y_clean, final_preds, target_names=["SHORT", "WAIT", "LONG"])
    logger.info(f"گزارش نهایی:\n{report}")

    # Feature Importance
    try:
        ensemble = pipeline.named_steps["clf"]
        rf_est = ensemble.named_estimators_["rf"]
        importance_df = pd.DataFrame({
            "feature": X_clean.columns,
            "importance": rf_est.feature_importances_
        }).sort_values("importance", ascending=False)
        logger.info(f"۱۵ ویژگی مهم‌تر:\n{importance_df.head(15).to_string()}")
    except Exception as e:
        logger.warning(f"Feature importance failed: {e}")
        importance_df = pd.DataFrame({"feature": X_clean.columns, "importance": 0})

    # کالیبراسیون
    final_model = pipeline
    calibrated = False
    try:
        calib_splits = 2 if len(X_clean) < 300 else 3
        calib_cv_raw = TimeSeriesSplit(n_splits=calib_splits)
        calib_splits_list = [
            (_purge_train_indices(tr), te)
            for tr, te in calib_cv_raw.split(X_clean)
            if len(_purge_train_indices(tr)) > 0
        ]
        if not calib_splits_list:
            raise ValueError("هیچ split معتبری برای کالیبراسیون باقی نماند.")

        calibrated_pipeline = CalibratedClassifierCV(
            estimator=_build_pipeline(),
            method="isotonic",
            cv=calib_splits_list,
        )
        calibrated_pipeline.fit(X_clean, y_clean)
        final_model = calibrated_pipeline
        calibrated = True
        logger.info("کالیبراسیون isotonic با موفقیت انجام شد.")
    except Exception as e:
        logger.warning(f"کالیبراسیون ناموفق — از مدل خام استفاده می‌شود: {e}")

    metrics = {
        "cv_accuracy": round(avg_score, 3),
        "cv_f1": round(avg_f1, 3),
        "fold_scores": [round(s, 3) for s in fold_scores],
        "n_samples": len(X_clean),
        "embargo_candles": EMBARGO,
        "feature_importance": importance_df.head(15).to_dict("records"),
        "calibrated": calibrated,
    }

    return final_model, metrics


def save_model(model, feature_cols: list) -> None:
    """ذخیره مدل + لیست ویژگی‌ها"""
    os.makedirs("model", exist_ok=True)
    joblib.dump({"pipeline": model, "features": feature_cols}, MODEL_PATH)
    logger.info(f"مدل ذخیره شد: {MODEL_PATH}")


def load_model() -> tuple:
    """بارگذاری مدل ذخیره‌شده"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("مدل آموزش‌دیده‌ای پیدا نشد. ابتدا /train را صدا بزنید.")
    data = joblib.load(MODEL_PATH)
    return data["pipeline"], data["features"]


def predict(pipeline, feature_cols: list, X_latest: pd.DataFrame, min_confidence: int = 65) -> dict:
    """پیش‌بینی سیگنال برای آخرین کندل"""
    row = X_latest.iloc[[-1]].copy()

    missing = [c for c in feature_cols if c not in row.columns]
    if missing:
        logger.warning(
            f"⚠️ {len(missing)} ستون در inference وجود ندارند — با ۰ پر می‌شوند: {missing[:5]}..."
        )
        for c in missing:
            row[c] = 0.0

    row = row[feature_cols]
    n_nan = int(row.isna().sum().sum())
    if n_nan > 0:
        logger.warning(f"⚠️ {n_nan} مقدار NaN با ۰ پر شد.")
    row = row.fillna(0.0)

    proba = pipeline.predict_proba(row)[0]
    classes = pipeline.classes_

    proba_dict = {LABEL_MAP[int(c)]: round(float(p) * 100, 1)
                  for c, p in zip(classes, proba)}

    best_class = classes[np.argmax(proba)]
    confidence = round(float(np.max(proba)) * 100, 1)

    signal_type = LABEL_MAP[int(best_class)] if confidence >= min_confidence else "WAIT"

    return {
        "type": signal_type,
        "confidence": confidence,
        "probabilities": proba_dict,
        "raw_prediction": LABEL_MAP[int(best_class)]
    }
