"""
ml_model.py — Ensemble XGBoost + LightGBM + Random Forest
"""
import os
import joblib
import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    
try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from config import MODEL_PATH, LABEL_FORWARD_CANDLES

logger = logging.getLogger(__name__)

LABEL_MAP = {-1: "SHORT", 0: "WAIT", 1: "LONG"}
EMBARGO = LABEL_FORWARD_CANDLES


def _purge_train_indices(train_idx: np.ndarray, embargo: int = EMBARGO) -> np.ndarray:
    if embargo <= 0 or len(train_idx) <= embargo:
        return train_idx
    return train_idx[:-embargo]


def _build_pipeline() -> Pipeline:
    estimators = []
    
    if HAS_XGB:
        estimators.append(("xgb", XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1,
            eval_metric="mlogloss",
        )))
    
    if HAS_LGB:
        estimators.append(("lgb", LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbose=-1,
        )))
    
    estimators.append(("rf", RandomForestClassifier(
        n_estimators=300, max_depth=10,
        min_samples_leaf=50, min_samples_split=80,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )))
    
    clf = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1) if len(estimators) > 1 else estimators[0][1]
    
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def train_model(X: pd.DataFrame, y: pd.Series) -> tuple:
    valid_idx = X.dropna().index.intersection(y.dropna().index)
    X_clean = X.loc[valid_idx]
    y_clean = y.loc[valid_idx]
    
    if len(X_clean) < 500:
        raise ValueError(f"داده ناکافی: {len(X_clean)} نمونه")
    
    logger.info(f"آموزش روی {len(X_clean)} نمونه")
    logger.info(f"توزیع کلاس‌ها:\n{y_clean.value_counts()}")
    
    pipeline = _build_pipeline()
    
    tscv = TimeSeriesSplit(n_splits=5)
    fold_scores = []
    fold_f1s = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_clean)):
        train_idx = _purge_train_indices(train_idx)
        if len(train_idx) == 0:
            continue
        
        X_tr, X_te = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
        y_tr, y_te = y_clean.iloc[train_idx], y_clean.iloc[test_idx]
        
        pipeline.fit(X_tr, y_tr)
        preds = pipeline.predict(X_te)
        fold_scores.append(accuracy_score(y_te, preds))
        fold_f1s.append(f1_score(y_te, preds, average="weighted"))
        logger.info(f"Fold {fold+1}: acc={fold_scores[-1]:.3f}, f1={fold_f1s[-1]:.3f}")
    
    if not fold_scores:
        raise ValueError("هیچ fold معتبری باقی نماند")
    
    avg_score = np.mean(fold_scores)
    avg_f1 = np.mean(fold_f1s)
    logger.info(f"CV Accuracy: {avg_score:.3f} | F1: {avg_f1:.3f}")
    
    pipeline.fit(X_clean, y_clean)
    final_preds = pipeline.predict(X_clean)
    report = classification_report(y_clean, final_preds, target_names=["SHORT", "WAIT", "LONG"])
    logger.info(f"گزارش نهایی:\n{report}")
    
    # Feature Importance
    try:
        if hasattr(pipeline.named_steps["clf"], "estimators_"):
            rf_est = next((est for name, est in pipeline.named_steps["clf"].estimators_ if name == "rf"), None)
            if rf_est is None:
                rf_est = pipeline.named_steps["clf"].estimators_[0][1]
        else:
            rf_est = pipeline.named_steps["clf"]
        
        importance_df = pd.DataFrame({
            "feature": X_clean.columns,
            "importance": rf_est.feature_importances_
        }).sort_values("importance", ascending=False)
        logger.info(f"۱۵ ویژگی مهم:\n{importance_df.head(15).to_string()}")
    except Exception as e:
        logger.warning(f"Feature importance failed: {e}")
        importance_df = pd.DataFrame({"feature": X_clean.columns, "importance": 0})
    
    # Calibration
    final_model = pipeline
    calibrated = False
    try:
        calib_cv_raw = TimeSeriesSplit(n_splits=3)
        calib_splits = [(_purge_train_indices(tr), te) for tr, te in calib_cv_raw.split(X_clean) if len(_purge_train_indices(tr)) > 0]
        if not calib_splits:
            raise ValueError("No valid calibration splits")
        
        calibrated_pipeline = CalibratedClassifierCV(
            estimator=_build_pipeline(), method="isotonic", cv=calib_splits,
        )
        calibrated_pipeline.fit(X_clean, y_clean)
        final_model = calibrated_pipeline
        calibrated = True
        logger.info("کالیبراسیون isotonic موفق.")
    except Exception as e:
        logger.warning(f"کالیبراسیون ناموفق: {e}")
    
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
    os.makedirs("model", exist_ok=True)
    joblib.dump({"pipeline": model, "features": feature_cols}, MODEL_PATH)
    logger.info(f"مدل ذخیره شد: {MODEL_PATH}")


def load_model() -> tuple:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("مدل پیدا نشد. ابتدا /train را صدا بزنید.")
    data = joblib.load(MODEL_PATH)
    return data["pipeline"], data["features"]


def predict(pipeline, feature_cols: list, X_latest: pd.DataFrame,
            min_confidence: int = 65) -> dict:
    row = X_latest.iloc[[-1]].copy()
    
    missing = [c for c in feature_cols if c not in row.columns]
    if missing:
        logger.warning(f"ستون‌های گمشده: {missing[:5]}... — با ۰ پر می‌شوند")
        for c in missing:
            row[c] = 0.0
    
    row = row[feature_cols]
    n_nan = int(row.isna().sum().sum())
    if n_nan > 0:
        logger.warning(f"{n_nan} NaN با ۰ پر شد")
    row = row.fillna(0.0)
    
    proba = pipeline.predict_proba(row)[0]
    classes = pipeline.classes_
    
    proba_dict = {LABEL_MAP[int(c)]: round(float(p) * 100, 1) for c, p in zip(classes, proba)}
    best_class = classes[np.argmax(proba)]
    confidence = round(float(np.max(proba)) * 100, 1)
    
    signal_type = LABEL_MAP[int(best_class)] if confidence >= min_confidence else "WAIT"
    
    return {
        "type": signal_type,
        "confidence": confidence,
        "probabilities": proba_dict,
        "raw_prediction": LABEL_MAP[int(best_class)]
    }
