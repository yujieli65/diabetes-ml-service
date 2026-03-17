# app/model.py
import joblib
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "model.pkl"

# 全局加载（避免每次请求重复加载）
model_bundle = joblib.load(MODEL_PATH)
model = model_bundle["model"]
scaler = model_bundle["scaler"]


def predict(features: dict) -> float:
    """
    输入: dict -> {feature_name: value}
    输出: float prediction
    """
    # 保证顺序一致（非常重要）
    feature_order = [
        "age", "sex", "bmi", "bp",
        "s1", "s2", "s3", "s4", "s5", "s6"
    ]

    X = np.array([[features[f] for f in feature_order]])
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]
    return float(pred)
