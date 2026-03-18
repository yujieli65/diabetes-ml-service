import joblib
import os
import numpy as np

MODEL_PATH = os.path.join("model", "model.pkl")
SCALER_PATH = os.path.join("model", "scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def predict(features: dict):
    """
    输入字典，输出连续 risk score 和高风险 flag
    """
    # 按正确顺序构建矩阵
    X = np.array([[
        features["age"],
        features["sex"],
        features["bmi"],
        features["bp"],
        features["s1"],
        features["s2"],
        features["s3"],
        features["s4"],
        features["s5"],
        features["s6"],
    ]])

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)[0]

    # 把 numpy 类型转成原生 Python 类型
    prediction = float(y_pred)
    high_risk = bool(y_pred > 140)

    return {"prediction": prediction, "high_risk": high_risk}