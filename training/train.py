# training/train.py
import json
import joblib
from pathlib import Path
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# 1️⃣ 固定随机种子
RANDOM_STATE = 42

# 2️⃣ 输出路径
MODEL_DIR = Path("./model")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

# 3️⃣ 加载数据
Xy = load_diabetes(as_frame=True)
X = Xy.frame.drop(columns=["target"])
y = Xy.frame["target"]

# 4️⃣ train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# 5️⃣ 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6️⃣ 训练模型
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 7️⃣ 预测
y_pred = model.predict(X_test_scaled)

# 8️⃣ 计算 RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {rmse:.2f}")

# 9️⃣ 保存模型和 scaler
joblib.dump({"model": model, "scaler": scaler}, MODEL_PATH)

# 10️⃣ 保存 metrics
metrics = {"rmse": rmse, "model": "LinearRegression", "version": "v0.1"}
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Model and metrics saved to {MODEL_DIR}")