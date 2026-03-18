import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import joblib

RANDOM_STATE = 42

# 1️⃣ 读取数据
Xy = load_diabetes(as_frame=True)
X = Xy.frame.drop(columns=["target"])
y = Xy.frame["target"]

# 2️⃣ 拆分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# 3️⃣ 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4️⃣ 模型训练
model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
model.fit(X_train_scaled, y_train)

# 5️⃣ 保存模型和 scaler
joblib.dump(model, "model/model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

# 6️⃣ 预测并计算 RMSE
y_pred = model.predict(X_test_scaled)

from sklearn.metrics import mean_squared_error
import numpy as np

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("RMSE:", rmse)
# 7️⃣ 可选：高风险 flag
threshold = 140
high_risk_pred = (y_pred > threshold).astype(int)
high_risk_true = (y_test > threshold).astype(int)

from sklearn.metrics import precision_score, recall_score
precision = precision_score(high_risk_true, high_risk_pred)
recall = recall_score(high_risk_true, high_risk_pred)
print(f"High-risk precision: {precision:.2f}, recall: {recall:.2f}")

# 8️⃣ 保存 metrics
metrics = {
    "rmse": rmse,
    "precision": precision,
    "recall": recall
}
joblib.dump(metrics, "model/metrics.pkl")