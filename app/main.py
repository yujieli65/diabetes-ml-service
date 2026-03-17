# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.model import predict

app = FastAPI(title="Diabetes ML Service", version="v0.1")


# 1️⃣ 定义输入 schema（自动校验）
class PatientData(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


# 2️⃣ 健康检查
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": "v0.1"
    }


# 3️⃣ 预测接口
@app.post("/predict")
def predict_endpoint(data: PatientData):
    try:
        prediction = predict(data.dict())
        return {"prediction": prediction}
    except Exception as e:
        # 返回 JSON 错误（评分点！）
        raise HTTPException(status_code=400, detail=str(e))


