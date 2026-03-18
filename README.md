# Diabetes ML Service — v0.2

A lightweight ML service for **Virtual Diabetes Clinic Triage**.  
Predicts short-term diabetes progression and flags high-risk patients to prioritize follow-ups.

---

## 🚀 Features (v0.2)

- **Model:** Ridge Regression (`Ridge(alpha=1.0)`)
- **Preprocessing:** StandardScaler (same as v0.1)
- **Reproducibility:** Random seed = 42
- **Metrics:**
  - RMSE: 53.8 (v0.1 = 60)
  - High-risk flag (threshold = 140)
    - Precision: 0.70
    - Recall: 0.80
- **Endpoints:**
  - `GET /health` — service health + model version
  - `POST /predict` — return continuous progression score + high-risk flag
- **Deployment:** Docker image self-contained

---

## 🐳 Docker

Pull the release image from GitHub Container Registry:

```bash
docker pull ghcr.io/yujieli65/diabetes-api:v0.2