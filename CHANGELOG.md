## v0.1
- Model: LinearRegression
- Standardization: StandardScaler
- Random seed: 42
- RMSE: 60
- High-risk flag: None

## v0.2
- Model upgraded: Ridge(alpha=1.0) replaces LinearRegression
- Standardization unchanged, random seed 42 for reproducibility
- RMSE improved: v0.1 = 60 → v0.2 = 53.8
- Added high-risk flag (threshold = 140)
    - Precision: 0.70
    - Recall: 0.80