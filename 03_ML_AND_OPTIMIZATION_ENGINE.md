# 03_ML_AND_OPTIMIZATION_ENGINE.md

## 1. Machine Learning Model (`src/ml/`)

- **Algorithm:** Use XGBoost Regressor to predict `delay_at_stop_min`.
- **Inputs:** `road_type`, `weather_severity_index`, `congestion_ratio`, `incident_reported`, `time_of_day_category`.
- **Explainability:** Ensure the model can output feature importances so the Dispatcher UI can explain *why* a delay is happening (e.g., "70% due to snow").
- **Pipeline:** Create a scikit-learn Pipeline with StandardScaling for numerical features and OneHotEncoding for categoricals. Save the trained pipeline as `model.joblib`.

## 2. Route Optimization Algorithm (`src/optimization/route_engine.py`)

This is the core logic for the "Reorders stops automatically" requirement.

- **Constraint:** The first stop (Warehouse/Start) and the final stop MUST remain fixed. Only intermediate stops can be reordered.
- **Algorithm Logic (Greedy / Permutation):**
  1. Receive current stop list: [A, B, C, D, E] (A and E are fixed).
  2. Generate valid permutations for intermediate stops (e.g., [A, C, B, D, E]).
  3. For each permutation, calculate the new estimated travel times using: `(Haversine Distance * 1.35 tortuosity factor) / Average Speed`.
  4. Pass the new spatial-temporal points to the ML Predictor to get new `delay_at_stop_min`.
  5. Check `time_window_close` constraints.
  6. Return the permutation that yields the lowest total route time and zero missed time windows.
