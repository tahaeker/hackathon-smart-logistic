# 01_PROJECT_MASTER_BLUEPRINT.md

## 1. Project Overview

This project is a "Smart Logistics" real-time delivery route optimization system developed for a Hackathon. The goal is to predict delays at delivery stops based on weather and traffic, and dynamically reorder the stops to minimize missed time windows.

## 2. Architecture & Modularity (CRITICAL)

The system MUST strictly follow a modular architecture. Spaghetti code is strictly forbidden.
Directory Structure:
/smart-logistics-app
  ├── data/ (Raw CSV files)
  ├── models/ (Saved XGBoost/ML models .pkl)
  ├── config/ (settings.yaml, config_loader.py)
  ├── src/
  │   ├── data_ingestion/ (csv_reader.py, api_client.py)
  │   ├── features/ (preprocessor.py, feature_engineering.py)
  │   ├── ml/ (trainer.py, predictor.py)
  │   ├── optimization/ (route_engine.py)
  │   └── api/ (fastapi_app.py)
  └── frontend/ (streamlit_dashboard.py)

## 3. The "Two-Phase" Hybrid Strategy (IMPORTANT)

- **Phase 1 (Training):** The Machine Learning model MUST be trained ONLY on the provided synthetic historical CSV dataset (simulating the year 2025 in Sivas, Turkey).
- **Phase 2 (Live MVP Inference):** For the Streamlit MVP demonstration, the system will accept a Route, but it will fetch LIVE current data from `OpenWeather API` (for weather) and use `Haversine formula` (for distances) to simulate a real-time environment.

## 4. Tech Stack

- Data Manipulation: Pandas, NumPy
- Machine Learning: Scikit-learn, XGBoost (Do NOT use Deep Learning)
- Backend/API: FastAPI, Pydantic
- Frontend/MVP: Streamlit
- Geographic Math: Haversine formula (No external heavy mapping APIs like Google Maps for routing to save time/complexity, use Haversine * 1.3 tortuosity factor for road distance estimation).
