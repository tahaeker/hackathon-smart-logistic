# 04_MVP_AND_API_INTEGRATION.md

## 1. FastAPI Backend (`src/api/fastapi_app.py`)

Create RESTful endpoints:

- `POST /predict`: Accepts a stop's coordinates and live context, returns predicted delay minutes.
- `POST /optimize`: Accepts a JSON payload of a route's stops. Calls the `route_engine.py`, which calls the ML model, and returns the strictly optimized sequence of stops.

## 2. Live API Integration (`src/data_ingestion/api_client.py`)

For the MVP showcase, we need a function `get_live_weather(lat, lon)`:

- It should make a real HTTP request to the free `OpenWeather API` (using an API key stored in `config/settings.yaml`).
- Extract temperature, weather condition, and wind speed, formatting it to match the ML model's expected `weather_severity_index` input.

## 3. Streamlit Dashboard (`frontend/streamlit_dashboard.py`)

The UI must be Dispatcher-friendly, clean, and actionable.

- **Layout:**
  - Sidebar: Select a `route_id` from the dataset.
  - Main Panel Top: Display "Current Route Status" (Total Planned Time vs. ML Predicted Time based on LIVE weather).
  - Alert Box (Red/Yellow): E.g., "Warning: Stop #3 has a predicted delay of 45 mins due to LIVE Snow conditions."
  - Action Button: "Optimize Route".
  - Main Panel Bottom: Upon clicking optimize, show a Before/After comparison table of the stop sequence, highlighting the time saved.
- **Rule:** Do NOT put complex ML training code or data merging code in the Streamlit file. Streamlit should ONLY make requests to the FastAPI endpoints or directly call the pre-built `predictor` and `route_engine` classes.
