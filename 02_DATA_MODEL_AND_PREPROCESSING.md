# 02_DATA_MODEL_AND_PREPROCESSING.md

## 1. The Dataset & Entities

We have 5 core CSV files:

- `routes.csv`: Route summaries (route_id, num_stops, planned_duration_min).
- `route_stops.csv`: Granular stop data (route_id, stop_sequence, lat, lon, road_type, planned_arrival).
- `traffic_segments.csv`: Traffic sensors (lat, lon, current_speed_kmh, congestion_ratio, incident_reported, timestamp).
- `weather_observations.csv`: Weather sensors (lat, lon, temperature_c, weather_condition, wind_speed_kmh, timestamp).
- `historical_delay_stats.csv`: Aggregated past data.

## 2. Data Merging Logic (The "Join" Strategy)

To create the training dataset for the ML model, `route_stops.csv` is the base table.
For every stop in `route_stops.csv`:

1. **Spatial-Temporal Join (Weather):** Find the nearest row in `weather_observations.csv` based on Euclidean distance (lat/lon) AND closest `timestamp` to the stop's `planned_arrival`.
2. **Spatial-Temporal Join (Traffic):** Find the nearest row in `traffic_segments.csv` based on Euclidean distance (lat/lon) AND closest `timestamp` to the stop's `planned_arrival`.
3. **Route Data:** Left join `routes.csv` using `route_id`.

## 3. Feature Engineering Requirements

The `preprocessor.py` MUST generate the following features before feeding data to the ML model:

- `time_of_day_category`: Encode `planned_arrival` into (morning_rush, midday, evening_rush, night).
- `weather_severity_index`: A custom calculated float based on `wind_speed_kmh`, `precipitation_mm`, and `weather_condition` (e.g., snow=high severity, clear=low).
- `distance_to_next_stop`: Calculate using Haversine formula between current stop lat/lon and next stop lat/lon.

## 4. Target Variables for ML

- Primary Target: `delay_probability` (Binary or Float) OR `delay_at_stop_min` (Regression).
