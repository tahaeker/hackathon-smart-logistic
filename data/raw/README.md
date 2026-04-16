# Smart Logistics — Synthetic Dataset
## Anadolu Hackathon 2026 | Case 1

### Overview
This dataset simulates real-world delivery logistics operations in the Sivas region, Turkey.
It is designed to support delay prediction and route optimization models.
Inspired by Yandex Shifts dataset structure (weather + motion prediction).

---

### Files

#### 1. `routes.csv` — Route-level summary (200 routes)
Each row represents one delivery route with aggregated weather, traffic, and performance data.

| Column | Description |
|--------|-------------|
| route_id | Unique route identifier (RT-XXXX) |
| vehicle_id | Vehicle identifier |
| vehicle_type | van / truck / motorcycle / car |
| driver_id | Driver identifier |
| num_stops | Number of delivery stops |
| departure_planned | Planned departure datetime |
| departure_actual | Actual departure datetime |
| total_distance_km | Total route distance |
| planned_duration_min | Expected total route time (minutes) |
| actual_duration_min | Real total route time (minutes) |
| total_delay_min | Total delay vs plan (minutes) |
| on_time_delivery_rate | Fraction of stops delivered on time [0–1] |
| weather_condition | clear / cloudy / rain / snow / fog / wind |
| temperature_c | Air temperature (°C) |
| precipitation_mm | Precipitation amount (mm) |
| wind_speed_kmh | Wind speed (km/h) |
| humidity_pct | Relative humidity (%) |
| visibility_km | Visibility (km) |
| traffic_level | low / moderate / high / congested |
| road_incident | 1 if road incident occurred, else 0 |
| incident_severity | Incident-induced delay multiplier (0 = none) |
| overall_delay_factor | Combined multiplicative delay factor |

#### 2. `route_stops.csv` — Stop-level detail (~1,200 rows)
Each row is one delivery stop within a route.

| Column | Description |
|--------|-------------|
| route_id | Parent route identifier |
| stop_sequence | Order of stop in route (1-indexed) |
| stop_id | Unique stop identifier |
| latitude / longitude | Stop coordinates |
| distance_from_prev_km | Distance from previous stop |
| road_type | highway / urban / rural / mountain |
| planned_arrival | Expected arrival time |
| actual_arrival | Real arrival time |
| time_window_open | Earliest acceptable delivery time |
| time_window_close | Latest acceptable delivery time |
| planned_service_min | Expected service time at stop |
| actual_service_min | Real service time at stop |
| planned_travel_min | Expected travel time to this stop |
| actual_travel_min | Real travel time to this stop |
| delay_at_stop_min | Travel delay (actual - planned) |
| missed_time_window | 1 if delivery outside window, else 0 |
| delay_probability | Model-ready delay risk score [0–1] |
| package_count | Number of packages at this stop |
| package_weight_kg | Total package weight (kg) |

#### 3. `traffic_segments.csv` — Traffic snapshots (500 records)
Road segment speed and congestion observations.

| Column | Description |
|--------|-------------|
| segment_id | Unique segment identifier |
| center_lat / center_lon | Segment center coordinates |
| road_type | Road category |
| hour_of_day | Hour (0–23) |
| day_of_week | 0=Monday … 6=Sunday |
| is_rush_hour | 1 during peak hours (7–9, 17–19 weekdays) |
| is_weekend | 1 if Saturday or Sunday |
| free_flow_speed_kmh | Speed under ideal conditions |
| current_speed_kmh | Observed speed |
| congestion_ratio | 1 - (current/free_flow), higher = worse |
| traffic_level | low / moderate / high / congested |
| incident_reported | 1 if incident on segment |
| avg_vehicle_count_per_min | Traffic density |
| timestamp | Observation datetime |

#### 4. `weather_observations.csv` — Weather station readings (300 records)
Point-in-time weather measurements across the region.

| Column | Description |
|--------|-------------|
| obs_id | Observation identifier |
| timestamp | Measurement datetime |
| latitude / longitude | Station coordinates |
| weather_condition | Current condition category |
| temperature_c | Air temperature (°C) |
| feels_like_c | Apparent temperature (°C) |
| precipitation_mm | Precipitation in past hour |
| precipitation_type | rain / snow / none |
| wind_speed_kmh | Wind speed |
| wind_direction_deg | Wind direction (degrees) |
| humidity_pct | Relative humidity (%) |
| pressure_hpa | Atmospheric pressure (hPa) |
| visibility_km | Visibility distance |
| cloud_cover_pct | Cloud cover (%) |
| uv_index | UV index |
| road_surface_condition | dry / wet / icy / snow_covered |
| delay_risk_score | Composite weather-based risk [0–1] |

#### 5. `historical_delay_stats.csv` — Aggregated delay statistics (960 records)
Pre-aggregated delay statistics by road type, traffic level, weather, and time of day.
Useful for feature engineering and baseline modeling.

| Column | Description |
|--------|-------------|
| road_type | Road category |
| traffic_level | Traffic condition |
| weather_condition | Weather category |
| time_bucket | morning_rush / midday / evening_rush / night / early_morning |
| sample_count | Number of observations in this group |
| mean_delay_min | Average delay (minutes) |
| median_delay_min | Median delay (minutes) |
| p90_delay_min | 90th percentile delay |
| p95_delay_min | 95th percentile delay |
| std_delay_min | Standard deviation of delay |
| on_time_rate | Fraction of on-time deliveries |
| delay_probability | Probability of any delay occurring |

---

### Suggested ML Tasks
1. **Delay probability prediction** — binary/regression on `delay_probability` using weather + traffic features
2. **Stop reordering optimization** — minimize `missed_time_window` count given current conditions
3. **Route duration estimation** — predict `actual_duration_min` from planned + context features
4. **Anomaly detection** — flag routes with `overall_delay_factor` > threshold

### Evaluation Metrics (per case spec)
- On-time delivery rate
- Average route duration (predicted vs actual)
- Delay prediction accuracy (MAE / RMSE)
