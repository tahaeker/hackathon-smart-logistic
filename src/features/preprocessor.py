"""
src/features/preprocessor.py

İki ana sorumluluğu vardır:
  1. SpatialTemporalJoiner  — route_stops'u, hava durumu ve trafik gözlemleriyle
                              en yakın konum + zaman penceresine göre eşleştirir.
  2. FeatureEngineer        — Birleştirilmiş DataFrame üzerinde makine öğrenmesi
                              için türetilmiş öznitelikler (features) üretir.

Kullanım:
    from config.config_loader import get_config
    from src.data_ingestion.csv_reader import CSVReader
    from src.features.preprocessor import SpatialTemporalJoiner, FeatureEngineer

    cfg    = get_config()
    reader = CSVReader(cfg)
    dfs    = reader.load_all()

    joiner   = SpatialTemporalJoiner(cfg)
    joined   = joiner.join(dfs)

    engineer = FeatureEngineer(cfg)
    features = engineer.build(joined)
"""

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from config.config_loader import Config, get_config

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


# ══════════════════════════════════════════════════════════════
# Yardımcı fonksiyonlar
# ══════════════════════════════════════════════════════════════

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    İki nokta arasındaki Haversine mesafesini km cinsinden döner.
    Vektörize numpy işlemleriyle çalışır.
    """
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _ensure_datetime(df: pd.DataFrame, cols: list[str], context: str = "") -> pd.DataFrame:
    """
    Verilen kolonların datetime64 tipinde olduğunu garanti eder.
    Zaten datetime ise dokunmaz; değilse pd.to_datetime ile parse eder.
    """
    for col in cols:
        if col not in df.columns:
            continue
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            logger.debug("%s '%s' yeniden datetime'a parse ediliyor.", context, col)
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _radians_array(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """cKDTree için radyan koordinat matrisi oluşturur."""
    return np.column_stack([np.radians(lat), np.radians(lon)])


# ══════════════════════════════════════════════════════════════
# 1. Sınıf: SpatialTemporalJoiner
# ══════════════════════════════════════════════════════════════

class SpatialTemporalJoiner:
    """
    route_stops'u hava durumu (weather_observations) ve trafik
    (traffic_segments) verileriyle uzaysal-zamansal olarak birleştirir.

    Strateji:
    • Hava durumu  → Her stop için planned_arrival zamanına en yakın
                     ±WEATHER_WINDOW_H saatlik penceredeki en yakın
                     (haversine) gözlemi seçer.
    • Trafik       → Her stop için (hour_of_day, day_of_week) eşleşen
                     en yakın segment kaydını seçer (cKDTree ile).
    """

    WEATHER_WINDOW_H: int = 2        # ±saat
    MAX_WEATHER_DIST_KM: float = 50.0
    MAX_TRAFFIC_DIST_KM: float = 30.0

    def __init__(self, config: Optional[Config] = None):
        self.cfg = config or get_config()
        logger.info("SpatialTemporalJoiner başlatıldı.")

    # ──────────────────────────────────────────────────────────
    # Genel giriş noktası
    # ──────────────────────────────────────────────────────────
    def join(self, dataframes: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Tüm birleştirme adımlarını sırayla çalıştırır.

        Args:
            dataframes: CSVReader.load_all() çıktısı.

        Returns:
            Zenginleştirilmiş route_stops DataFrame'i.
        """
        try:
            stops   = dataframes.get("route_stops", pd.DataFrame())
            routes  = dataframes.get("routes", pd.DataFrame())
            weather = dataframes.get("weather_observations", pd.DataFrame())
            traffic = dataframes.get("traffic_segments", pd.DataFrame())

            if stops.empty:
                raise ValueError("route_stops verisi boş, birleştirme yapılamıyor.")
            if routes.empty:
                logger.warning("routes verisi boş; rota-düzey öznitelikler eklenemeyecek.")

            # 1. Rota meta bilgilerini ekle
            df = self._merge_route_meta(stops, routes)

            # 2. Hava durumu birleştirmesi
            if not weather.empty:
                df = self._join_weather(df, weather)
            else:
                logger.warning("weather_observations boş; hava durumu öznitelikleri atlandı.")

            # 3. Trafik birleştirmesi
            if not traffic.empty:
                df = self._join_traffic(df, traffic)
            else:
                logger.warning("traffic_segments boş; trafik öznitelikleri atlandı.")

            logger.info(
                "Birleştirme tamamlandı. Sonuç: %d satır, %d kolon.",
                len(df), len(df.columns),
            )
            return df

        except Exception as exc:
            logger.error("join() hatası: %s", exc, exc_info=True)
            raise

    # ──────────────────────────────────────────────────────────
    # Adım 1 — Rota meta bilgileri (routes → route_stops)
    # ──────────────────────────────────────────────────────────
    def _merge_route_meta(
        self, stops: pd.DataFrame, routes: pd.DataFrame
    ) -> pd.DataFrame:
        """
        routes tablosundan araç tipi, hava durumu, gecikme faktörü gibi
        rota-düzey bilgileri stop düzeyine indirir.
        """
        try:
            if routes.empty:
                return stops.copy()

            route_cols = [
                "route_id", "vehicle_type", "weather_condition",
                "traffic_level", "road_incident", "incident_severity",
                "temperature_c", "precipitation_mm",
                "wind_speed_kmh", "humidity_pct", "visibility_km",
            ]
            available = [c for c in route_cols if c in routes.columns]
            merged = stops.merge(
                routes[available].drop_duplicates("route_id"),
                on="route_id",
                how="left",
                suffixes=("", "_route"),
            )
            logger.debug(
                "Rota meta birleştirmesi: %d kolon eklendi.", len(available) - 1
            )
            return merged

        except Exception as exc:
            logger.error("_merge_route_meta hatası: %s", exc)
            return stops.copy()

    # ──────────────────────────────────────────────────────────
    # Adım 2 — Hava durumu (Spatial + Temporal)
    # ──────────────────────────────────────────────────────────
    def _join_weather(
        self, df: pd.DataFrame, weather: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Her stop satırı için:
          a. planned_arrival ± WEATHER_WINDOW_H saatlik zaman filtresi uygular.
          b. Kalan gözlemler içinden haversine en yakınını seçer.
          c. Eşleşen hava durumu özniteliklerini df'e ekler.
        """
        try:
            # Datetime garantisi
            df      = _ensure_datetime(df, ["planned_arrival"], "stops")
            weather = _ensure_datetime(weather, ["timestamp"], "weather")

            weather_cols = [
                "temperature_c", "feels_like_c", "precipitation_mm",
                "precipitation_type", "wind_speed_kmh", "wind_direction_deg",
                "humidity_pct", "pressure_hpa", "visibility_km",
                "cloud_cover_pct", "uv_index", "road_surface_condition",
                "delay_risk_score", "weather_condition",
            ]
            available_w = [c for c in weather_cols if c in weather.columns]

            # Çıktı kolonları için prefix
            prefix = "w_"
            result_cols = {c: f"{prefix}{c}" for c in available_w}

            matched_rows = []

            for _, stop in df.iterrows():
                stop_lat  = stop.get("latitude", np.nan)
                stop_lon  = stop.get("longitude", np.nan)
                stop_time = stop.get("planned_arrival", pd.NaT)

                # NaN koordinat veya NaT zaman → boş eşleşme
                if pd.isna(stop_lat) or pd.isna(stop_lon) or pd.isna(stop_time):
                    matched_rows.append({f"{prefix}{c}": np.nan for c in available_w})
                    continue

                # Zaman penceresi filtresi
                window = pd.Timedelta(hours=self.WEATHER_WINDOW_H)
                mask   = (
                    (weather["timestamp"] >= stop_time - window) &
                    (weather["timestamp"] <= stop_time + window)
                )
                w_window = weather[mask]

                if w_window.empty:
                    matched_rows.append({f"{prefix}{c}": np.nan for c in available_w})
                    continue

                # Haversine mesafesi hesapla
                dists = _haversine_km(
                    stop_lat, stop_lon,
                    w_window["latitude"].values,
                    w_window["longitude"].values,
                )
                min_idx  = np.argmin(dists)
                min_dist = dists[min_idx]

                if min_dist > self.MAX_WEATHER_DIST_KM:
                    matched_rows.append({f"{prefix}{c}": np.nan for c in available_w})
                    continue

                best = w_window.iloc[min_idx]
                matched_rows.append(
                    {f"{prefix}{c}": best[c] if c in best.index else np.nan
                     for c in available_w}
                )

            weather_df = pd.DataFrame(matched_rows, index=df.index)
            df = pd.concat([df, weather_df], axis=1)
            logger.info(
                "Hava durumu birleştirmesi tamamlandı. "
                "Eşleşen satır: %d / %d",
                weather_df.notna().any(axis=1).sum(), len(df),
            )
            return df

        except Exception as exc:
            logger.error("_join_weather hatası: %s", exc, exc_info=True)
            return df

    # ──────────────────────────────────────────────────────────
    # Adım 3 — Trafik (Spatial + Time-of-Day)
    # ──────────────────────────────────────────────────────────
    def _join_traffic(
        self, df: pd.DataFrame, traffic: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Her stop için:
          a. planned_arrival'dan hour_of_day ve day_of_week çıkarır.
          b. traffic_segments'i aynı (hour, dow) gruba filtreler.
          c. cKDTree ile en yakın segmenti bulur.
        """
        try:
            df      = _ensure_datetime(df, ["planned_arrival"], "stops")
            traffic = _ensure_datetime(traffic, ["timestamp"], "traffic")

            # Trafik üzerinde zaman özniteliklerini hazırla
            if "hour_of_day" not in traffic.columns:
                traffic = traffic.copy()
                traffic["hour_of_day"] = traffic["timestamp"].dt.hour
            if "day_of_week" not in traffic.columns:
                traffic = traffic.copy()
                traffic["day_of_week"] = traffic["timestamp"].dt.dayofweek

            traffic_cols = [
                "free_flow_speed_kmh", "current_speed_kmh", "congestion_ratio",
                "traffic_level", "incident_reported", "avg_vehicle_count_per_min",
            ]
            available_t = [c for c in traffic_cols if c in traffic.columns]
            prefix = "t_"

            matched_rows = []

            for _, stop in df.iterrows():
                stop_lat  = stop.get("latitude", np.nan)
                stop_lon  = stop.get("longitude", np.nan)
                stop_time = stop.get("planned_arrival", pd.NaT)

                if pd.isna(stop_lat) or pd.isna(stop_lon) or pd.isna(stop_time):
                    matched_rows.append({f"{prefix}{c}": np.nan for c in available_t})
                    continue

                hour = stop_time.hour
                dow  = stop_time.dayofweek

                # Aynı saat ve gün grubunu filtrele (±1 saat tolerans)
                mask = (
                    (traffic["hour_of_day"].between(max(0, hour - 1), min(23, hour + 1))) &
                    (traffic["day_of_week"] == dow)
                )
                t_group = traffic[mask]

                if t_group.empty:
                    # Sadece spatial eşleştir
                    t_group = traffic

                # cKDTree ile en yakın spatial eşleşme
                try:
                    tree = cKDTree(
                        _radians_array(
                            t_group["center_lat"].values,
                            t_group["center_lon"].values,
                        )
                    )
                    query_pt = np.radians([stop_lat, stop_lon])
                    dist_rad, idx = tree.query(query_pt, k=1)

                    # Radyan → km dönüşümü
                    dist_km = dist_rad * 6371.0

                    if dist_km > self.MAX_TRAFFIC_DIST_KM:
                        matched_rows.append({f"{prefix}{c}": np.nan for c in available_t})
                        continue

                    best = t_group.iloc[idx]
                    matched_rows.append(
                        {f"{prefix}{c}": best[c] if c in best.index else np.nan
                         for c in available_t}
                    )
                except Exception as kdtree_exc:
                    logger.debug("cKDTree hatası (satır atlandı): %s", kdtree_exc)
                    matched_rows.append({f"{prefix}{c}": np.nan for c in available_t})

            traffic_df = pd.DataFrame(matched_rows, index=df.index)
            df = pd.concat([df, traffic_df], axis=1)
            logger.info(
                "Trafik birleştirmesi tamamlandı. "
                "Eşleşen satır: %d / %d",
                traffic_df.notna().any(axis=1).sum(), len(df),
            )
            return df

        except Exception as exc:
            logger.error("_join_traffic hatası: %s", exc, exc_info=True)
            return df


# ══════════════════════════════════════════════════════════════
# 2. Sınıf: FeatureEngineer
# ══════════════════════════════════════════════════════════════

class FeatureEngineer:
    """
    Birleştirilmiş stop DataFrame'i üzerinde makine öğrenmesi
    için türetilmiş öznitelikler (features) üretir.

    Üretilen öznitelikler:
    ─ Zamansal  : time_of_day_category, is_rush_hour, is_weekend,
                  hour_of_day, day_of_week, month, day_of_month
    ─ Hava      : weather_severity_index, is_adverse_weather,
                  effective_visibility, precip_intensity_cat
    ─ Trafik    : speed_ratio, congestion_category,
                  congestion_delay_factor
    ─ Mekansal  : distance_to_next_stop (rota içi hesaplama)
    ─ Paket     : weight_per_package
    """

    # Gün bölümü sınır saatleri
    _TIME_BUCKETS = {
        "night":          (0, 6),
        "morning_rush":   (6, 9),
        "midday":         (9, 16),
        "afternoon_rush": (16, 19),
        "evening":        (19, 22),
        "late_night":     (22, 24),
    }

    def __init__(self, config: Optional[Config] = None):
        self.cfg = config or get_config()
        logger.info("FeatureEngineer başlatıldı.")

    # ──────────────────────────────────────────────────────────
    # Genel giriş noktası
    # ──────────────────────────────────────────────────────────
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tüm feature üretim adımlarını sırayla çalıştırır.

        Args:
            df: SpatialTemporalJoiner.join() çıktısı.

        Returns:
            Özniteliklerle zenginleştirilmiş DataFrame.
        """
        try:
            if df.empty:
                raise ValueError("Giriş DataFrame'i boş.")

            df = df.copy()

            df = self._datetime_features(df)
            df = self._weather_features(df)
            df = self._traffic_features(df)
            df = self._spatial_features(df)
            df = self._delay_features(df)
            df = self._package_features(df)

            logger.info(
                "Feature engineering tamamlandı. "
                "Toplam kolon: %d | Yeni feature sayısı tahmini: ~%d",
                len(df.columns),
                len(df.columns),
            )
            return df

        except Exception as exc:
            logger.error("build() hatası: %s", exc, exc_info=True)
            raise

    # ──────────────────────────────────────────────────────────
    # Zamansal öznitelikler
    # ──────────────────────────────────────────────────────────
    def _datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """planned_arrival üzerinden tüm zaman özniteliklerini üretir."""
        try:
            df = _ensure_datetime(df, ["planned_arrival", "actual_arrival"], "features")

            ref = df["planned_arrival"]

            df["hour_of_day"]  = ref.dt.hour
            df["day_of_week"]  = ref.dt.dayofweek          # 0=Pazartesi
            df["month"]        = ref.dt.month
            df["day_of_month"] = ref.dt.day
            df["is_weekend"]   = (ref.dt.dayofweek >= 5).astype(int)

            df["time_of_day_category"] = df["hour_of_day"].apply(
                self._classify_time_of_day
            )

            # is_rush_hour: mevcut kolona öncelik ver, yoksa türet
            if "is_rush_hour" not in df.columns:
                df["is_rush_hour"] = df["time_of_day_category"].isin(
                    ["morning_rush", "afternoon_rush"]
                ).astype(int)

            logger.debug("Zamansal öznitelikler üretildi.")
        except Exception as exc:
            logger.error("_datetime_features hatası: %s", exc)
        return df

    def _classify_time_of_day(self, hour: float) -> str:
        """Saat değerini gün bölümü kategorisine çevirir."""
        if pd.isna(hour):
            return "unknown"
        for category, (start, end) in self._TIME_BUCKETS.items():
            if start <= int(hour) < end:
                return category
        return "unknown"

    # ──────────────────────────────────────────────────────────
    # Hava durumu öznitelikleri
    # ──────────────────────────────────────────────────────────
    def _weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        weather_severity_index: normalize edilmiş bileşik hava riski skoru.
          = 0.35 * norm_precip
          + 0.25 * norm_wind
          + 0.25 * norm_visibility_inv
          + 0.15 * norm_humidity
        Tüm bileşenler [0, 1] aralığında normalize edilir.
        """
        try:
            # Hangi sütundan okuyacağımıza karar ver (joined veya rota-düzeyi)
            def _col(joined_name: str, route_name: str) -> pd.Series:
                if joined_name in df.columns:
                    return pd.to_numeric(df[joined_name], errors="coerce")
                elif route_name in df.columns:
                    return pd.to_numeric(df[route_name], errors="coerce")
                return pd.Series(np.nan, index=df.index)

            precip   = _col("w_precipitation_mm", "precipitation_mm").fillna(0)
            wind     = _col("w_wind_speed_kmh", "wind_speed_kmh").fillna(0)
            vis      = _col("w_visibility_km", "visibility_km").fillna(30)
            humidity = _col("w_humidity_pct", "humidity_pct").fillna(50)

            # Min-max normalize
            def _norm(s: pd.Series, max_val: float) -> pd.Series:
                return (s.clip(0, max_val) / max_val)

            norm_precip   = _norm(precip, 50)          # 50 mm max
            norm_wind     = _norm(wind, 100)            # 100 km/h max
            norm_vis_inv  = 1 - _norm(vis, 30)         # düşük görüş = yüksek risk
            norm_humidity = _norm(humidity, 100)

            df["weather_severity_index"] = (
                0.35 * norm_precip
                + 0.25 * norm_wind
                + 0.25 * norm_vis_inv
                + 0.15 * norm_humidity
            ).round(4)

            df["is_adverse_weather"] = (df["weather_severity_index"] > 0.4).astype(int)

            # Efektif görüş (joint kayıt öncelikli)
            df["effective_visibility"] = _col("w_visibility_km", "visibility_km")

            # Yağış yoğunluğu kategorisi
            df["precip_intensity_cat"] = pd.cut(
                precip,
                bins=[-1, 0, 2, 10, 50, 9999],
                labels=["none", "light", "moderate", "heavy", "extreme"],
            )

            logger.debug("Hava durumu öznitelikleri üretildi.")
        except Exception as exc:
            logger.error("_weather_features hatası: %s", exc)
        return df

    # ──────────────────────────────────────────────────────────
    # Trafik öznitelikleri
    # ──────────────────────────────────────────────────────────
    def _traffic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        speed_ratio: current_speed / free_flow_speed.
        congestion_category: ratio'ya göre kategorik etiket.
        congestion_delay_factor: gecikme çarpanı tahmini.
        """
        try:
            def _tcol(name: str) -> pd.Series:
                joined = f"t_{name}"
                if joined in df.columns:
                    return pd.to_numeric(df[joined], errors="coerce")
                elif name in df.columns:
                    return pd.to_numeric(df[name], errors="coerce")
                return pd.Series(np.nan, index=df.index)

            free_flow = _tcol("free_flow_speed_kmh").replace(0, np.nan)
            current   = _tcol("current_speed_kmh")
            congestion = _tcol("congestion_ratio")

            # speed_ratio: 1 = akıcı, 0'a yakın = çok sıkışık
            df["speed_ratio"] = (current / free_flow).clip(0, 1).round(4)

            # congestion_ratio: joined'dan veya speed_ratio türev
            if "t_congestion_ratio" in df.columns:
                df["congestion_ratio_eff"] = pd.to_numeric(
                    df["t_congestion_ratio"], errors="coerce"
                )
            else:
                df["congestion_ratio_eff"] = (1 - df["speed_ratio"]).clip(0, 1)

            df["congestion_category"] = pd.cut(
                df["congestion_ratio_eff"],
                bins=[-0.01, 0.1, 0.3, 0.6, 1.01],
                labels=["free_flow", "low", "moderate", "heavy"],
            )

            # Gecikme çarpanı: 1.0 (akıcı) → ~2.0 (tamamen sıkışık)
            df["congestion_delay_factor"] = (
                1.0 + df["congestion_ratio_eff"].fillna(0)
            ).round(4)

            logger.debug("Trafik öznitelikleri üretildi.")
        except Exception as exc:
            logger.error("_traffic_features hatası: %s", exc)
        return df

    # ──────────────────────────────────────────────────────────
    # Mekansal öznitelikler
    # ──────────────────────────────────────────────────────────
    def _spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        distance_to_next_stop: Rota içinde ardışık duraklar arasındaki
        haversine mesafesi (km). Son durağa NaN atanır.

        distance_from_prev_km zaten var ise tekrar hesaplanmaz;
        sadece normalize edilmiş bir versiyon türetilir.
        """
        try:
            if "distance_from_prev_km" in df.columns:
                df["distance_from_prev_km"] = pd.to_numeric(
                    df["distance_from_prev_km"], errors="coerce"
                )

            if "latitude" not in df.columns or "longitude" not in df.columns:
                logger.warning("latitude/longitude eksik; distance_to_next_stop üretilemedi.")
                return df

            df["latitude"]  = pd.to_numeric(df["latitude"], errors="coerce")
            df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

            # Rota-bazlı, durak sırasına göre sırala ve next stop mesafesi hesapla
            dist_to_next = pd.Series(np.nan, index=df.index)

            if "route_id" in df.columns and "stop_sequence" in df.columns:
                df_sorted = df.sort_values(["route_id", "stop_sequence"])
                for _, group in df_sorted.groupby("route_id"):
                    lats = group["latitude"].values
                    lons = group["longitude"].values
                    idxs = group.index.tolist()
                    for i in range(len(idxs) - 1):
                        if pd.notna(lats[i]) and pd.notna(lons[i]) and \
                           pd.notna(lats[i + 1]) and pd.notna(lons[i + 1]):
                            dist_to_next[idxs[i]] = _haversine_km(
                                lats[i], lons[i], lats[i + 1], lons[i + 1]
                            )

            df["distance_to_next_stop"] = dist_to_next.round(4)
            logger.debug("Mekansal öznitelikler üretildi.")
        except Exception as exc:
            logger.error("_spatial_features hatası: %s", exc)
        return df

    # ──────────────────────────────────────────────────────────
    # Gecikme öznitelikleri
    # ──────────────────────────────────────────────────────────
    def _delay_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Yalnızca inference zamanında da erişilebilen planned değerleri işler.
        actual_travel_min / delay_at_stop_min kullanan türetmeler (delay_ratio,
        cumulative_delay_at_stop, is_delayed) kasıtlı olarak hesaplanmaz —
        bunlar canlı tahmin sırasında bilinmez ve veri sızıntısı yaratır.
        """
        try:
            for col in ["planned_travel_min", "delay_at_stop_min"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            logger.debug("Gecikme öznitelikleri üretildi.")
        except Exception as exc:
            logger.error("_delay_features hatası: %s", exc)
        return df

    # ──────────────────────────────────────────────────────────
    # Paket öznitelikleri
    # ──────────────────────────────────────────────────────────
    def _package_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        weight_per_package: toplam ağırlık / paket sayısı.
        """
        try:
            for col in ["package_count", "package_weight_kg"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            if "package_count" in df.columns and "package_weight_kg" in df.columns:
                safe_count = df["package_count"].replace(0, np.nan)
                df["weight_per_package"] = (
                    df["package_weight_kg"] / safe_count
                ).round(4)

            logger.debug("Paket öznitelikleri üretildi.")
        except Exception as exc:
            logger.error("_package_features hatası: %s", exc)
        return df


# ══════════════════════════════════════════════════════════════
# Pipeline kolaylığı
# ══════════════════════════════════════════════════════════════

def run_preprocessing_pipeline(
    dataframes: dict[str, pd.DataFrame],
    config: Optional[Config] = None,
) -> pd.DataFrame:
    """
    SpatialTemporalJoiner + FeatureEngineer'ı tek çağrıda çalıştırır.

    Args:
        dataframes: CSVReader.load_all() çıktısı.
        config    : Config nesnesi (None ise singleton alınır).

    Returns:
        Feature-zenginleştirilmiş DataFrame.
    """
    cfg     = config or get_config()
    joiner  = SpatialTemporalJoiner(cfg)
    joined  = joiner.join(dataframes)

    engineer = FeatureEngineer(cfg)
    features = engineer.build(joined)

    return features


# ══════════════════════════════════════════════════════════════
# Hızlı test
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from src.data_ingestion.csv_reader import CSVReader

    cfg    = get_config()
    reader = CSVReader(cfg)
    dfs    = reader.load_all()

    features = run_preprocessing_pipeline(dfs, cfg)

    print(f"\nSonuç shape : {features.shape}")
    print(f"Kolonlar    : {features.columns.tolist()}")
    print(f"\nÖrnek satır:\n{features.iloc[0].to_dict()}")

    # Üretilen feature kontrolü
    expected = [
        "time_of_day_category", "weather_severity_index",
        "distance_to_next_stop", "congestion_delay_factor", "weight_per_package",
    ]
    for feat in expected:
        status = "OK" if feat in features.columns else "EKSIK"
        print(f"  [{status}] {feat}")
