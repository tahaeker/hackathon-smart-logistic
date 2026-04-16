"""
src/data_ingestion/api_client.py

Anlık hava durumu verisini OpenWeatherMap API'den çeken istemci.

Zırh Katmanı (3 seviye):
  1. Ağ hatası  → Mock veri döner (uygulama çalışmaya devam eder)
  2. API limiti → Mock veri döner
  3. Bozuk JSON → Mock veri döner + hata loglanır

Kullanım:
    from src.data_ingestion.api_client import LiveWeatherClient

    client  = LiveWeatherClient()
    weather = client.get_weather(lat=39.62, lon=37.05)
    print(weather["weather_severity_index"])   # → 0.07 (örnek)
    print(weather["is_mock"])                  # → False (gerçek API)
"""

import logging
import os
from typing import Any, Optional

import numpy as np

from config.config_loader import Config, ConfigNode, get_config

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Ağır kütüphaneler lazy import (sunum sırasında yükleme hatası önlemi)
# ──────────────────────────────────────────────────────────────
try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False
    logger.warning("'requests' paketi bulunamadı. Sadece Mock veri kullanılacak.")


# ══════════════════════════════════════════════════════════════
# Yardımcı: weather_severity_index hesaplayıcı
# ══════════════════════════════════════════════════════════════

def _compute_severity_index(
    precipitation_mm: float,
    wind_speed_kmh: float,
    visibility_km: float,
    humidity_pct: float,
) -> float:
    """
    preprocessor.py ile birebir aynı formülü kullanır:
      0.35 × norm_precip + 0.25 × norm_wind
    + 0.25 × (1 − norm_vis) + 0.15 × norm_humidity
    Sonuç: [0.0 – 1.0] aralığında float.
    """
    norm_precip   = min(precipitation_mm, 50.0) / 50.0
    norm_wind     = min(wind_speed_kmh, 100.0) / 100.0
    norm_vis_inv  = 1.0 - min(visibility_km, 30.0) / 30.0
    norm_humidity = min(humidity_pct, 100.0) / 100.0

    index = (
        0.35 * norm_precip
        + 0.25 * norm_wind
        + 0.25 * norm_vis_inv
        + 0.15 * norm_humidity
    )
    return round(float(np.clip(index, 0.0, 1.0)), 4)


def _wind_ms_to_kmh(ms: float) -> float:
    """OpenWeather m/s → km/h dönüşümü."""
    return round(ms * 3.6, 2)


def _visibility_m_to_km(m: float) -> float:
    """OpenWeather metre → km (max 20 km olarak sınırlanır)."""
    return round(min(m / 1000.0, 20.0), 2)


def _map_road_surface(weather_id: int, precipitation_mm: float) -> str:
    """
    OpenWeather condition id + yağış miktarına göre yol yüzey durumu tahmini.
    Gruplar: 2xx=thunderstorm, 3xx=drizzle, 5xx=rain, 6xx=snow, 7xx=atmosphere, 8xx=clear/cloudy
    """
    if weather_id // 100 == 6:
        return "icy"
    if weather_id // 100 in (2, 3, 5) or precipitation_mm > 0:
        return "wet"
    return "dry"


def _map_condition_str(weather_main: str) -> str:
    """OpenWeather 'main' alanını sistemin beklediği kategoriye eşler."""
    mapping = {
        "Clear":        "clear",
        "Clouds":       "cloudy",
        "Rain":         "rainy",
        "Drizzle":      "rainy",
        "Thunderstorm": "stormy",
        "Snow":         "snowy",
        "Mist":         "foggy",
        "Fog":          "foggy",
        "Haze":         "hazy",
        "Smoke":        "hazy",
        "Dust":         "hazy",
        "Sand":         "hazy",
        "Ash":          "hazy",
        "Squall":       "stormy",
        "Tornado":      "stormy",
    }
    return mapping.get(weather_main, "cloudy")


# ══════════════════════════════════════════════════════════════
# LiveWeatherClient
# ══════════════════════════════════════════════════════════════

class LiveWeatherClient:
    """
    OpenWeatherMap Current Weather API istemcisi.

    Her get_weather() çağrısı şu sözlüğü döner (ML ile uyumlu format):
    {
      "weather_condition"      : str,    # "clear" | "rainy" | ...
      "temperature_c"          : float,
      "feels_like_c"           : float,
      "precipitation_mm"       : float,
      "precipitation_type"     : str,    # "none" | "rain" | "snow"
      "wind_speed_kmh"         : float,
      "wind_direction_deg"     : int,
      "humidity_pct"           : float,
      "pressure_hpa"           : float,
      "visibility_km"          : float,
      "cloud_cover_pct"        : float,
      "uv_index"               : float,  # API'den gelmez; sabit 0
      "road_surface_condition" : str,    # "dry" | "wet" | "icy"
      "delay_risk_score"       : float,  # weather_severity_index ile aynı
      "weather_severity_index" : float,  # [0-1] bileşik risk skoru
      "is_mock"                : bool,   # True = gerçek API verisi yok
      "source"                 : str,    # "openweather" | "mock_config" | "hardcoded_fallback"
    }
    """

    def __init__(self, config: Optional[Config] = None):
        self.cfg      = config or get_config()
        self._ow_cfg  = self._load_ow_config()
        self._mock    = self._load_mock_config()
        self._api_key = self._resolve_api_key()
        self._base_url = getattr(self._ow_cfg, "base_url",
                                 "https://api.openweathermap.org/data/2.5/weather")
        self._timeout  = float(getattr(self._ow_cfg, "timeout_sec", 5))
        logger.info(
            "LiveWeatherClient başlatıldı. API key: %s",
            "***" + self._api_key[-4:] if len(self._api_key) > 8 else "(YOK)",
        )

    # ──────────────────────────────────────────────────────────
    # Ana metod
    # ──────────────────────────────────────────────────────────
    def get_weather(self, lat: float, lon: float) -> dict[str, Any]:
        """
        Verilen koordinat için anlık hava durumunu döner.

        Başarısız → Mock veri döner; uygulama asla çökmez.

        Args:
            lat: Enlem (örn. 39.62)
            lon: Boylam (örn. 37.05)

        Returns:
            ML uyumlu hava durumu sözlüğü (her zaman eksiksiz).
        """
        # API Key yoksa veya requests paketi eksikse doğrudan mock'a git
        if not self._api_key or not _REQUESTS_OK:
            reason = "API key eksik" if not self._api_key else "'requests' paketi yok"
            logger.info("Mock hava durumu kullanılıyor (%s).", reason)
            return self._mock_response(f"mock — {reason}")

        try:
            raw = self._fetch_raw(lat, lon)
            parsed = self._parse_response(raw)
            logger.info(
                "OpenWeather verisi alındı: lat=%.4f lon=%.4f | %s %.1f°C",
                lat, lon, parsed["weather_condition"], parsed["temperature_c"],
            )
            return parsed

        except _ApiLimitError as exc:
            logger.warning("OpenWeather API limiti aşıldı: %s. Mock kullanılıyor.", exc)
            return self._mock_response("mock — api_limit")

        except _NetworkError as exc:
            logger.warning("Ağ hatası: %s. Mock kullanılıyor.", exc)
            return self._mock_response("mock — network_error")

        except _ParseError as exc:
            logger.error("API yanıtı parse edilemedi: %s. Mock kullanılıyor.", exc)
            return self._mock_response("mock — parse_error")

        except Exception as exc:
            logger.error(
                "LiveWeatherClient beklenmeyen hata: %s. Mock kullanılıyor.", exc,
                exc_info=True,
            )
            return self._mock_response("mock — unexpected_error")

    # ──────────────────────────────────────────────────────────
    # HTTP isteği
    # ──────────────────────────────────────────────────────────
    def _fetch_raw(self, lat: float, lon: float) -> dict:
        """
        OpenWeatherMap API'ye HTTP GET isteği atar.
        Hata tiplerini özel exception'lara dönüştürür.
        """
        params = {
            "lat":   lat,
            "lon":   lon,
            "appid": self._api_key,
            "units": getattr(self._ow_cfg, "units", "metric"),
        }
        try:
            resp = requests.get(
                self._base_url,
                params=params,
                timeout=self._timeout,
            )
        except requests.exceptions.ConnectionError as exc:
            raise _NetworkError(f"Bağlantı hatası: {exc}") from exc
        except requests.exceptions.Timeout as exc:
            raise _NetworkError(f"Zaman aşımı ({self._timeout}s): {exc}") from exc
        except requests.exceptions.RequestException as exc:
            raise _NetworkError(f"Request hatası: {exc}") from exc

        # HTTP hata kodları
        if resp.status_code == 401:
            raise _ApiLimitError("Geçersiz API key (401).")
        if resp.status_code == 429:
            raise _ApiLimitError("API istek limiti aşıldı (429).")
        if resp.status_code != 200:
            raise _NetworkError(f"Beklenmeyen HTTP {resp.status_code}: {resp.text[:200]}")

        try:
            return resp.json()
        except ValueError as exc:
            raise _ParseError(f"JSON decode hatası: {exc}") from exc

    # ──────────────────────────────────────────────────────────
    # Yanıt dönüştürme
    # ──────────────────────────────────────────────────────────
    def _parse_response(self, raw: dict) -> dict[str, Any]:
        """
        OpenWeatherMap JSON yanıtını ML uyumlu sözlüğe dönüştürür.

        OpenWeather yanıt yapısı:
          raw["weather"][0]["main"]  → "Rain"
          raw["main"]["temp"]        → 18.5  (°C, units=metric)
          raw["wind"]["speed"]       → 5.2   (m/s)
          raw["visibility"]          → 8000  (metre)
          raw["rain"]["1h"]          → 2.5   (mm, opsiyonel)
        """
        try:
            # Hava durumu genel
            weather_list = raw.get("weather", [{}])
            weather_main = weather_list[0].get("main", "Clear") if weather_list else "Clear"
            weather_id   = weather_list[0].get("id", 800) if weather_list else 800

            # Sıcaklık
            main_block   = raw.get("main", {})
            temperature  = float(main_block.get("temp", 20.0))
            feels_like   = float(main_block.get("feels_like", temperature))
            humidity     = float(main_block.get("humidity", 55.0))
            pressure     = float(main_block.get("pressure", 1013.0))

            # Rüzgar (m/s → km/h)
            wind_block   = raw.get("wind", {})
            wind_kmh     = _wind_ms_to_kmh(float(wind_block.get("speed", 2.78)))
            wind_deg     = int(wind_block.get("deg", 180))

            # Görüş (metre → km)
            visibility_km = _visibility_m_to_km(float(raw.get("visibility", 10000)))

            # Bulut
            cloud_pct    = float(raw.get("clouds", {}).get("all", 0))

            # Yağış (opsiyonel alan)
            rain_block   = raw.get("rain", {})
            snow_block   = raw.get("snow", {})
            precip_mm    = float(rain_block.get("1h", rain_block.get("3h", 0.0)))
            snow_mm      = float(snow_block.get("1h", snow_block.get("3h", 0.0)))
            total_precip = precip_mm + snow_mm
            precip_type  = (
                "snow" if snow_mm > 0
                else "rain" if precip_mm > 0
                else "none"
            )

            # Türetilmiş alanlar
            condition    = _map_condition_str(weather_main)
            road_surface = _map_road_surface(weather_id, total_precip)
            severity     = _compute_severity_index(
                total_precip, wind_kmh, visibility_km, humidity
            )

            return {
                "weather_condition":      condition,
                "temperature_c":          round(temperature, 1),
                "feels_like_c":           round(feels_like, 1),
                "precipitation_mm":       round(total_precip, 2),
                "precipitation_type":     precip_type,
                "wind_speed_kmh":         wind_kmh,
                "wind_direction_deg":     wind_deg,
                "humidity_pct":           humidity,
                "pressure_hpa":           pressure,
                "visibility_km":          visibility_km,
                "cloud_cover_pct":        cloud_pct,
                "uv_index":               0.0,     # Current Weather API'de yok
                "road_surface_condition": road_surface,
                "delay_risk_score":       severity,
                "weather_severity_index": severity,
                "is_mock":                False,
                "source":                 "openweather",
            }

        except (KeyError, TypeError, ValueError) as exc:
            raise _ParseError(f"Yanıt yapısı beklenmedik: {exc} | raw={raw}") from exc

    # ──────────────────────────────────────────────────────────
    # Mock yanıtı
    # ──────────────────────────────────────────────────────────
    def _mock_response(self, source: str = "mock") -> dict[str, Any]:
        """
        settings.yaml'daki mock_weather değerlerinden yanıt üretir.
        Mock da weather_severity_index hesaplar; tamamen tutarlı veri döner.
        """
        m = self._mock
        precip   = float(getattr(m, "precipitation_mm", 0.0))
        wind     = float(getattr(m, "wind_speed_kmh", 10.0))
        vis      = float(getattr(m, "visibility_km", 20.0))
        humidity = float(getattr(m, "humidity_pct", 55.0))
        severity = _compute_severity_index(precip, wind, vis, humidity)

        return {
            "weather_condition":      str(getattr(m, "weather_condition", "clear")),
            "temperature_c":          float(getattr(m, "temperature_c", 20.0)),
            "feels_like_c":           float(getattr(m, "feels_like_c", 19.0)),
            "precipitation_mm":       precip,
            "precipitation_type":     str(getattr(m, "precipitation_type", "none")),
            "wind_speed_kmh":         wind,
            "wind_direction_deg":     int(getattr(m, "wind_direction_deg", 180)),
            "humidity_pct":           humidity,
            "pressure_hpa":           float(getattr(m, "pressure_hpa", 1013.0)),
            "visibility_km":          vis,
            "cloud_cover_pct":        float(getattr(m, "cloud_cover_pct", 10.0)),
            "uv_index":               float(getattr(m, "uv_index", 3.0)),
            "road_surface_condition": str(getattr(m, "road_surface_condition", "dry")),
            "delay_risk_score":       severity,
            "weather_severity_index": severity,
            "is_mock":                True,
            "source":                 source,
        }

    # ──────────────────────────────────────────────────────────
    # Config yardımcıları
    # ──────────────────────────────────────────────────────────
    def _load_ow_config(self) -> ConfigNode:
        try:
            return getattr(
                getattr(self.cfg, "external_apis", None), "openweather", None
            ) or ConfigNode({})
        except Exception:
            return ConfigNode({})

    def _load_mock_config(self) -> ConfigNode:
        try:
            return getattr(
                getattr(self.cfg, "external_apis", None), "mock_weather", None
            ) or ConfigNode({})
        except Exception:
            return ConfigNode({})

    def _resolve_api_key(self) -> str:
        """
        API key öncelik sırası:
          1. OPENWEATHER_API_KEY environment variable
          2. settings.yaml → external_apis.openweather.api_key
        """
        env_key = os.environ.get("OPENWEATHER_API_KEY", "").strip()
        if env_key:
            logger.debug("API key env değişkeninden alındı.")
            return env_key

        cfg_key = str(getattr(self._ow_cfg, "api_key", "")).strip()
        if cfg_key and cfg_key != "YOUR_OPENWEATHER_API_KEY":
            logger.debug("API key settings.yaml'dan alındı.")
            return cfg_key

        logger.warning(
            "OpenWeather API key bulunamadı. "
            "OPENWEATHER_API_KEY env değişkenini veya settings.yaml'ı ayarla. "
            "Mock veri kullanılacak."
        )
        return ""


# ══════════════════════════════════════════════════════════════
# Özel exception sınıfları
# ══════════════════════════════════════════════════════════════

class _NetworkError(Exception):
    """Ağ bağlantısı veya HTTP durum kodu hatası."""


class _ApiLimitError(Exception):
    """API key geçersiz veya istek limiti aşıldı."""


class _ParseError(Exception):
    """API yanıtı beklenmedik formatta."""


# ══════════════════════════════════════════════════════════════
# Hızlı test
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    client = LiveWeatherClient()

    # Gerçek API ile Samsun koordinatı
    result = client.get_weather(lat=41.2867, lon=36.33)

    print("\n=== Hava Durumu Yanıtı ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nKaynak        : {result['source']}")
    print(f"Mock mu?       : {result['is_mock']}")
    print(f"Severity Index : {result['weather_severity_index']}")
