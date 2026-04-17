"""
src/api/fastapi_app.py

Akıllı Lojistik ve Rota Optimizasyonu — FastAPI Uygulaması

Endpoint'ler:
  GET  /health          → Sistem ve model sağlık kontrolü
  POST /predict         → Tek durak için gecikme tahmini
  POST /optimize        → Rota optimizasyonu (durak sıralaması)

Çalıştırma:
  uvicorn src.api.fastapi_app:app --host 0.0.0.0 --port 8000 --reload

Swagger UI:
  http://localhost:8000/docs
ReDoc:
  http://localhost:8000/redoc
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from config.config_loader import Config, get_config

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
# Pydantic Şemaları (Request & Response modelleri)
# ══════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────
# Ortak / paylaşılan
# ──────────────────────────────────────────────────────────────

class WeatherSnapshot(BaseModel):
    """Hava durumu anlık görüntüsü (API yanıtlarında kullanılır)."""
    weather_condition:      str
    temperature_c:          float
    precipitation_mm:       float
    wind_speed_kmh:         float
    visibility_km:          float
    humidity_pct:           float
    weather_severity_index: float
    road_surface_condition: str
    is_mock:                bool
    source:                 str


# ──────────────────────────────────────────────────────────────
# POST /predict
# ──────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """
    Tek bir durak için gecikme tahmini isteği.

    Örnek JSON:
    {
      "latitude": 39.62,
      "longitude": 37.05,
      "planned_arrival": "2025-06-15T08:30:00",
      "road_type": "highway",
      "planned_travel_min": 25.0,
      "planned_service_min": 12.0,
      "package_count": 10,
      "distance_from_prev_km": 15.5
    }
    """
    latitude:              float = Field(..., ge=-90, le=90,
                                         description="Durak enlemi")
    longitude:             float = Field(..., ge=-180, le=180,
                                         description="Durak boylamı")
    planned_arrival:       datetime = Field(...,
                                            description="Planlanan varış zamanı (ISO 8601)")
    road_type:             Optional[str]  = Field("highway",
                                                  description="Yol tipi: highway|rural|mountain|urban")
    vehicle_type:          Optional[str]  = Field("van",
                                                  description="Araç tipi: van|car|truck")
    planned_travel_min:    Optional[float] = Field(20.0, ge=0,
                                                   description="Planlanan seyahat süresi (dk)")
    planned_service_min:   Optional[float] = Field(10.0, ge=0,
                                                   description="Planlanan hizmet süresi (dk)")
    package_count:         Optional[int]  = Field(5, ge=0,
                                                  description="Paket sayısı")
    package_weight_kg:     Optional[float] = Field(10.0, ge=0,
                                                   description="Toplam yük (kg)")
    distance_from_prev_km: Optional[float] = Field(10.0, ge=0,
                                                    description="Bir önceki durağa mesafe (km)")
    delay_probability:     Optional[float] = Field(0.2, ge=0, le=1,
                                                   description="Tarihsel gecikme olasılığı")

    @field_validator("road_type")
    @classmethod
    def validate_road_type(cls, v: Optional[str]) -> Optional[str]:
        allowed = {"highway", "rural", "mountain", "urban", "residential"}
        if v and v.lower() not in allowed:
            raise ValueError(f"road_type '{v}' geçersiz. İzin verilenler: {allowed}")
        return v.lower() if v else v


class PredictResponse(BaseModel):
    """Gecikme tahmini yanıtı."""
    predicted_delay_min:    float
    confidence_interval:    Optional[dict[str, float]] = None
    weather:                WeatherSnapshot
    feature_summary:        dict[str, Any]
    status:                 str
    elapsed_ms:             float


# ──────────────────────────────────────────────────────────────
# POST /optimize
# ──────────────────────────────────────────────────────────────

class StopInput(BaseModel):
    """
    Tek bir durak verisi (optimize isteğinde kullanılır).

    Minimum gereksinim: stop_id, latitude, longitude, stop_sequence.
    Diğer alanlar boş bırakılabilir; sistem varsayılan değerleri kullanır.
    """
    stop_id:               str   = Field(..., description="Benzersiz durak kimliği")
    stop_sequence:         int   = Field(..., ge=1, description="Rota içi sıra numarası")
    latitude:              float = Field(..., ge=-90,  le=90)
    longitude:             float = Field(..., ge=-180, le=180)
    planned_arrival:       Optional[datetime] = None
    time_window_open:      Optional[datetime] = None
    time_window_close:     Optional[datetime] = None
    planned_service_min:   Optional[float]    = Field(10.0, ge=0)
    planned_travel_min:    Optional[float]    = Field(20.0, ge=0)
    distance_from_prev_km: Optional[float]    = Field(0.0,  ge=0)
    delay_probability:     Optional[float]    = Field(0.2,  ge=0, le=1)
    road_type:             Optional[str]      = "highway"
    package_count:         Optional[int]      = Field(5,    ge=0)
    package_weight_kg:     Optional[float]    = Field(10.0, ge=0)

    @field_validator("stop_sequence")
    @classmethod
    def seq_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("stop_sequence 1'den küçük olamaz.")
        return v


class OptimizeRequest(BaseModel):
    """
    Rota optimizasyon isteği.

    Örnek JSON:
    {
      "route_id": "RT-0001",
      "departure_time": "2025-06-15T08:00:00",
      "stops": [ {...}, {...}, ... ]
    }
    """
    route_id:        str              = Field(..., description="Rota kimliği")
    departure_time:  Optional[datetime] = Field(None,
                                                description="Hareket zamanı (None = şimdiki an)")
    stops:           list[StopInput]  = Field(..., min_length=2,
                                             description="En az 2 durak gereklidir")
    fixed_positions: Optional[list[int]] = Field(
        None,
        description="Sabit tutulacak 0-tabanlı pozisyonlar (None = tüm duraklar dinamik)"
    )

    @field_validator("stops")
    @classmethod
    def validate_stop_sequences(cls, stops: list[StopInput]) -> list[StopInput]:
        seqs = [s.stop_sequence for s in stops]
        if len(set(seqs)) != len(seqs):
            raise ValueError("stop_sequence değerleri benzersiz olmalıdır.")
        return sorted(stops, key=lambda s: s.stop_sequence)


class OptimizedStopDetail(BaseModel):
    """Optimize edilmiş sıradaki bir durağın detayı."""
    position:           int
    stop_id:            str
    original_sequence:  int
    lat:                float
    lon:                float


class OptimizeResponse(BaseModel):
    """Rota optimizasyon yanıtı."""
    route_id:                   str
    algorithm_used:             str
    original_sequence:          list[int]
    optimized_sequence:         list[int]
    optimized_stop_ids:         list[str]
    original_total_min:         float
    optimized_total_min:        float
    time_saved_min:             float
    improvement_pct:            float
    original_missed_windows:    int
    optimized_missed_windows:   int
    missed_windows_reduced:     int
    original_predicted_delay:   float
    optimized_predicted_delay:  float
    tortuosity_factor:          float
    weather_at_first_stop:      WeatherSnapshot
    stop_details:               list[OptimizedStopDetail]
    elapsed_ms:                 float
    status:                     str


# ──────────────────────────────────────────────────────────────
# GET /health
# ──────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Sistem sağlık durumu."""
    status:           str
    model_loaded:     bool
    model_info:       dict[str, Any]
    weather_api:      str           # "live" | "mock"
    uptime_sec:       float
    version:          str
    environment:      str


# ══════════════════════════════════════════════════════════════
# Uygulama başlangıç / bitiş (Lifespan)
# ══════════════════════════════════════════════════════════════

_APP_START_TIME = time.time()
_SERVICES: dict[str, Any] = {}   # global servis kasası


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Uygulama başlarken servisleri yükler; kapanırken temizler."""
    logger.info("=" * 58)
    logger.info("  Akıllı Lojistik API başlatılıyor...")
    logger.info("=" * 58)

    cfg = get_config()
    _SERVICES["config"] = cfg

    # DelayPredictor — model yoksa fallback çalışır
    try:
        from src.ml.predictor import DelayPredictor
        _SERVICES["predictor"] = DelayPredictor.get_instance(cfg)
        logger.info("DelayPredictor yüklendi.")
    except Exception as exc:
        logger.warning("DelayPredictor yüklenemedi (%s). Fallback aktif.", exc)
        _SERVICES["predictor"] = None

    # RouteOptimizer
    try:
        from src.optimization.route_engine import RouteOptimizer
        _SERVICES["optimizer"] = RouteOptimizer(cfg)
        logger.info("RouteOptimizer hazır.")
    except Exception as exc:
        logger.error("RouteOptimizer yüklenemedi: %s", exc)
        _SERVICES["optimizer"] = None

    # LiveWeatherClient
    try:
        from src.data_ingestion.api_client import LiveWeatherClient
        _SERVICES["weather"] = LiveWeatherClient(cfg)
        logger.info("LiveWeatherClient hazır.")
    except Exception as exc:
        logger.error("LiveWeatherClient yüklenemedi: %s", exc)
        _SERVICES["weather"] = None

    logger.info("API başlatma tamamlandı. http://localhost:8000/docs")
    yield

    logger.info("API kapatılıyor...")
    _SERVICES.clear()


# ══════════════════════════════════════════════════════════════
# FastAPI uygulaması
# ══════════════════════════════════════════════════════════════

def _create_app() -> FastAPI:
    cfg = get_config()
    api_cfg = getattr(cfg, "api", None)

    app = FastAPI(
        title=getattr(api_cfg, "title", "Akıllı Lojistik API"),
        description=(
            "**Akıllı Lojistik ve Rota Optimizasyonu** servisi.\n\n"
            "### Özellikler\n"
            "- 🌦 Anlık hava durumu entegrasyonu (OpenWeatherMap)\n"
            "- 🤖 XGBoost tabanlı gecikme tahmini\n"
            "- 🗺 Dinamik rota optimizasyonu (Exact Permutation + 2-opt)\n"
            "- 🛡 Sunum zırhı: API hataları Mock veri ile karşılanır\n\n"
            "Tüm endpoint'leri aşağıdan test edebilirsiniz."
        ),
        version=getattr(api_cfg, "version", "1.0.0"),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS
    cors_origins = getattr(api_cfg, "cors_origins", ["*"])
    if isinstance(cors_origins, list):
        pass
    else:
        cors_origins = [cors_origins]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


app = _create_app()


# ──────────────────────────────────────────────────────────────
# Dependency: servis erişimi
# ──────────────────────────────────────────────────────────────

def _get_weather_client():
    client = _SERVICES.get("weather")
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LiveWeatherClient başlatılamadı.",
        )
    return client


def _get_predictor():
    predictor = _SERVICES.get("predictor")
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="DelayPredictor yüklenemedi. Önce modeli eğit.",
        )
    return predictor


def _get_optimizer():
    optimizer = _SERVICES.get("optimizer")
    if optimizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RouteOptimizer başlatılamadı.",
        )
    return optimizer


# ══════════════════════════════════════════════════════════════
# Endpoint'ler
# ══════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────
# GET /health
# ──────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Sistem"],
    summary="Sistem sağlık kontrolü",
    description=(
        "Uygulamanın, ML modelinin ve hava durumu servisinin "
        "çalışır durumda olup olmadığını döner. "
        "Sunum öncesi kontrol için kullanılır."
    ),
)
async def health_check():
    try:
        cfg = _SERVICES.get("config") or get_config()

        # Model bilgisi
        predictor  = _SERVICES.get("predictor")
        model_info = predictor.info() if predictor else {"status": "not_loaded"}
        model_ok   = model_info.get("status") == "loaded"

        # Hava durumu servisi — _api_key yoksa "mock"
        weather_client = _SERVICES.get("weather")
        weather_status = "unavailable"
        if weather_client:
            try:
                weather_status = "live" if weather_client._api_key else "mock"
            except Exception:
                weather_status = "mock"

        return HealthResponse(
            status="healthy" if model_ok else "degraded",
            model_loaded=model_ok,
            model_info=model_info,
            weather_api=weather_status,
            uptime_sec=round(time.time() - _APP_START_TIME, 1),
            version=getattr(getattr(cfg, "project", None), "version", "1.0.0"),
            environment=getattr(getattr(cfg, "project", None), "environment", "?"),
        )
    except Exception as exc:
        logger.error("/health endpoint hatası: %s", exc, exc_info=True)
        return HealthResponse(
            status="error",
            model_loaded=False,
            model_info={"status": "error", "message": str(exc)},
            weather_api="unknown",
            uptime_sec=round(time.time() - _APP_START_TIME, 1),
            version="1.0.0",
            environment="unknown",
        )


# ──────────────────────────────────────────────────────────────
# POST /predict
# ──────────────────────────────────────────────────────────────

@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Tahmin"],
    summary="Durak gecikme tahmini",
    description=(
        "Bir durağın koordinatlarını ve özelliklerini alır; "
        "anlık hava durumunu otomatik çeker ve "
        "**XGBoost modeli** ile tahmini gecikme süresini döner.\n\n"
        "**Not:** `planned_arrival` ISO 8601 formatında olmalı "
        "(örn. `2025-06-15T08:30:00`)."
    ),
    response_description="Tahmini gecikme süresi (dakika) ve hava durumu bilgisi.",
)
async def predict_delay(
    request: PredictRequest,
    predictor=Depends(_get_predictor),
    weather_client=Depends(_get_weather_client),
):
    t0 = time.perf_counter()
    try:
        # 1. Anlık hava durumunu çek
        weather_raw = weather_client.get_weather(
            lat=request.latitude,
            lon=request.longitude,
        )

        # 2. Feature satırı oluştur
        arrival = pd.Timestamp(request.planned_arrival)
        feature_row = {
            # Zaman
            "hour_of_day":          arrival.hour,
            "day_of_week":          arrival.dayofweek,
            "month":                arrival.month,
            "is_weekend":           int(arrival.dayofweek >= 5),
            "is_rush_hour":         int(arrival.hour in range(6, 10) or
                                        arrival.hour in range(16, 20)),
            "time_of_day_category": _classify_time(arrival.hour),
            # Rota
            "road_type":             request.road_type,
            "vehicle_type":          request.vehicle_type,
            "planned_travel_min":    request.planned_travel_min,
            "planned_service_min":   request.planned_service_min,
            "distance_from_prev_km": request.distance_from_prev_km,
            "delay_probability":     request.delay_probability,
            "package_count":         request.package_count,
            "package_weight_kg":     request.package_weight_kg,
            "weight_per_package":    (
                request.package_weight_kg / request.package_count
                if request.package_count and request.package_count > 0
                else 0.0
            ),
            # Hava durumu (anlık)
            "w_temperature_c":          weather_raw["temperature_c"],
            "w_precipitation_mm":       weather_raw["precipitation_mm"],
            "w_wind_speed_kmh":         weather_raw["wind_speed_kmh"],
            "w_visibility_km":          weather_raw["visibility_km"],
            "w_humidity_pct":           weather_raw["humidity_pct"],
            "w_delay_risk_score":       weather_raw["delay_risk_score"],
            "w_road_surface_condition": weather_raw["road_surface_condition"],
            "w_weather_condition":      weather_raw["weather_condition"],
            "weather_severity_index":   weather_raw["weather_severity_index"],
            "effective_visibility":     weather_raw["visibility_km"],
        }

        # 3. Tahmin — event loop'u bloke etmemek için thread pool'da çalıştır
        pred_result = await asyncio.to_thread(predictor.predict, feature_row)
        predicted_delay = float(pred_result.get("predicted_delay_min") or 0.0)
        if pred_result.get("status") == "error":
            logger.warning(
                "Model tahmin hatası (fallback 0.0 kullanılıyor): %s",
                pred_result.get("message"),
            )

        # 4. Güven aralığı — hafif matematiksel tahmin (bootstrap yok → timeout yok)
        ci: Optional[dict[str, float]] = None
        try:
            std_est = predicted_delay * 0.18 + 1.5   # heuristic std tahmini
            ci = {
                "lower_90": round(max(0.0, predicted_delay - 1.645 * std_est), 2),
                "upper_90": round(predicted_delay + 1.645 * std_est, 2),
                "std_dev":  round(std_est, 2),
            }
        except Exception as ci_exc:
            logger.debug("CI hesaplanamadı (atlanıyor): %s", ci_exc)

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            "POST /predict | lat=%.4f lon=%.4f | delay=%.2f dk | %.0fms",
            request.latitude, request.longitude, predicted_delay, elapsed_ms,
        )

        return PredictResponse(
            predicted_delay_min=predicted_delay,
            confidence_interval=ci,
            weather=WeatherSnapshot(**{k: weather_raw[k] for k in WeatherSnapshot.model_fields}),
            feature_summary={
                "time_of_day":  str(feature_row["time_of_day_category"]),
                "weather_risk": float(feature_row["weather_severity_index"]),
                "is_rush_hour": bool(feature_row["is_rush_hour"]),
                "road_type":    str(feature_row["road_type"] or ""),
            },
            status="ok",
            elapsed_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("POST /predict beklenmeyen hata: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sunucu hatası: {str(exc)}",
        )


# ──────────────────────────────────────────────────────────────
# POST /optimize
# ──────────────────────────────────────────────────────────────

@app.post(
    "/optimize",
    response_model=OptimizeResponse,
    tags=["Optimizasyon"],
    summary="Rota optimizasyonu",
    description=(
        "Bir rotadaki durakların listesini alır, anlık hava durumunu çeker, "
        "**RouteEngine** ile en iyi durak sıralamasını bulur ve "
        "kazanılan süreyi döner.\n\n"
        "### Algoritma seçimi\n"
        "- **N ≤ 10 mobil durak** → Tam permutasyon (garantili optimal)\n"
        "- **N > 10 mobil durak** → 2-opt yerel arama (hızlı yaklaşık)\n\n"
        "### Kısıtlar\n"
        "- Varsayılan: **tüm duraklar dinamik** (hiçbiri sabit değil)\n"
        "- `fixed_positions` ile opsiyonel pozisyon kilitleme yapılabilir\n"
        "- Zaman penceresi ihlalleri maliyet fonksiyonuna ceza olarak eklenir\n"
        "- ML modeli her permutasyon için gecikme tahmini yapar"
    ),
    response_description="Optimize edilmiş sıralama ve süre kazancı.",
)
async def optimize_route(
    request: OptimizeRequest,
    optimizer=Depends(_get_optimizer),
    weather_client=Depends(_get_weather_client),
):
    t0 = time.perf_counter()
    try:
        # 1. İlk durağın koordinatları için hava durumu çek
        first_stop   = request.stops[0]
        weather_raw  = weather_client.get_weather(
            lat=first_stop.latitude,
            lon=first_stop.longitude,
        )

        # 2. StopInput listesini DataFrame'e çevir
        stops_df = _stops_to_dataframe(request.stops, weather_raw)

        # 3. Hareket zamanı
        departure = (
            pd.Timestamp(request.departure_time)
            if request.departure_time
            else pd.Timestamp.now()
        )

        # 4. Optimizasyon — ağır hesaplamayı thread pool'a taşı
        fixed_pos = (
            frozenset(request.fixed_positions)
            if request.fixed_positions
            else frozenset()
        )
        result = await asyncio.to_thread(
            optimizer.optimize_route,
            request.route_id,
            stops_df,
            departure,
            fixed_pos,
        )

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        time_saved = round(
            result.original_total_min - result.optimized_total_min, 2
        )

        logger.info(
            "POST /optimize | route=%s | kazanılan=%.1f dk | TW iyileşme=%d→%d | %.0fms",
            request.route_id,
            time_saved,
            result.original_missed_windows,
            result.optimized_missed_windows,
            elapsed_ms,
        )

        return OptimizeResponse(
            route_id=result.route_id,
            algorithm_used=result.algorithm_used,
            original_sequence=result.original_sequence,
            optimized_sequence=result.optimized_sequence,
            optimized_stop_ids=result.optimized_stop_ids,
            original_total_min=result.original_total_min,
            optimized_total_min=result.optimized_total_min,
            time_saved_min=time_saved,
            improvement_pct=result.improvement_pct,
            original_missed_windows=result.original_missed_windows,
            optimized_missed_windows=result.optimized_missed_windows,
            missed_windows_reduced=max(
                0, result.original_missed_windows - result.optimized_missed_windows
            ),
            original_predicted_delay=result.original_predicted_delay,
            optimized_predicted_delay=result.optimized_predicted_delay,
            tortuosity_factor=result.tortuosity_factor,
            weather_at_first_stop=WeatherSnapshot(
                **{k: weather_raw[k] for k in WeatherSnapshot.model_fields}
            ),
            stop_details=[
                OptimizedStopDetail(
                    position=d["position"],
                    stop_id=d["stop_id"],
                    original_sequence=d["sequence_original"],
                    lat=d["lat"],
                    lon=d["lon"],
                )
                for d in result.stop_details
            ],
            elapsed_ms=elapsed_ms,
            status="ok",
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("POST /optimize beklenmeyen hata: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimizasyon hatası: {str(exc)}",
        )


# ══════════════════════════════════════════════════════════════
# Hata yakalama
# ══════════════════════════════════════════════════════════════

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Global exception [%s %s]: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "İç sunucu hatası. Lütfen logları kontrol edin.",
                 "error": str(exc)},
    )


# ══════════════════════════════════════════════════════════════
# Yardımcı fonksiyonlar
# ══════════════════════════════════════════════════════════════

def _classify_time(hour: int) -> str:
    """Saat → gün bölümü kategorisi."""
    if 0 <= hour < 6:   return "night"
    if 6 <= hour < 9:   return "morning_rush"
    if 9 <= hour < 16:  return "midday"
    if 16 <= hour < 19: return "afternoon_rush"
    if 19 <= hour < 22: return "evening"
    return "late_night"


def _stops_to_dataframe(
    stops: list[StopInput],
    weather: dict[str, Any],
) -> pd.DataFrame:
    """
    Pydantic StopInput listesini, RouteOptimizer'ın beklediği
    DataFrame'e dönüştürür. Hava durumu feature'larını da ekler.
    """
    rows = []
    for stop in stops:
        arrival = pd.Timestamp(stop.planned_arrival) if stop.planned_arrival else pd.Timestamp.now()
        row: dict[str, Any] = {
            # Durak kimliği
            "stop_id":               stop.stop_id,
            "stop_sequence":         stop.stop_sequence,
            "latitude":              stop.latitude,
            "longitude":             stop.longitude,
            # Zaman pencereleri
            "planned_arrival":       arrival,
            "time_window_open":      (
                pd.Timestamp(stop.time_window_open) if stop.time_window_open else None
            ),
            "time_window_close":     (
                pd.Timestamp(stop.time_window_close) if stop.time_window_close else None
            ),
            # Durak özellikleri
            "planned_service_min":   stop.planned_service_min,
            "planned_travel_min":    stop.planned_travel_min,
            "distance_from_prev_km": stop.distance_from_prev_km,
            "delay_probability":     stop.delay_probability,
            "road_type":             stop.road_type,
            "package_count":         stop.package_count,
            "package_weight_kg":     stop.package_weight_kg,
            # Zaman öznitelikleri
            "hour_of_day":           arrival.hour,
            "day_of_week":           arrival.dayofweek,
            "month":                 arrival.month,
            "is_weekend":            int(arrival.dayofweek >= 5),
            "is_rush_hour":          int(arrival.hour in range(6, 10) or
                                         arrival.hour in range(16, 20)),
            "time_of_day_category":  _classify_time(arrival.hour),
            # Hava durumu (anlık)
            "w_temperature_c":          weather["temperature_c"],
            "w_precipitation_mm":       weather["precipitation_mm"],
            "w_wind_speed_kmh":         weather["wind_speed_kmh"],
            "w_visibility_km":          weather["visibility_km"],
            "w_humidity_pct":           weather["humidity_pct"],
            "w_delay_risk_score":       weather["delay_risk_score"],
            "w_weather_condition":      weather["weather_condition"],
            "w_road_surface_condition": weather["road_surface_condition"],
            "weather_severity_index":   weather["weather_severity_index"],
            "effective_visibility":     weather["visibility_km"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df["weight_per_package"] = (
        df["package_weight_kg"] / df["package_count"].replace(0, float("nan"))
    ).fillna(0.0)
    return df


# ══════════════════════════════════════════════════════════════
# Doğrudan çalıştırma
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    cfg     = get_config()
    api_cfg = getattr(cfg, "api", None)

    uvicorn.run(
        "src.api.fastapi_app:app",
        host=getattr(api_cfg, "host", "0.0.0.0"),
        port=int(getattr(api_cfg, "port", 8000)),
        reload=bool(getattr(api_cfg, "reload", True)),
        workers=int(getattr(api_cfg, "workers", 1)),
        log_level="info",
    )
