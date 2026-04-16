"""
src/optimization/route_engine.py

Akıllı Lojistik — Rota Optimizasyon Motoru

5 Gelişmiş Mühendislik Kuralı:
  1. Anchor Point       — 1. durak sabit; yalnızca 2..N sırası optimize edilir.
  2. Tortuosity Factor  — Rotaya özel dinamik kıvrım katsayısı (gerçek/haversine).
  3. Distance Matrix    — Permutasyon öncesi önhesaplanmış mesafe matrisi (O(1) lookup).
  4. Adapter Pattern    — get_route_distance() soyutlaması (OSRM/GMaps için hazır).
  5. ML Entegrasyonu    — DelayPredictor ile her permutasyon için gecikme tahmini;
                          hedef: toplam süre minimizasyonu + missed_time_window → 0.

Kullanım:
    from config.config_loader import get_config
    from src.optimization.route_engine import RouteOptimizer

    optimizer = RouteOptimizer(get_config())
    result    = optimizer.optimize_route("RT-0001", stops_df)

    print(result.summary())
"""

import itertools
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from config.config_loader import Config, get_config

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Sabitler
# ──────────────────────────────────────────────────────────────
EARTH_RADIUS_KM       = 6371.0
MAX_EXACT_PERM_STOPS  = 10          # Bu sayıya kadar tam permutasyon; üstü 2-opt
MISSED_WINDOW_PENALTY = 30.0        # Kaçırılan zaman penceresi başına ceza (dakika)
ASSUMED_AVG_SPEED_KMH = 50.0        # Mesafe → süre dönüşüm için varsayılan hız


# ══════════════════════════════════════════════════════════════
# Veri yapıları
# ══════════════════════════════════════════════════════════════

@dataclass
class StopPoint:
    """Tek bir durağı temsil eden hafif veri nesnesi."""
    idx: int                          # Orijinal DataFrame index
    stop_id: str
    sequence: int
    lat: float
    lon: float
    time_window_open: Optional[pd.Timestamp]
    time_window_close: Optional[pd.Timestamp]
    planned_service_min: float
    planned_travel_min: float
    delay_probability: float
    distance_from_prev_km: float      # Orijinal CSV değeri
    features: dict = field(default_factory=dict)  # ML feature sözlüğü


@dataclass
class OptimizationResult:
    """Optimizasyon çıktısını tutan veri nesnesi."""
    route_id: str
    original_sequence: list[int]       # Orijinal stop_sequence listesi
    optimized_sequence: list[int]      # Optimize edilmiş stop_sequence listesi
    optimized_stop_ids: list[str]
    original_total_min: float
    optimized_total_min: float
    original_missed_windows: int
    optimized_missed_windows: int
    original_predicted_delay: float
    optimized_predicted_delay: float
    tortuosity_factor: float
    algorithm_used: str
    elapsed_sec: float
    improvement_pct: float
    stop_details: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        sign  = "-" if self.improvement_pct >= 0 else "+"
        delta = abs(self.improvement_pct)
        return (
            f"\n{'═' * 58}\n"
            f"  Rota: {self.route_id}  |  Algoritma: {self.algorithm_used}\n"
            f"{'─' * 58}\n"
            f"  Tortuosity Faktörü     : {self.tortuosity_factor:.4f}\n"
            f"  Orijinal  süre         : {self.original_total_min:.1f} dk\n"
            f"  Optimize  süre         : {self.optimized_total_min:.1f} dk  "
            f"({sign}{delta:.1f}% iyileşme)\n"
            f"  Orijinal  gecikme      : {self.original_predicted_delay:.1f} dk\n"
            f"  Optimize  gecikme      : {self.optimized_predicted_delay:.1f} dk\n"
            f"  Orijinal  kaçırılan TW : {self.original_missed_windows}\n"
            f"  Optimize  kaçırılan TW : {self.optimized_missed_windows}\n"
            f"  Hesaplama süresi       : {self.elapsed_sec:.3f} sn\n"
            f"{'═' * 58}"
        )


# ══════════════════════════════════════════════════════════════
# 1. Modül: TortuosityCalculator
# ══════════════════════════════════════════════════════════════

class TortuosityCalculator:
    """
    Rotaya özel dinamik Kıvrım Katsayısı (Tortuosity Factor) hesaplar.

    Formül:
        TF = mean( distance_from_prev_km[i] / haversine(stop[i-1], stop[i]) )
        (sadece haversine > 0.1 km olan durak çiftleri için)

    TF ≈ 1.0 → düz yol;  TF > 1.5 → kıvrımlı/dağlık yol.
    Fallback: TF = 1.3 (kentsel ortalama, önceki çalışmalardan)
    """
    DEFAULT_TORTUOSITY = 1.3

    def compute(self, stops: list[StopPoint]) -> float:
        """
        Args:
            stops: Sıralı StopPoint listesi (stop_sequence'a göre).

        Returns:
            Rotaya özel tortuosity factor (float).
        """
        try:
            ratios: list[float] = []
            for i in range(1, len(stops)):
                real_dist = stops[i].distance_from_prev_km
                hav_dist  = _haversine_km(
                    stops[i - 1].lat, stops[i - 1].lon,
                    stops[i].lat,     stops[i].lon,
                )
                if hav_dist > 0.1 and real_dist > 0.0:
                    ratios.append(real_dist / hav_dist)

            if not ratios:
                logger.warning(
                    "Tortuosity hesaplanamadı (yetersiz veri). "
                    "Fallback: %.2f", self.DEFAULT_TORTUOSITY
                )
                return self.DEFAULT_TORTUOSITY

            tf = float(np.mean(ratios))
            # Fiziksel anlam kontrolü: 0.9 – 4.0 aralığında olmalı
            tf = float(np.clip(tf, 0.9, 4.0))
            logger.debug("Tortuosity Factor hesaplandı: %.4f (%d çift)", tf, len(ratios))
            return tf

        except Exception as exc:
            logger.error("TortuosityCalculator.compute() hatası: %s", exc)
            return self.DEFAULT_TORTUOSITY


# ══════════════════════════════════════════════════════════════
# 2. Modül: RouteDistanceAdapter  (Adapter Pattern)
# ══════════════════════════════════════════════════════════════

class RouteDistanceAdapter:
    """
    Mesafe hesaplama adaptörü.

    Mevcut implementasyon: Haversine × Tortuosity Factor.

    ┌─────────────────────────────────────────────────────────┐
    │  GELECEK GENİŞLEME NOKTASI (Adapter Pattern)           │
    │  get_route_distance() içine aşağıdaki alternatifler    │
    │  eklenebilir:                                           │
    │    • OSRM   : http://router.project-osrm.org/route/v1/ │
    │    • Google : maps.googleapis.com/maps/api/distancematrix │
    │    • Valhalla, GraphHopper, HERE API                    │
    │  Değişiklik sadece bu sınıfı etkiler; üst katmanlar    │
    │  hiç değişmez.                                          │
    └─────────────────────────────────────────────────────────┘
    """

    def __init__(self, tortuosity_factor: float = 1.3):
        self.tortuosity_factor = tortuosity_factor

    def get_route_distance(
        self,
        point_a: StopPoint,
        point_b: StopPoint,
    ) -> float:
        """
        İki durak arasındaki tahmini yol mesafesini km olarak döner.

        Şu an:  Haversine(a, b) × tortuosity_factor
        İleride: OSRM / Google Maps API isteği buraya eklenecek.

        Args:
            point_a: Başlangıç durağı.
            point_b: Bitiş durağı.

        Returns:
            Tahmini yol mesafesi (km).
        """
        try:
            # ── Mevcut implementasyon ──────────────────────────────
            hav = _haversine_km(point_a.lat, point_a.lon,
                                point_b.lat, point_b.lon)
            road_dist = hav * self.tortuosity_factor
            return round(road_dist, 4)

            # ── GELECEK: OSRM API ─────────────────────────────────
            # import requests
            # url = (
            #     f"http://router.project-osrm.org/route/v1/driving/"
            #     f"{point_a.lon},{point_a.lat};{point_b.lon},{point_b.lat}"
            #     f"?overview=false"
            # )
            # resp = requests.get(url, timeout=5).json()
            # return resp["routes"][0]["distance"] / 1000.0  # metre → km

            # ── GELECEK: Google Maps Distance Matrix API ──────────
            # import googlemaps
            # gmaps  = googlemaps.Client(key=os.environ["GOOGLE_MAPS_KEY"])
            # result = gmaps.distance_matrix(
            #     origins=[(point_a.lat, point_a.lon)],
            #     destinations=[(point_b.lat, point_b.lon)],
            #     mode="driving",
            # )
            # return result["rows"][0]["elements"][0]["distance"]["value"] / 1000.0

        except Exception as exc:
            logger.warning(
                "get_route_distance hatası (%s→%s): %s. Haversine kullanılıyor.",
                point_a.stop_id, point_b.stop_id, exc,
            )
            return _haversine_km(point_a.lat, point_a.lon,
                                 point_b.lat, point_b.lon) * self.tortuosity_factor

    def distance_to_minutes(self, km: float, speed_kmh: float = ASSUMED_AVG_SPEED_KMH) -> float:
        """Mesafeyi (km) tahmini seyahat süresine (dakika) çevirir."""
        if speed_kmh <= 0:
            speed_kmh = ASSUMED_AVG_SPEED_KMH
        return round((km / speed_kmh) * 60.0, 2)


# ══════════════════════════════════════════════════════════════
# 3. Modül: DistanceMatrix
# ══════════════════════════════════════════════════════════════

class DistanceMatrix:
    """
    Tüm duraklar arası yol mesafelerini (km) ve seyahat sürelerini
    (dakika) optimizasyon başlamadan önce hesaplar ve önbelleğe alır.

    Permutasyon arama sırasında O(1) ile lookup sağlar.
    N durak → N² hesaplama (tek sefer), sonrasında matris indexleme.
    """

    def __init__(
        self,
        stops: list[StopPoint],
        adapter: RouteDistanceAdapter,
    ):
        self.stops   = stops
        self.adapter = adapter
        self.n       = len(stops)
        self._dist_km:  np.ndarray  # [n x n] yol mesafesi (km)
        self._dist_min: np.ndarray  # [n x n] seyahat süresi (dakika)
        self._id_to_pos: dict[str, int] = {}  # stop_id → matris pozisyonu
        self._build()

    def _build(self) -> None:
        """Tüm durak çiftleri için mesafe ve süre matrisini oluşturur."""
        try:
            t0 = time.perf_counter()
            self._dist_km  = np.zeros((self.n, self.n), dtype=np.float32)
            self._dist_min = np.zeros((self.n, self.n), dtype=np.float32)

            for i, s_a in enumerate(self.stops):
                self._id_to_pos[s_a.stop_id] = i
                for j, s_b in enumerate(self.stops):
                    if i == j:
                        continue
                    km = self.adapter.get_route_distance(s_a, s_b)
                    self._dist_km[i, j]  = km
                    self._dist_min[i, j] = self.adapter.distance_to_minutes(km)

            elapsed = time.perf_counter() - t0
            logger.info(
                "Distance Matrix oluşturuldu: %d×%d (%d hesaplama, %.3f sn)",
                self.n, self.n, self.n * (self.n - 1), elapsed,
            )
        except Exception as exc:
            logger.error("DistanceMatrix._build() hatası: %s", exc, exc_info=True)
            raise

    def km(self, i: int, j: int) -> float:
        """stop index i'den j'ye yol mesafesi (km)."""
        return float(self._dist_km[i, j])

    def minutes(self, i: int, j: int) -> float:
        """stop index i'den j'ye seyahat süresi (dakika)."""
        return float(self._dist_min[i, j])

    def route_total_minutes(self, sequence: list[int]) -> float:
        """
        Verilen stop index sırası için toplam seyahat süresini hesaplar.
        Servis sürelerini dahil etmez (ayrıca eklenir).
        """
        total = 0.0
        for k in range(len(sequence) - 1):
            total += self.minutes(sequence[k], sequence[k + 1])
        return round(total, 2)


# ══════════════════════════════════════════════════════════════
# 4. Modül: ObjectiveFunction
# ══════════════════════════════════════════════════════════════

class ObjectiveFunction:
    """
    Bir permutasyonun "maliyet" skorunu hesaplar.

    Maliyet = toplam_seyahat_süresi
            + toplam_servis_süresi
            + tahmin_edilen_gecikme         (ML)
            + missed_windows × penalty

    Hedef: minimize et.
    """

    def __init__(
        self,
        matrix: DistanceMatrix,
        stops: list[StopPoint],
        predictor: Any,                   # DelayPredictor (import döngüsü olmasın)
        departure_time: pd.Timestamp,
    ):
        self.matrix         = matrix
        self.stops          = stops
        self.predictor      = predictor
        self.departure_time = departure_time
        self._cache: dict[tuple, float] = {}   # permutation tuple → maliyet

    def evaluate(self, sequence: list[int]) -> dict[str, float]:
        """
        Verilen matris index sırası için tam maliyeti döner.

        Args:
            sequence: matris indexlerinin listesi (anchor dahil).

        Returns:
            {"cost": float, "travel_min": float, "delay_min": float,
             "missed_windows": int, "total_min": float}
        """
        seq_key = tuple(sequence)
        if seq_key in self._cache:
            return {"cost": self._cache[seq_key]}

        try:
            travel_min   = self.matrix.route_total_minutes(sequence)
            service_min  = sum(self.stops[i].planned_service_min for i in sequence)
            delay_min, missed = self._estimate_delays(sequence, travel_min)

            cost = (
                travel_min
                + service_min
                + delay_min
                + missed * MISSED_WINDOW_PENALTY
            )
            self._cache[seq_key] = cost

            return {
                "cost":          round(cost, 2),
                "travel_min":    round(travel_min, 2),
                "service_min":   round(service_min, 2),
                "delay_min":     round(delay_min, 2),
                "missed_windows": missed,
                "total_min":     round(travel_min + service_min, 2),
            }

        except Exception as exc:
            logger.warning("ObjectiveFunction.evaluate() hatası: %s", exc)
            return {"cost": float("inf"), "travel_min": 0, "service_min": 0,
                    "delay_min": 0, "missed_windows": 999, "total_min": 0}

    def _estimate_delays(
        self, sequence: list[int], total_travel_min: float
    ) -> tuple[float, int]:
        """
        ML modeli (DelayPredictor) ile her durağın gecikme tahminini yapar
        ve kaçırılan zaman penceresi sayısını hesaplar.

        Returns:
            (toplam_gecikme_dk, missed_time_window_sayısı)
        """
        total_delay   = 0.0
        missed        = 0
        current_time  = self.departure_time
        cum_delay     = 0.0

        for pos, mat_idx in enumerate(sequence):
            stop = self.stops[mat_idx]

            # Bir önceki durağa olan seyahat süresi
            travel = (
                self.matrix.minutes(sequence[pos - 1], mat_idx)
                if pos > 0 else 0.0
            )
            current_time = current_time + pd.Timedelta(minutes=travel)

            # ML feature'larını hazırla
            feature_row = self._build_feature_row(stop, pos, travel, cum_delay)

            # DelayPredictor tahmini
            try:
                pred = self.predictor.predict(feature_row)
                stop_delay = pred.get("predicted_delay_min") or 0.0
                stop_delay = max(0.0, float(stop_delay))
            except Exception:
                stop_delay = stop.delay_probability * 15.0   # fallback: prob × 15 dk

            total_delay += stop_delay
            cum_delay   += stop_delay
            current_time = current_time + pd.Timedelta(minutes=stop.planned_service_min + stop_delay)

            # Zaman penceresi kontrolü
            if stop.time_window_close is not None:
                if current_time > stop.time_window_close:
                    missed += 1
                    logger.debug(
                        "  TW kaçırıldı: %s (varış=%s, pencere=%s)",
                        stop.stop_id,
                        current_time.strftime("%H:%M"),
                        stop.time_window_close.strftime("%H:%M"),
                    )

        return round(total_delay, 2), missed

    def _build_feature_row(
        self,
        stop: StopPoint,
        position: int,
        travel_min: float,
        cum_delay: float,
    ) -> dict:
        """ML için feature satırı oluşturur."""
        base = dict(stop.features)   # preprocessor'dan gelen öznitelikler
        base.update({
            "stop_sequence":        position + 1,
            "planned_travel_min":   travel_min,
            "cumulative_delay_at_stop": cum_delay,
            "delay_probability":    stop.delay_probability,
        })
        return base


# ══════════════════════════════════════════════════════════════
# 5. Ana Modül: RouteOptimizer
# ══════════════════════════════════════════════════════════════

class RouteOptimizer:
    """
    Rota Optimizasyon Motoru.

    Algoritma seçimi:
      • N_mobil ≤ MAX_EXACT_PERM_STOPS → Tam permutasyon (garanti optimal)
      • N_mobil  > MAX_EXACT_PERM_STOPS → 2-opt yerel arama (hızlı yaklaşık)

    (N_mobil = toplam durak sayısı - 1 anchor)
    """

    def __init__(self, config: Optional[Config] = None):
        self.cfg = config or get_config()
        self._predictor = None          # lazy load (döngüsel import önlemi)
        self._tortuosity_calc = TortuosityCalculator()
        logger.info("RouteOptimizer başlatıldı.")

    # ──────────────────────────────────────────────────────────
    # Genel giriş noktası
    # ──────────────────────────────────────────────────────────
    def optimize_route(
        self,
        route_id: str,
        stops_df: pd.DataFrame,
        departure_time: Optional[pd.Timestamp] = None,
    ) -> OptimizationResult:
        """
        Tek bir rotayı optimize eder.

        Args:
            route_id      : Rota kimliği (loglama için).
            stops_df      : Bu rotaya ait stop satırları (preprocessed).
            departure_time: Hareket zamanı; None → şimdiki zaman.

        Returns:
            OptimizationResult
        """
        t_start = time.perf_counter()
        try:
            logger.info("─" * 58)
            logger.info("Optimizasyon başlıyor: %s (%d durak)", route_id, len(stops_df))

            # 1. Veriyi doğrula ve parse et
            stops_df = self._validate_and_parse(stops_df)
            if len(stops_df) < 2:
                raise ValueError(f"{route_id}: En az 2 durak gerekli.")

            # 2. StopPoint nesneleri oluştur
            stop_points = self._build_stop_points(stops_df)

            # 3. Tortuosity Factor (Kural 2)
            tf = self._tortuosity_calc.compute(stop_points)
            logger.info("Tortuosity Factor: %.4f", tf)

            # 4. Adapter + Distance Matrix (Kural 3 + 4)
            adapter = RouteDistanceAdapter(tortuosity_factor=tf)
            matrix  = DistanceMatrix(stop_points, adapter)

            # 5. Anchor Point — 1. durak sabit (Kural 1)
            anchor_idx   = 0                          # matris pozisyonu
            mobile_idxs  = list(range(1, len(stop_points)))  # optimize edilecek

            # 6. Hareket zamanı
            dep_time = departure_time or pd.Timestamp.now()

            # 7. Predictor (lazy load) (Kural 5)
            predictor = self._get_predictor()

            # 8. Objective function
            obj = ObjectiveFunction(matrix, stop_points, predictor, dep_time)

            # 9. Orijinal sıranın maliyetini hesapla
            original_seq  = [anchor_idx] + mobile_idxs
            original_eval = obj.evaluate(original_seq)

            # 10. Optimizasyon algoritmasını seç ve çalıştır
            if len(mobile_idxs) <= MAX_EXACT_PERM_STOPS:
                algorithm     = "exact_permutation"
                best_seq, best_eval = self._exact_permutation(
                    anchor_idx, mobile_idxs, obj
                )
            else:
                algorithm     = "two_opt"
                best_seq, best_eval = self._two_opt(original_seq, obj)

            # 11. Sonuç oluştur
            elapsed = time.perf_counter() - t_start
            result  = self._build_result(
                route_id, stop_points, original_seq, best_seq,
                original_eval, best_eval, tf, algorithm, elapsed,
            )

            logger.info(result.summary())
            return result

        except Exception as exc:
            logger.error("optimize_route(%s) hatası: %s", route_id, exc, exc_info=True)
            raise

    def optimize_all_routes(
        self,
        stops_df: pd.DataFrame,
        routes_df: Optional[pd.DataFrame] = None,
    ) -> dict[str, OptimizationResult]:
        """
        DataFrame'deki tüm route_id'ler için toplu optimizasyon.

        Args:
            stops_df : preprocessed stop verisi (tüm rotalar).
            routes_df: routes tablosu (departure zamanı için, opsiyonel).

        Returns:
            {route_id: OptimizationResult}
        """
        results: dict[str, OptimizationResult] = {}

        if "route_id" not in stops_df.columns:
            raise KeyError("stops_df'de 'route_id' kolonu bulunamadı.")

        route_ids = stops_df["route_id"].unique()
        logger.info("Toplu optimizasyon: %d rota", len(route_ids))

        for rid in route_ids:
            route_stops = stops_df[stops_df["route_id"] == rid].copy()

            # Hareket zamanını routes tablosundan al
            dep_time = None
            if routes_df is not None and not routes_df.empty:
                row = routes_df[routes_df["route_id"] == rid]
                if not row.empty and "departure_planned" in row.columns:
                    dep_time = pd.to_datetime(
                        row["departure_planned"].iloc[0], errors="coerce"
                    )
                    if pd.isna(dep_time):
                        dep_time = None

            try:
                results[rid] = self.optimize_route(rid, route_stops, dep_time)
            except Exception as exc:
                logger.error("Rota %s atlandı: %s", rid, exc)
                continue

        success = len(results)
        logger.info(
            "Toplu optimizasyon tamamlandı: %d/%d rota başarılı.",
            success, len(route_ids),
        )
        return results

    # ──────────────────────────────────────────────────────────
    # Algoritma 1: Tam Permutasyon (Kural 1 + 3 + 5)
    # ──────────────────────────────────────────────────────────
    def _exact_permutation(
        self,
        anchor_idx: int,
        mobile_idxs: list[int],
        obj: ObjectiveFunction,
    ) -> tuple[list[int], dict]:
        """
        Tüm permutasyonları dener; en düşük maliyetli sırayı döner.
        Anchor daima başta sabit tutulur (Kural 1).

        N_mobile! permutasyon sayısı → küçük N için garantili optimal.
        """
        best_seq  = [anchor_idx] + mobile_idxs
        best_eval = obj.evaluate(best_seq)

        n_perms   = 0
        logger.info(
            "Tam permutasyon başlıyor: %d! = %d permutasyon",
            len(mobile_idxs),
            _factorial(len(mobile_idxs)),
        )

        for perm in itertools.permutations(mobile_idxs):
            candidate  = [anchor_idx] + list(perm)
            candidate_eval = obj.evaluate(candidate)
            n_perms   += 1

            if candidate_eval["cost"] < best_eval["cost"]:
                best_eval = candidate_eval
                best_seq  = candidate
                logger.debug(
                    "  Yeni en iyi: cost=%.2f, missed_tw=%d (perm #%d)",
                    best_eval["cost"], best_eval["missed_windows"], n_perms,
                )

        logger.info(
            "Tam permutasyon tamamlandı: %d denendi | "
            "en iyi cost=%.2f | missed_tw=%d",
            n_perms, best_eval["cost"], best_eval["missed_windows"],
        )
        return best_seq, best_eval

    # ──────────────────────────────────────────────────────────
    # Algoritma 2: 2-opt Yerel Arama (büyük rotalar için)
    # ──────────────────────────────────────────────────────────
    def _two_opt(
        self,
        initial_seq: list[int],
        obj: ObjectiveFunction,
        max_iterations: int = 500,
    ) -> tuple[list[int], dict]:
        """
        2-opt yerel arama; büyük rotalarda (N > MAX_EXACT_PERM_STOPS)
        makul sürede kaliteli çözüm üretir.
        Anchor (index 0) hiçbir zaman yer değiştirmez.
        """
        best_seq  = initial_seq[:]
        best_eval = obj.evaluate(best_seq)
        improved  = True
        iteration = 0

        logger.info(
            "2-opt yerel arama başlıyor: %d durak, başlangıç cost=%.2f",
            len(best_seq), best_eval["cost"],
        )

        while improved and iteration < max_iterations:
            improved  = False
            iteration += 1

            # Anchor (pos=0) dokunulmaz → range(1, n-1)
            for i in range(1, len(best_seq) - 1):
                for j in range(i + 1, len(best_seq)):
                    candidate     = best_seq[:i] + best_seq[i:j + 1][::-1] + best_seq[j + 1:]
                    candidate_eval = obj.evaluate(candidate)

                    if candidate_eval["cost"] < best_eval["cost"]:
                        best_eval = candidate_eval
                        best_seq  = candidate
                        improved  = True

        logger.info(
            "2-opt tamamlandı: %d iterasyon | "
            "en iyi cost=%.2f | missed_tw=%d",
            iteration, best_eval["cost"], best_eval["missed_windows"],
        )
        return best_seq, best_eval

    # ──────────────────────────────────────────────────────────
    # Veri yardımcıları
    # ──────────────────────────────────────────────────────────
    def _validate_and_parse(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gerekli kolonları kontrol eder ve datetime parse eder."""
        required = ["stop_id", "latitude", "longitude", "stop_sequence"]
        missing  = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"stops_df'de eksik kolonlar: {missing}")

        dt_cols = [
            "planned_arrival", "actual_arrival",
            "time_window_open", "time_window_close",
        ]
        for col in dt_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        df = df.sort_values("stop_sequence").reset_index(drop=True)
        return df

    def _build_stop_points(self, df: pd.DataFrame) -> list[StopPoint]:
        """DataFrame satırlarını StopPoint nesnelerine dönüştürür."""
        stops = []
        for _, row in df.iterrows():
            # ML için feature sözlüğünü hazırla
            feature_cols = [
                "hour_of_day", "day_of_week", "is_rush_hour", "is_weekend",
                "month", "time_of_day_category", "weather_severity_index",
                "effective_visibility", "precip_intensity_cat",
                "speed_ratio", "congestion_delay_factor", "congestion_category",
                "road_type", "vehicle_type", "weather_condition",
                "t_traffic_level", "overall_delay_factor",
                "package_count", "package_weight_kg", "weight_per_package",
                "w_delay_risk_score",
            ]
            features = {
                col: row[col]
                for col in feature_cols
                if col in row.index and pd.notna(row[col])
            }

            stops.append(StopPoint(
                idx=int(row.name),
                stop_id=str(row.get("stop_id", f"STP-{row.name}")),
                sequence=int(row.get("stop_sequence", 0)),
                lat=float(row.get("latitude", 0.0)),
                lon=float(row.get("longitude", 0.0)),
                time_window_open=row.get("time_window_open"),
                time_window_close=row.get("time_window_close"),
                planned_service_min=float(row.get("planned_service_min", 10.0)),
                planned_travel_min=float(row.get("planned_travel_min", 20.0)),
                delay_probability=float(row.get("delay_probability", 0.2)),
                distance_from_prev_km=float(row.get("distance_from_prev_km", 0.0)),
                features=features,
            ))
        return stops

    def _build_result(
        self,
        route_id: str,
        stops: list[StopPoint],
        original_seq: list[int],
        best_seq: list[int],
        orig_eval: dict,
        best_eval: dict,
        tf: float,
        algorithm: str,
        elapsed: float,
    ) -> OptimizationResult:
        """OptimizationResult nesnesini oluşturur."""
        orig_total = orig_eval.get("total_min", 0.0)
        best_total = best_eval.get("total_min", 0.0)

        improvement = (
            ((orig_total - best_total) / orig_total * 100)
            if orig_total > 0 else 0.0
        )

        # Stop detaylarını sıralı olarak hazırla
        stop_details = [
            {
                "position":    pos + 1,
                "stop_id":     stops[mat_idx].stop_id,
                "sequence_original": stops[mat_idx].sequence,
                "lat":         stops[mat_idx].lat,
                "lon":         stops[mat_idx].lon,
            }
            for pos, mat_idx in enumerate(best_seq)
        ]

        return OptimizationResult(
            route_id=route_id,
            original_sequence=[stops[i].sequence for i in original_seq],
            optimized_sequence=[stops[i].sequence for i in best_seq],
            optimized_stop_ids=[stops[i].stop_id for i in best_seq],
            original_total_min=orig_total,
            optimized_total_min=best_total,
            original_missed_windows=orig_eval.get("missed_windows", 0),
            optimized_missed_windows=best_eval.get("missed_windows", 0),
            original_predicted_delay=orig_eval.get("delay_min", 0.0),
            optimized_predicted_delay=best_eval.get("delay_min", 0.0),
            tortuosity_factor=round(tf, 4),
            algorithm_used=algorithm,
            elapsed_sec=round(elapsed, 3),
            improvement_pct=round(improvement, 2),
            stop_details=stop_details,
        )

    def _get_predictor(self):
        """DelayPredictor singleton'ını lazy load eder."""
        if self._predictor is None:
            try:
                from src.ml.predictor import DelayPredictor  # noqa: PLC0415
                self._predictor = DelayPredictor.get_instance(self.cfg)
                logger.info("DelayPredictor ML modeli yüklendi.")
            except Exception as exc:
                logger.warning(
                    "DelayPredictor yüklenemedi (%s). "
                    "Fallback: delay_probability × 15 dk kullanılacak.", exc
                )
                self._predictor = _FallbackPredictor()
        return self._predictor


# ══════════════════════════════════════════════════════════════
# Fallback: ML modeli yokken
# ══════════════════════════════════════════════════════════════

class _FallbackPredictor:
    """
    Model dosyası henüz eğitilmemişse kullanılan basit kural tabanlı tahminci.
    delay_probability × 15 dk heuristic'i uygular.
    """

    def predict(self, data: dict) -> dict:
        prob  = float(data.get("delay_probability", 0.2))
        delay = round(prob * 15.0, 2)
        return {"predicted_delay_min": delay, "status": "fallback"}


# ══════════════════════════════════════════════════════════════
# Yardımcı saf fonksiyonlar
# ══════════════════════════════════════════════════════════════

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """İki koordinat arasındaki kuş uçuşu mesafesini km cinsinden döner."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _factorial(n: int) -> int:
    """n! hesaplar (küçük N için)."""
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


# ══════════════════════════════════════════════════════════════
# Hızlı test
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    from src.data_ingestion.csv_reader import CSVReader
    from src.features.preprocessor import run_preprocessing_pipeline

    cfg      = get_config()
    dfs      = CSVReader(cfg).load_all()
    features = run_preprocessing_pipeline(dfs, cfg)

    # İlk rotayı al
    first_route_id = features["route_id"].iloc[0]
    route_stops    = features[features["route_id"] == first_route_id].copy()
    routes_df      = dfs.get("routes", pd.DataFrame())

    optimizer = RouteOptimizer(cfg)
    result    = optimizer.optimize_route(
        first_route_id,
        route_stops,
        departure_time=pd.Timestamp("2025-02-23 15:43:46"),
    )

    print(result.summary())
    print("\nOptimize edilmiş durak sırası:")
    for d in result.stop_details:
        print(f"  {d['position']}. {d['stop_id']} (orijinal seq={d['sequence_original']})")
