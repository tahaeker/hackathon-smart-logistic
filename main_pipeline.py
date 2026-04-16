"""
main_pipeline.py  —  Eğitim Boru Hattı Orkestratörü
====================================================
Çalıştırma:
    python main_pipeline.py
    python main_pipeline.py --skip-join   (join adımını atla, işlenmiş veri varsa)
    python main_pipeline.py --routes-only (sadece routes verisini kullan)

Adımlar:
  1. Konfigürasyon yükleme
  2. CSV okuma       (CSVReader)
  3. Ön işleme       (SpatialTemporalJoiner + FeatureEngineer)
  4. Model eğitimi   (DelayModelTrainer)
  5. Özet raporu
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# ── Proje kök dizinini Python path'ine ekle ──────────────────
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Konfigürasyonu EN ERKEN yükle (logger kurulumu için) ────
from config.config_loader import get_config  # noqa: E402

cfg = get_config()

# ── Şimdi diğer modülleri import et ─────────────────────────
import pandas as pd  # noqa: E402

from src.data_ingestion.csv_reader import CSVReader          # noqa: E402
from src.features.preprocessor import run_preprocessing_pipeline  # noqa: E402
from src.ml.trainer import DelayModelTrainer                 # noqa: E402

logger = logging.getLogger("main_pipeline")

# ═══════════════════════════════════════════════════════════════
# Yardımcı: zaman damgalı başlık yazdır
# ═══════════════════════════════════════════════════════════════

def _banner(step: int, title: str, total: int = 5) -> None:
    bar  = "█" * step + "░" * (total - step)
    logger.info("")
    logger.info("┌─────────────────────────────────────────────────┐")
    logger.info("│  Adım %d/%d  %s  │", step, total, bar.ljust(10))
    logger.info("│  %-47s│", title)
    logger.info("└─────────────────────────────────────────────────┘")


def _elapsed(t0: float) -> str:
    s = time.perf_counter() - t0
    return f"{s:.2f}s" if s < 60 else f"{s/60:.1f}dk"


# ═══════════════════════════════════════════════════════════════
# Adım 1: Konfigürasyon özeti
# ═══════════════════════════════════════════════════════════════

def step_config() -> None:
    _banner(1, "Konfigürasyon Doğrulama")
    t0 = time.perf_counter()

    logger.info("  Proje   : %s v%s", cfg.project.name, cfg.project.version)
    logger.info("  Ortam   : %s", cfg.project.environment)
    logger.info("  Raw dir : %s", cfg.paths.data.raw)
    logger.info("  Model   : %s", cfg.paths.outputs.models)
    logger.info("  Reports : %s", cfg.paths.outputs.reports)

    # CSV dosyaları var mı kontrol et
    missing = []
    for key in ["routes", "route_stops", "traffic_segments",
                "weather_observations", "historical_delay_stats"]:
        p = cfg.get_raw_data_path(key)
        if not Path(p).exists():
            missing.append(str(p))

    if missing:
        logger.warning("  ⚠  Eksik CSV dosyaları:")
        for m in missing:
            logger.warning("       %s", m)
        logger.warning("  CSV'leri data/raw/ klasörüne kopyaladığından emin ol.")
    else:
        logger.info("  ✓  Tüm CSV dosyaları mevcut.")

    logger.info("  ──── Adım 1 tamamlandı (%s)", _elapsed(t0))


# ═══════════════════════════════════════════════════════════════
# Adım 2: CSV Okuma
# ═══════════════════════════════════════════════════════════════

def step_load_csv() -> dict[str, pd.DataFrame]:
    _banner(2, "CSV Dosyaları Yükleniyor")
    t0 = time.perf_counter()

    reader = CSVReader(cfg)
    dfs    = reader.load_all()
    reader.summary(dfs)

    total_rows = sum(len(v) for v in dfs.values() if not v.empty)
    loaded     = [k for k, v in dfs.items() if not v.empty]
    failed     = [k for k, v in dfs.items() if v.empty]

    logger.info("  Yüklenen tablolar  : %s", loaded)
    if failed:
        logger.warning("  Yüklenemeyen      : %s", failed)
    logger.info("  Toplam satır       : %d", total_rows)
    logger.info("  ──── Adım 2 tamamlandı (%s)", _elapsed(t0))

    return dfs


# ═══════════════════════════════════════════════════════════════
# Adım 3: Ön İşleme (Spatial-Temporal Join + Feature Engineering)
# ═══════════════════════════════════════════════════════════════

def step_preprocess(
    dfs: dict[str, pd.DataFrame],
    skip_join: bool = False,
) -> pd.DataFrame:
    _banner(3, "Ön İşleme & Feature Engineering")
    t0 = time.perf_counter()

    # Önbellek: eğer join atlanacaksa işlenmiş veri var mı?
    cache_path = cfg.paths.data.processed / "features_cache.parquet"
    if skip_join and cache_path.exists():
        logger.info("  --skip-join aktif. Cache'den yükleniyor: %s", cache_path)
        features = pd.read_parquet(cache_path)
        logger.info("  Cache'den yüklendi: %d satır, %d kolon.", len(features), len(features.columns))
        logger.info("  ──── Adım 3 atlandı (cache kullanıldı, %s)", _elapsed(t0))
        return features

    logger.info("  Spatial-Temporal Join başlıyor...")
    features = run_preprocessing_pipeline(dfs, cfg)

    logger.info("  Sonuç shape      : %d satır × %d kolon", len(features), len(features.columns))

    # Feature özeti
    dt_cols  = [c for c in features.columns if pd.api.types.is_datetime64_any_dtype(features[c])]
    num_cols = features.select_dtypes(include="number").columns.tolist()
    cat_cols = features.select_dtypes(include=["object", "category"]).columns.tolist()
    logger.info("  Datetime kolonlar: %d", len(dt_cols))
    logger.info("  Numeric kolonlar : %d", len(num_cols))
    logger.info("  Kategori kolonlar: %d", len(cat_cols))
    logger.info("  Toplam null      : %d", features.isnull().sum().sum())

    # Üretilen anahtar feature'ları kontrol et
    key_features = [
        "time_of_day_category", "weather_severity_index",
        "distance_to_next_stop", "delay_ratio",
        "is_delayed", "congestion_delay_factor",
        "weight_per_package",
    ]
    missing_feats = [f for f in key_features if f not in features.columns]
    if missing_feats:
        logger.warning("  ⚠  Eksik feature'lar: %s", missing_feats)
    else:
        logger.info("  ✓  Tüm anahtar feature'lar mevcut.")

    # Cache'e yaz (bir sonraki çalıştırmada --skip-join için)
    try:
        features.to_parquet(cache_path, index=False)
        logger.info("  Feature cache kaydedildi: %s", cache_path)
    except Exception as exc:
        logger.warning("  Cache yazılamadı (opsiyonel): %s", exc)

    logger.info("  ──── Adım 3 tamamlandı (%s)", _elapsed(t0))
    return features


# ═══════════════════════════════════════════════════════════════
# Adım 4: Model Eğitimi
# ═══════════════════════════════════════════════════════════════

def step_train(features: pd.DataFrame) -> dict:
    _banner(4, "XGBoost Model Eğitimi")
    t0 = time.perf_counter()

    # Target kolon var mı?
    TARGET = "delay_at_stop_min"
    if TARGET not in features.columns:
        logger.error("  ✗  Target kolon '%s' bulunamadı!", TARGET)
        logger.error("     Mevcut kolonlar: %s", features.columns.tolist()[:20])
        raise ValueError(f"Target kolon eksik: {TARGET}")

    valid_rows = features[TARGET].notna().sum()
    total_rows = len(features)
    logger.info("  Target kolon     : %s", TARGET)
    logger.info("  Geçerli satır    : %d / %d (%.1f%%)",
                valid_rows, total_rows, 100 * valid_rows / max(total_rows, 1))
    logger.info("  Target istatistikleri:")
    stats = features[TARGET].describe()
    logger.info("    min=%.2f  ortalama=%.2f  max=%.2f  std=%.2f",
                stats["min"], stats["mean"], stats["max"], stats["std"])

    logger.info("  XGBoost Pipeline eğitimi başlıyor...")
    trainer = DelayModelTrainer(cfg)
    metrics = trainer.train(features)

    logger.info("  ──── Adım 4 tamamlandı (%s)", _elapsed(t0))
    return metrics


# ═══════════════════════════════════════════════════════════════
# Adım 5: Özet Rapor
# ═══════════════════════════════════════════════════════════════

def step_summary(metrics: dict, pipeline_start: float) -> None:
    _banner(5, "Pipeline Tamamlandı — Özet Rapor")

    total_elapsed = time.perf_counter() - pipeline_start

    # Metrik tablosu
    logger.info("")
    logger.info("  ╔══════════════════════════════════════════════╗")
    logger.info("  ║            MODEL PERFORMANS METRİKLERİ      ║")
    logger.info("  ╠══════════════════════════════════════════════╣")
    logger.info("  ║  Train  MAE   : %6.2f dakika               ║", metrics.get("train_mae", 0))
    logger.info("  ║  Test   MAE   : %6.2f dakika  ◄ Ana metrik ║", metrics.get("mae", 0))
    logger.info("  ║  Test   RMSE  : %6.2f dakika               ║", metrics.get("rmse", 0))
    logger.info("  ║  Test   R²    : %8.4f                    ║", metrics.get("r2", 0))
    if metrics.get("cv_mae_mean"):
        logger.info("  ║  CV MAE (5-K) : %6.2f ± %-5.2f dakika     ║",
                    metrics["cv_mae_mean"], metrics.get("cv_mae_std", 0))
    logger.info("  ╠══════════════════════════════════════════════╣")
    logger.info("  ║  Eğitim seti  : %6d satır                ║", metrics.get("n_train", 0))
    logger.info("  ║  Test seti    : %6d satır                ║", metrics.get("n_test", 0))
    logger.info("  ╚══════════════════════════════════════════════╝")

    # Overfitting uyarısı
    train_mae = metrics.get("train_mae", 0)
    test_mae  = metrics.get("mae", 0)
    gap       = test_mae - train_mae
    if gap > 5:
        logger.warning("")
        logger.warning("  ⚠  Olası overfitting tespit edildi!")
        logger.warning("     Train MAE=%.2f  Test MAE=%.2f  Fark=%.2f dk",
                       train_mae, test_mae, gap)
        logger.warning("     settings.yaml'da XGBoost regularizasyon değerlerini artırmayı dene.")
    elif test_mae < 5:
        logger.info("")
        logger.info("  ✓  MAE < 5 dk — model üretim kalitesinde.")
    elif test_mae < 15:
        logger.info("")
        logger.info("  ✓  MAE < 15 dk — kabul edilebilir performans.")
    else:
        logger.info("")
        logger.info("  ~  MAE > 15 dk — daha fazla veri veya feature tuning önerilir.")

    # Dosya konumları
    logger.info("")
    logger.info("  Kaydedilen dosyalar:")
    logger.info("    Model  → %s", metrics.get("model_path", "?"))
    logger.info("    FI CSV → %s", metrics.get("feature_importance_path", "?"))

    logger.info("")
    logger.info("  ⏱  Toplam pipeline süresi: %.1f saniye", total_elapsed)
    logger.info("")
    logger.info("  Sonraki adım:")
    logger.info("    uvicorn src.api.fastapi_app:app --reload --port 8000")
    logger.info("")


# ═══════════════════════════════════════════════════════════════
# Argüman ayrıştırma
# ═══════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Akıllı Lojistik — Eğitim Pipeline Orkestratörü",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python main_pipeline.py                  # Tam pipeline
  python main_pipeline.py --skip-join      # Join atla, cache kullan
  python main_pipeline.py --debug          # DEBUG log seviyesi
        """,
    )
    parser.add_argument(
        "--skip-join",
        action="store_true",
        help="Spatial-Temporal Join'ı atla; data/processed/features_cache.parquet kullan.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Log seviyesini DEBUG'a yükselt.",
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════
# Ana giriş noktası
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args_safe()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("DEBUG modu aktif.")

    pipeline_start = time.perf_counter()

    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║   Akıllı Lojistik — Eğitim Pipeline Başlıyor        ║")
    logger.info("╚══════════════════════════════════════════════════════╝")

    try:
        # ── Adım 1: Config ───────────────────────────────────────
        step_config()

        # ── Adım 2: CSV Okuma ────────────────────────────────────
        dfs = step_load_csv()

        if all(v.empty for v in dfs.values()):
            logger.error("")
            logger.error("  ✗  Hiçbir CSV yüklenemedi.")
            logger.error("     CSV dosyalarını data/raw/ klasörüne kopyala ve tekrar çalıştır.")
            logger.error("")
            sys.exit(1)

        # ── Adım 3: Ön İşleme ────────────────────────────────────
        features = step_preprocess(dfs, skip_join=args.skip_join)

        if features.empty:
            logger.error("  ✗  Feature DataFrame boş. Pipeline durduruluyor.")
            sys.exit(1)

        # ── Adım 4: Eğitim ───────────────────────────────────────
        metrics = step_train(features)

        # ── Adım 5: Özet ─────────────────────────────────────────
        step_summary(metrics, pipeline_start)

    except KeyboardInterrupt:
        logger.warning("")
        logger.warning("  Pipeline kullanıcı tarafından durduruldu (Ctrl+C).")
        sys.exit(130)

    except Exception as exc:
        logger.error("")
        logger.error("  ✗  Pipeline kritik hata ile durdu: %s", exc, exc_info=True)
        logger.error("     Log dosyası: %s", cfg.paths.logs / "app.log")
        sys.exit(1)


def parse_args_safe() -> argparse.Namespace:
    """Test ortamında argparse crash'ini önler."""
    try:
        return _parse_args()
    except SystemExit:
        # pytest çalıştırırken argparse sys.argv'yi karıştırabilir
        return argparse.Namespace(skip_join=False, debug=False)


if __name__ == "__main__":
    main()
