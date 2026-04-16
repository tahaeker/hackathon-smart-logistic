"""
src/data_ingestion/csv_reader.py

settings.yaml'dan dosya yollarını ve şema tanımlarını alıp
tüm CSV dosyalarını tip-güvenli biçimde Pandas DataFrame olarak yükler.

Kullanım:
    from config.config_loader import get_config
    from src.data_ingestion.csv_reader import CSVReader

    reader = CSVReader(get_config())
    dfs    = reader.load_all()
    routes = dfs["routes"]
    stops  = dfs["route_stops"]
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from config.config_loader import Config, ConfigNode, get_config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Yardımcı: schema listesini güvenle al
# ──────────────────────────────────────────────────────────────
def _safe_list(node: ConfigNode, attr: str) -> list:
    """ConfigNode'dan liste attribute'ü döner; yoksa boş liste."""
    val = getattr(node, attr, None)
    return val if isinstance(val, list) else []


# ──────────────────────────────────────────────────────────────
# Ana sınıf
# ──────────────────────────────────────────────────────────────
class CSVReader:
    """
    settings.yaml şemasına göre CSV dosyalarını okuyan sınıf.

    - Dosya yollarını config üzerinden çözer.
    - datetime kolonlarını pd.to_datetime ile parse eder.
    - numeric / categorical / binary kolonları doğru tipe cast eder.
    - Eksik dosya veya kolon hatalarını loglayıp yönetir.
    """

    # settings.yaml data_files altındaki key → schema altındaki key eşlemesi
    _TABLE_KEYS: list[str] = [
        "routes",
        "route_stops",
        "traffic_segments",
        "weather_observations",
        "historical_delay_stats",
    ]

    def __init__(self, config: Optional[Config] = None):
        self.cfg = config or get_config()
        self._dt_format: str = getattr(
            getattr(self.cfg, "schema", None), "datetime_format", "%Y-%m-%d %H:%M:%S"
        )
        logger.info("CSVReader başlatıldı. Datetime formatı: %s", self._dt_format)

    # ──────────────────────────────────────────────────────────
    # Tek tablo yükleyici
    # ──────────────────────────────────────────────────────────
    def load_table(self, table_key: str) -> pd.DataFrame:
        """
        Belirtilen tabloyu yükleyip tip dönüşümlerini uygular.

        Args:
            table_key: "routes" | "route_stops" | "traffic_segments"
                       | "weather_observations" | "historical_delay_stats"

        Returns:
            pd.DataFrame (boş DataFrame hata durumunda)
        """
        try:
            file_path = self.cfg.get_raw_data_path(table_key)
            logger.info("[%s] Okunuyor: %s", table_key, file_path)

            if not Path(file_path).exists():
                raise FileNotFoundError(
                    f"CSV dosyası bulunamadı: {file_path}\n"
                    f"CSV'yi data/raw/ klasörüne kopyaladığından emin ol."
                )

            df = pd.read_csv(file_path, low_memory=False)
            logger.info("[%s] %d satır, %d kolon yüklendi.", table_key, len(df), len(df.columns))

            df = self._apply_schema(df, table_key)
            return df

        except FileNotFoundError as exc:
            logger.error("[%s] Dosya hatası: %s", table_key, exc)
            return pd.DataFrame()
        except pd.errors.ParserError as exc:
            logger.error("[%s] CSV parse hatası: %s", table_key, exc)
            return pd.DataFrame()
        except Exception as exc:
            logger.error("[%s] Beklenmeyen hata: %s", table_key, exc, exc_info=True)
            return pd.DataFrame()

    # ──────────────────────────────────────────────────────────
    # Tüm tabloları yükle
    # ──────────────────────────────────────────────────────────
    def load_all(self) -> dict[str, pd.DataFrame]:
        """
        Tüm tanımlı tabloları yükler.

        Returns:
            {table_key: pd.DataFrame} sözlüğü
        """
        result: dict[str, pd.DataFrame] = {}
        for key in self._TABLE_KEYS:
            result[key] = self.load_table(key)

        loaded = [k for k, v in result.items() if not v.empty]
        failed = [k for k, v in result.items() if v.empty]

        logger.info("Yükleme tamamlandı. Başarılı: %s | Başarısız: %s", loaded, failed)
        return result

    # ──────────────────────────────────────────────────────────
    # Şema uygulama
    # ──────────────────────────────────────────────────────────
    def _apply_schema(self, df: pd.DataFrame, table_key: str) -> pd.DataFrame:
        """DataFrame'e settings.yaml şemasındaki tip dönüşümlerini uygular."""
        try:
            schema: ConfigNode = self.cfg.get_schema(table_key)
        except (AttributeError, KeyError):
            logger.warning("[%s] Şema bulunamadı, ham DataFrame döndürülüyor.", table_key)
            return df

        df = self._parse_datetimes(df, schema, table_key)
        df = self._cast_numeric(df, schema, table_key)
        df = self._cast_categorical(df, schema, table_key)
        df = self._cast_binary(df, schema, table_key)
        return df

    def _parse_datetimes(
        self, df: pd.DataFrame, schema: ConfigNode, table_key: str
    ) -> pd.DataFrame:
        """
        datetime_cols listesindeki tüm kolonları pd.to_datetime ile parse eder.
        Hatalı parse → NaT (satırı silmez, uyarı loglar).
        """
        dt_cols = _safe_list(schema, "datetime_cols")
        for col in dt_cols:
            if col not in df.columns:
                logger.debug("[%s] datetime kolonu eksik, atlanıyor: %s", table_key, col)
                continue
            try:
                before_nulls = df[col].isna().sum()
                df[col] = pd.to_datetime(df[col], format=self._dt_format, errors="coerce")
                after_nulls = df[col].isna().sum()
                new_nulls = after_nulls - before_nulls
                if new_nulls > 0:
                    logger.warning(
                        "[%s] '%s' kolonunda %d satır NaT'a dönüştü (format uyumsuzluğu).",
                        table_key, col, new_nulls,
                    )
                logger.debug("[%s] '%s' → datetime64 dönüşümü OK.", table_key, col)
            except Exception as exc:
                logger.error(
                    "[%s] '%s' datetime parse hatası: %s", table_key, col, exc
                )
        return df

    def _cast_numeric(
        self, df: pd.DataFrame, schema: ConfigNode, table_key: str
    ) -> pd.DataFrame:
        """numeric_cols listesindeki kolonları float64'e cast eder."""
        num_cols = _safe_list(schema, "numeric_cols")
        for col in num_cols:
            if col not in df.columns:
                continue
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception as exc:
                logger.error("[%s] '%s' numeric cast hatası: %s", table_key, col, exc)
        return df

    def _cast_categorical(
        self, df: pd.DataFrame, schema: ConfigNode, table_key: str
    ) -> pd.DataFrame:
        """categorical_cols listesindeki kolonları string'e normalize eder."""
        cat_cols = _safe_list(schema, "categorical_cols")
        for col in cat_cols:
            if col not in df.columns:
                continue
            try:
                df[col] = df[col].astype(str).str.strip().str.lower()
            except Exception as exc:
                logger.error("[%s] '%s' categorical cast hatası: %s", table_key, col, exc)
        return df

    def _cast_binary(
        self, df: pd.DataFrame, schema: ConfigNode, table_key: str
    ) -> pd.DataFrame:
        """binary_cols listesindeki kolonları Int8'e (nullable) cast eder."""
        bin_cols = _safe_list(schema, "binary_cols")
        for col in bin_cols:
            if col not in df.columns:
                continue
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int8")
            except Exception as exc:
                logger.error("[%s] '%s' binary cast hatası: %s", table_key, col, exc)
        return df

    # ──────────────────────────────────────────────────────────
    # Özet rapor
    # ──────────────────────────────────────────────────────────
    def summary(self, dataframes: dict[str, pd.DataFrame]) -> None:
        """Yüklenen DataFrame'lerin boyut ve null özetini loglar."""
        logger.info("=" * 60)
        logger.info("CSV YÜKLEME ÖZETİ")
        logger.info("=" * 60)
        for key, df in dataframes.items():
            if df.empty:
                logger.warning("  %-30s → BOŞ (yüklenemedi)", key)
            else:
                null_counts = df.isnull().sum()
                total_nulls = null_counts.sum()
                logger.info(
                    "  %-30s → %6d satır | %3d kolon | %d null",
                    key, len(df), len(df.columns), total_nulls,
                )
        logger.info("=" * 60)


# ──────────────────────────────────────────────────────────────
# Hızlı test
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    cfg = get_config()
    reader = CSVReader(cfg)
    dfs = reader.load_all()
    reader.summary(dfs)

    # Temel doğrulamalar
    for name, df in dfs.items():
        if not df.empty:
            print(f"\n[{name}] dtypes:")
            dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
            print(f"  Datetime kolonlar : {dt_cols}")
            print(f"  Toplam null       : {df.isnull().sum().sum()}")
            print(f"  İlk satır:\n{df.iloc[0].to_dict()}")
