"""
config/config_loader.py

settings.yaml'ı yükler, path'leri resolve eder ve proje genelinde
tek bir Config nesnesi sağlar (singleton pattern).

Kullanım:
    from config.config_loader import get_config
    cfg = get_config()
    print(cfg.paths.data.raw)          # → Path("data/raw")
    print(cfg.models.delay_predictor)  # → dict
"""

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# -----------------------------------------------------------
# Sabitler
# -----------------------------------------------------------
CONFIG_DIR = Path(__file__).parent
PROJECT_ROOT = CONFIG_DIR.parent
DEFAULT_SETTINGS_FILE = CONFIG_DIR / "settings.yaml"


# -----------------------------------------------------------
# Basit dot-access wrapper (cfg.key.subkey şeklinde erişim)
# -----------------------------------------------------------
class ConfigNode:
    """Dict'i özyinelemeli olarak dot-access nesnesine dönüştürür."""

    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigNode(value))
            else:
                setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict:
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigNode):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        return f"ConfigNode({self.to_dict()})"


# -----------------------------------------------------------
# Path resolver — tüm path değerlerini PROJECT_ROOT'a göre çözer
# -----------------------------------------------------------
def _resolve_paths(paths_node: ConfigNode) -> None:
    """
    config.paths altındaki tüm string path değerlerini
    PROJECT_ROOT'a göre mutlak Path nesnelerine dönüştürür.
    """
    for attr in vars(paths_node):
        value = getattr(paths_node, attr)
        if isinstance(value, ConfigNode):
            _resolve_paths(value)
        elif isinstance(value, str):
            setattr(paths_node, attr, PROJECT_ROOT / value)


# -----------------------------------------------------------
# Logger kurulumu (settings.yaml yüklenmeden önce basit kurulum)
# -----------------------------------------------------------
def _setup_logging(log_cfg: ConfigNode) -> None:
    log_dir = PROJECT_ROOT / log_cfg.get("file", "logs/app.log")
    log_dir.parent.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, log_cfg.get("level", "INFO"), logging.INFO)
    fmt = log_cfg.get(
        "format",
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    date_fmt = log_cfg.get("date_format", "%Y-%m-%d %H:%M:%S")

    handlers: list[logging.Handler] = []

    # Dosya handler
    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(PROJECT_ROOT / log_cfg.get("file", "logs/app.log")),
        maxBytes=log_cfg.get("max_bytes", 10_485_760),
        backupCount=log_cfg.get("backup_count", 5),
        encoding="utf-8",
    )
    handlers.append(file_handler)

    # Konsol handler
    if log_cfg.get("console", True):
        handlers.append(logging.StreamHandler())

    logging.basicConfig(level=level, format=fmt, datefmt=date_fmt, handlers=handlers)


# -----------------------------------------------------------
# Ana yükleyici
# -----------------------------------------------------------
class Config:
    """
    settings.yaml'dan yüklenen tüm konfigürasyonu tutan sınıf.
    Attributes PROJECT_ROOT'a göre resolve edilmiş Path nesneleridir.
    """

    def __init__(self, settings_file: Path = DEFAULT_SETTINGS_FILE):
        if not settings_file.exists():
            raise FileNotFoundError(
                f"settings.yaml bulunamadı: {settings_file}\n"
                f"Beklenen konum: {DEFAULT_SETTINGS_FILE}"
            )

        with open(settings_file, encoding="utf-8") as f:
            raw: dict = yaml.safe_load(f)

        # Tüm bölümleri ConfigNode'a dönüştür
        for section, data in raw.items():
            if isinstance(data, dict):
                setattr(self, section, ConfigNode(data))
            else:
                setattr(self, section, data)

        # Path'leri resolve et
        if hasattr(self, "paths"):
            _resolve_paths(self.paths)

        # OUTPUT klasörlerini oluştur
        self._ensure_directories()

        # Logging kur
        if hasattr(self, "logging"):
            import logging.handlers  # noqa: PLC0415  (lazy import)
            _setup_logging(self.logging)

        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.info(
            "Konfigürasyon yüklendi: %s (env=%s)",
            settings_file.name,
            getattr(getattr(self, "project", None), "environment", "?"),
        )

    def _ensure_directories(self) -> None:
        """Outputs ve logs klasörlerini yoksa oluşturur."""
        if not hasattr(self, "paths"):
            return
        try:
            outputs = self.paths.get("outputs")
            if outputs:
                for attr in vars(outputs):
                    path: Path = getattr(outputs, attr)
                    if isinstance(path, Path):
                        path.mkdir(parents=True, exist_ok=True)

            logs = self.paths.get("logs")
            if isinstance(logs, Path):
                logs.mkdir(parents=True, exist_ok=True)

            data = self.paths.get("data")
            if data:
                for attr in vars(data):
                    path = getattr(data, attr)
                    if isinstance(path, Path):
                        path.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass  # Klasör oluşturma hatası kritik değil

    # ----------------------------------------------------------
    # Kolaylık metodları
    # ----------------------------------------------------------
    def get_raw_data_path(self, filename_key: str) -> Path:
        """
        data_files altındaki key'e karşılık gelen tam dosya yolunu döner.
        Örnek: cfg.get_raw_data_path("routes") → Path("data/raw/routes.csv")
        """
        data_files: ConfigNode = getattr(self, "data_files", None)
        if data_files is None:
            raise AttributeError("settings.yaml içinde 'data_files' bölümü bulunamadı.")
        filename: str = getattr(data_files, filename_key, None)
        if filename is None:
            raise KeyError(f"data_files altında '{filename_key}' anahtarı bulunamadı.")
        return self.paths.data.raw / filename

    def get_processed_data_path(self, filename: str) -> Path:
        """data/processed/ altında bir dosya yolu döner."""
        return self.paths.data.processed / filename

    def get_model_path(self, filename: str) -> Path:
        """outputs/models/ altında bir dosya yolu döner."""
        return self.paths.outputs.models / filename

    def get_schema(self, table: str) -> ConfigNode:
        """schema.<table> ConfigNode'unu döner."""
        schema: ConfigNode = getattr(self, "schema", None)
        if schema is None:
            raise AttributeError("settings.yaml içinde 'schema' bölümü bulunamadı.")
        table_schema = getattr(schema, table, None)
        if table_schema is None:
            raise KeyError(f"schema altında '{table}' bulunamadı.")
        return table_schema

    def __repr__(self) -> str:
        env = getattr(getattr(self, "project", None), "environment", "?")
        return f"<Config env={env} root={PROJECT_ROOT}>"


# -----------------------------------------------------------
# Singleton erişim noktası
# -----------------------------------------------------------
@lru_cache(maxsize=1)
def get_config(settings_file: str | None = None) -> Config:
    """
    Config nesnesini döner (ilk çağrıda oluşturur, sonra cache'den verir).

    Args:
        settings_file: Özel bir settings dosyası belirtmek için (opsiyonel).
                       None ise config/settings.yaml kullanılır.

    Returns:
        Config nesnesi
    """
    path = Path(settings_file) if settings_file else DEFAULT_SETTINGS_FILE
    return Config(settings_file=path)


# -----------------------------------------------------------
# Hızlı test — doğrudan çalıştırılırsa
# -----------------------------------------------------------
if __name__ == "__main__":
    import logging.handlers  # noqa

    cfg = get_config()
    print(f"Proje   : {cfg.project.name} v{cfg.project.version}")
    print(f"Root    : {PROJECT_ROOT}")
    print(f"Raw dir : {cfg.paths.data.raw}")
    print(f"Models  : {cfg.paths.outputs.models}")
    print(f"Routes  : {cfg.get_raw_data_path('routes')}")
    print(f"Stops   : {cfg.get_raw_data_path('route_stops')}")
    schema = cfg.get_schema("routes")
    print(f"Target  : {schema.target_col}")
    print(f"DT cols : {schema.datetime_cols}")
    print("\nConfig nesnesi hazır.")
