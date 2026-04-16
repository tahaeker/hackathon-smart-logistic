"""
src/ml/predictor.py

Diske kaydedilmiş XGBoost Pipeline'ı yükler ve gelen veri satırları
için gecikme tahmini (delay_at_stop_min) üretir.

Tasarım:
  • DelayPredictor sınıfı Singleton pattern ile model dosyasını
    yalnızca bir kez yükler (tekrar eden API çağrılarında I/O yoktur).
  • predict() metodu; tek satır dict, çok satırlı DataFrame veya
    JSON-uyumlu liste-dict formatlarını kabul eder.
  • predict_with_confidence() → tahmin + güven aralığı (bootstrap).

Kullanım:
    from src.ml.predictor import DelayPredictor

    predictor = DelayPredictor.get_instance()
    result    = predictor.predict(row_dict)
    # {"predicted_delay_min": 14.3, "status": "ok"}

    batch = predictor.predict_batch(df)
    # pd.Series of predictions
"""

import logging
import threading
from pathlib import Path
from typing import Any, Optional, Union

import joblib
import numpy as np
import pandas as pd

from config.config_loader import Config, get_config

logger = logging.getLogger(__name__)

MODEL_FILE = "xgboost_delay_model.joblib"

# ══════════════════════════════════════════════════════════════
# Singleton yardımcı meta-sınıfı (thread-safe)
# ══════════════════════════════════════════════════════════════

class _SingletonMeta(type):
    """
    Thread-safe Singleton meta-sınıfı.
    Her Config yoluna karşılık tek bir DelayPredictor örneği tutar.
    """
    _instances: dict[str, Any] = {}
    _lock: threading.Lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        # Anahtar = model dosya yolu (farklı modeller → farklı singleton)
        cfg: Config = kwargs.get("config") or (args[0] if args else get_config())
        key = str(cfg.get_model_path(MODEL_FILE))

        with cls._lock:
            if key not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[key] = instance
        return cls._instances[key]

    @classmethod
    def reset(cls, key: Optional[str] = None) -> None:
        """Test veya model güncellemesi sonrası singleton'ı temizler."""
        with cls._lock:
            if key:
                cls._instances.pop(key, None)
            else:
                cls._instances.clear()


# ══════════════════════════════════════════════════════════════
# DelayPredictor
# ══════════════════════════════════════════════════════════════

class DelayPredictor(metaclass=_SingletonMeta):
    """
    Kaydedilmiş XGBoost Pipeline'ını yükleyip tahmin yapan sınıf.

    Singleton olduğu için model dosyası yalnızca ilk çağrıda okunur.
    Sonraki çağrılar aynı nesneden tahmin alır.

    Kullanım:
        predictor = DelayPredictor(get_config())
        # veya kısaca:
        predictor = DelayPredictor.get_instance()
    """

    def __init__(self, config: Optional[Config] = None):
        # Singleton meta __call__'ı yönettiği için __init__ birden
        # fazla çalışabilir; model yükleme sadece bir kez yapılsın.
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.cfg      = config or get_config()
        self.pipeline = None
        self._model_path: Path = self.cfg.get_model_path(MODEL_FILE)
        self._initialized      = False

        self._load_model()

    # ──────────────────────────────────────────────────────────
    # Alternatif giriş noktası
    # ──────────────────────────────────────────────────────────
    @classmethod
    def get_instance(cls, config: Optional[Config] = None) -> "DelayPredictor":
        """Singleton örneğini döner (ilk çağrıda oluşturur)."""
        return cls(config=config or get_config())

    # ──────────────────────────────────────────────────────────
    # Model yükleme
    # ──────────────────────────────────────────────────────────
    def _load_model(self) -> None:
        """
        Joblib ile model dosyasını yükler.
        Dosya yoksa ModelNotFoundError fırlatır.
        """
        try:
            if not self._model_path.exists():
                raise FileNotFoundError(
                    f"Model dosyası bulunamadı: {self._model_path}\n"
                    "Önce trainer.py ile modeli eğit ve kaydet."
                )

            logger.info("Model yükleniyor: %s", self._model_path)
            self.pipeline     = joblib.load(self._model_path)
            self._initialized = True
            logger.info(
                "Model başarıyla yüklendi. Pipeline adımları: %s",
                [step[0] for step in self.pipeline.steps],
            )

        except FileNotFoundError:
            raise
        except Exception as exc:
            logger.error("Model yüklenirken hata: %s", exc, exc_info=True)
            raise RuntimeError(f"Model yüklenemedi: {exc}") from exc

    def reload(self) -> None:
        """
        Model dosyasını yeniden yükler.
        Yeni bir model eğitildikten sonra çağrılabilir.
        """
        logger.info("Model yeniden yükleniyor...")
        self._initialized = False
        self._load_model()

    # ──────────────────────────────────────────────────────────
    # Tek satır tahmini
    # ──────────────────────────────────────────────────────────
    def predict(
        self,
        data: Union[dict, pd.DataFrame, pd.Series],
    ) -> dict[str, Any]:
        """
        Tek bir satır için gecikme tahmini yapar.

        Args:
            data: Tahmin yapılacak veri.
                  dict     → tek satır (JSON'dan gelen)
                  pd.Series → tek satır
                  pd.DataFrame → ilk satır kullanılır

        Returns:
            {
              "predicted_delay_min": float,   ← tahmin (dakika)
              "status": "ok" | "error",
              "message": str                  ← hata durumunda
            }
        """
        try:
            self._check_ready()
            df = self._to_dataframe(data)

            if df.empty:
                raise ValueError("Giriş verisi boş DataFrame'e dönüştürüldü.")

            # Eksik kolonları NaN ile tamamla → SimpleImputer halleder
            df  = self._align_columns(df)
            row = df.iloc[[0]]
            raw_pred = float(self.pipeline.predict(row)[0])

            # Negatif tahminleri sıfırla (fiziksel anlam)
            predicted = max(0.0, round(raw_pred, 2))

            logger.debug("Tahmin: %.2f dk", predicted)
            return {
                "predicted_delay_min": predicted,
                "status": "ok",
            }

        except FileNotFoundError as exc:
            logger.error("predict() — model dosyası yok: %s", exc)
            return {"predicted_delay_min": None, "status": "error", "message": str(exc)}
        except KeyError as exc:
            logger.error("predict() — eksik kolon: %s", exc)
            return {"predicted_delay_min": None, "status": "error",
                    "message": f"Eksik kolon: {exc}"}
        except Exception as exc:
            logger.error("predict() — beklenmeyen hata: %s", exc, exc_info=True)
            return {"predicted_delay_min": None, "status": "error", "message": str(exc)}

    # ──────────────────────────────────────────────────────────
    # Toplu tahmin
    # ──────────────────────────────────────────────────────────
    def predict_batch(
        self,
        df: pd.DataFrame,
        return_dataframe: bool = False,
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Birden fazla satır için toplu tahmin yapar.

        Args:
            df              : Tahmin yapılacak DataFrame.
            return_dataframe: True → tahmini df'e 'predicted_delay_min'
                              kolonu olarak ekleyip döner.

        Returns:
            pd.Series (tahmin değerleri) veya df + tahmin kolonu.
        """
        try:
            self._check_ready()

            if df.empty:
                raise ValueError("Giriş DataFrame'i boş.")

            raw_preds = self.pipeline.predict(df)
            preds     = pd.Series(
                np.maximum(0.0, raw_preds).round(2),
                index=df.index,
                name="predicted_delay_min",
            )

            logger.info(
                "Toplu tahmin tamamlandı: %d satır | ort=%.2f dk | "
                "min=%.2f dk | max=%.2f dk",
                len(preds), preds.mean(), preds.min(), preds.max(),
            )

            if return_dataframe:
                result_df = df.copy()
                result_df["predicted_delay_min"] = preds
                return result_df

            return preds

        except Exception as exc:
            logger.error("predict_batch() hatası: %s", exc, exc_info=True)
            raise

    # ──────────────────────────────────────────────────────────
    # Tahmin + Güven Aralığı (Bootstrap)
    # ──────────────────────────────────────────────────────────
    def predict_with_confidence(
        self,
        data: Union[dict, pd.DataFrame, pd.Series],
        n_bootstrap: int = 100,
        ci: float = 0.90,
    ) -> dict[str, Any]:
        """
        Bootstrap yöntemiyle tahmin güven aralığı üretir.
        XGBoost deterministik olduğundan Gaussian noise eklenerek
        yaklaşık bir dağılım simüle edilir.

        Args:
            data       : Tahmin girdisi.
            n_bootstrap: Bootstrap yineleme sayısı.
            ci         : Güven aralığı genişliği (0.90 → %90 CI).

        Returns:
            {
              "predicted_delay_min": float,
              "ci_lower": float,
              "ci_upper": float,
              "std_dev": float,
              "confidence_level": float,
              "status": "ok"
            }
        """
        try:
            self._check_ready()
            df  = self._to_dataframe(data)
            df  = self._align_columns(df)
            df  = df.iloc[[0]]
            rng = np.random.default_rng(seed=42)

            base_pred = float(self.pipeline.predict(df)[0])
            noise_std = abs(base_pred) * 0.05 + 0.5   # %5 + sabit 0.5 dk

            bootstrap_preds = [
                max(0.0, base_pred + rng.normal(0, noise_std))
                for _ in range(n_bootstrap)
            ]

            alpha  = (1 - ci) / 2
            lower  = float(np.quantile(bootstrap_preds, alpha))
            upper  = float(np.quantile(bootstrap_preds, 1 - alpha))
            std    = float(np.std(bootstrap_preds))

            return {
                "predicted_delay_min": round(max(0.0, base_pred), 2),
                "ci_lower":            round(lower, 2),
                "ci_upper":            round(upper, 2),
                "std_dev":             round(std, 2),
                "confidence_level":    ci,
                "status":              "ok",
            }

        except Exception as exc:
            logger.error("predict_with_confidence() hatası: %s", exc, exc_info=True)
            return {"status": "error", "message": str(exc)}

    # ──────────────────────────────────────────────────────────
    # Model bilgisi
    # ──────────────────────────────────────────────────────────
    def info(self) -> dict[str, Any]:
        """
        Yüklü model hakkında meta bilgi döner.
        API'nin /health veya /model-info endpoint'inde kullanılabilir.
        """
        try:
            self._check_ready()
            xgb = self.pipeline.named_steps["model"]
            return {
                "model_file":    str(self._model_path),
                "model_type":    type(xgb).__name__,
                "n_estimators":  xgb.n_estimators,
                "max_depth":     xgb.max_depth,
                "learning_rate": xgb.learning_rate,
                "status":        "loaded",
            }
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    # ──────────────────────────────────────────────────────────
    # Yardımcı metodlar
    # ──────────────────────────────────────────────────────────
    def _check_ready(self) -> None:
        """Model yüklü değilse RuntimeError fırlatır."""
        if not self._initialized or self.pipeline is None:
            raise RuntimeError(
                "Model henüz yüklenmedi. "
                "DelayPredictor() veya DelayPredictor.get_instance() çağır."
            )

    def _align_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pipeline'ın eğitim sırasında gördüğü kolonlara göre
        giriş DataFrame'ini hizalar.

        Eksik kolonlar NaN olarak eklenir — pipeline içindeki
        SimpleImputer(strategy='median') bunları doldurur.
        Fazladan kolonlar sessizce bırakılır (ColumnTransformer
        remainder='drop' ile zaten atıyor).
        """
        try:
            preprocessor = self.pipeline.named_steps["preprocessor"]
            required: list[str] = []
            for _, _, cols in preprocessor.transformers_:
                if isinstance(cols, list):
                    required.extend(cols)

            df = df.copy()
            for col in required:
                if col not in df.columns:
                    df[col] = np.nan
            return df
        except Exception as exc:
            logger.debug("_align_columns atlanamadı: %s", exc)
            return df

    @staticmethod
    def _to_dataframe(
        data: Union[dict, pd.DataFrame, pd.Series]
    ) -> pd.DataFrame:
        """Giriş verisini güvenli biçimde DataFrame'e dönüştürür."""
        try:
            if isinstance(data, pd.DataFrame):
                return data
            if isinstance(data, pd.Series):
                return data.to_frame().T.reset_index(drop=True)
            if isinstance(data, dict):
                return pd.DataFrame([data])
            if isinstance(data, list):
                return pd.DataFrame(data)
            raise TypeError(
                f"Desteklenmeyen veri tipi: {type(data)}. "
                "dict, pd.Series, pd.DataFrame veya list bekleniyor."
            )
        except Exception as exc:
            raise ValueError(f"Veri DataFrame'e dönüştürülemedi: {exc}") from exc


# ──────────────────────────────────────────────────────────────
# Hızlı test
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    predictor = DelayPredictor.get_instance()
    print("Model bilgisi:", predictor.info())

    # Örnek tahmin (eğitim sırasında kullanılan feature kolonlarıyla)
    sample = {
        "hour_of_day": 8,
        "day_of_week": 1,
        "is_rush_hour": 1,
        "is_weekend": 0,
        "distance_from_prev_km": 12.5,
        "planned_travel_min": 25.0,
        "weather_severity_index": 0.3,
        "congestion_delay_factor": 1.4,
        "time_of_day_category": "morning_rush",
        "road_type": "highway",
    }

    result = predictor.predict(sample)
    print("Tahmin sonucu :", result)

    ci_result = predictor.predict_with_confidence(sample, n_bootstrap=200)
    print("Güven aralığı :", ci_result)

    # Singleton testi
    p2 = DelayPredictor.get_instance()
    print("Singleton kontrolü:", predictor is p2)   # → True
