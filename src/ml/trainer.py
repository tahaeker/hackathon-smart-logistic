"""
src/ml/trainer.py

preprocessor.py çıktısını alır, bir scikit-learn Pipeline içinde
XGBoost Regressor eğitir ve modeli diske kaydeder.

Sorumluluklar:
  1. Feature seçimi  — numeric + categorical kolonları otomatik saptar
  2. Pipeline inşası — ColumnTransformer (StandardScaler | OneHotEncoder)
                       + XGBRegressor
  3. Eğitim          — train_test_split + cross-validation
  4. Metrik raporu   — MAE, RMSE, R² (train & test)
  5. Kayıt           — outputs/models/xgboost_delay_model.joblib
  6. Feature Importance — outputs/reports/feature_importance.csv

Kullanım:
    from config.config_loader import get_config
    from src.ml.trainer import DelayModelTrainer

    trainer = DelayModelTrainer(get_config())
    result  = trainer.train(features_df)
    print(result)   # {"mae": ..., "rmse": ..., "r2": ..., "model_path": ...}
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from config.config_loader import Config, get_config

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Sabitler
# ──────────────────────────────────────────────────────────────
TARGET_COL   = "delay_at_stop_min"
MODEL_FILE   = "xgboost_delay_model.joblib"
FI_FILE      = "feature_importance.csv"
METRICS_FILE = "train_metrics.json"

# Modelde kullanılacak aday feature kolonlar
# (DataFrame'de bulunanlar otomatik filtrelenir)
NUMERIC_CANDIDATES: list[str] = [
    "stop_sequence",
    "distance_from_prev_km",
    "distance_to_next_stop",
    "planned_service_min",
    "planned_travel_min",
    "package_count",
    "package_weight_kg",
    "weight_per_package",
    "delay_probability",
    # Hava durumu
    "w_temperature_c",    "temperature_c",
    "w_precipitation_mm", "precipitation_mm",
    "w_wind_speed_kmh",   "wind_speed_kmh",
    "w_visibility_km",    "visibility_km",
    "w_humidity_pct",     "humidity_pct",
    "w_delay_risk_score",
    "weather_severity_index",
    "effective_visibility",
    # Trafik
    "t_congestion_ratio", "congestion_ratio_eff",
    "t_free_flow_speed_kmh",
    "t_current_speed_kmh",
    "speed_ratio",
    "congestion_delay_factor",
    "incident_severity",
    # Zaman
    "hour_of_day",
    "day_of_week",
    "month",
    "day_of_month",
    "is_rush_hour",
    "is_weekend",
    # Gecikme
    "delay_ratio",
    "cumulative_delay_at_stop",
    "overall_delay_factor",
]

CATEGORICAL_CANDIDATES: list[str] = [
    "road_type",
    "vehicle_type",
    "weather_condition",   "w_weather_condition",
    "traffic_level",       "t_traffic_level",
    "time_of_day_category",
    "congestion_category",
    "precip_intensity_cat",
    "w_road_surface_condition",
    "w_precipitation_type",
]

# ══════════════════════════════════════════════════════════════
# DelayModelTrainer
# ══════════════════════════════════════════════════════════════

class DelayModelTrainer:
    """
    XGBoost tabanlı gecikme tahmin modeli eğiticisi.

    Pipeline:
        ColumnTransformer
          ├── numeric  → SimpleImputer(median) → StandardScaler
          └── categoric→ SimpleImputer(constant='missing') → OneHotEncoder
        XGBRegressor

    Attributes:
        cfg          : Config nesnesi
        pipeline     : eğitilmiş sklearn Pipeline (train() sonrası)
        feature_names: modele giren kolon isimleri listesi
    """

    def __init__(self, config: Optional[Config] = None):
        self.cfg: Config           = config or get_config()
        self.pipeline: Optional[Pipeline] = None
        self.feature_names: list[str]     = []
        self._model_path: Optional[Path]  = None
        logger.info("DelayModelTrainer başlatıldı.")

    # ──────────────────────────────────────────────────────────
    # Ana eğitim metodu
    # ──────────────────────────────────────────────────────────
    def train(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Modeli eğitir, değerlendirir ve diske kaydeder.

        Args:
            df: FeatureEngineer.build() çıktısı DataFrame.

        Returns:
            {mae, rmse, r2, cv_mae_mean, cv_mae_std, model_path,
             feature_importance_path, n_train, n_test}
        """
        try:
            logger.info("=" * 60)
            logger.info("MODEL EĞİTİMİ BAŞLIYOR")
            logger.info("=" * 60)

            # 1. Veriyi hazırla
            X, y = self._prepare_data(df)
            if X.empty or y.empty:
                raise ValueError("Eğitim verisi boş, işlem durduruldu.")

            # 2. Train / test ayır
            X_train, X_test, y_train, y_test = self._split(X, y)

            # 3. Pipeline inşa et
            self.pipeline = self._build_pipeline(X_train)

            # 4. Eğit
            logger.info("XGBoost eğitimi başlıyor... (train=%d, test=%d)",
                        len(X_train), len(X_test))
            self.pipeline.fit(X_train, y_train)
            logger.info("Eğitim tamamlandı.")

            # 5. Metrikleri hesapla
            metrics = self._evaluate(X_train, X_test, y_train, y_test)

            # 6. Cross-validation
            cv_results = self._cross_validate(X, y)
            metrics.update(cv_results)

            # 7. Kaydet
            model_path = self._save_model()
            fi_path    = self._save_feature_importance()
            self._save_metrics(metrics)

            metrics["model_path"]              = str(model_path)
            metrics["feature_importance_path"] = str(fi_path)
            metrics["n_train"]                 = len(X_train)
            metrics["n_test"]                  = len(X_test)

            self._log_summary(metrics)
            return metrics

        except Exception as exc:
            logger.error("train() hatası: %s", exc, exc_info=True)
            raise

    # ──────────────────────────────────────────────────────────
    # Veri hazırlama
    # ──────────────────────────────────────────────────────────
    def _prepare_data(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Target kolonu df'den ayırır; geçerli numeric ve categorical
        feature kolonlarını tespit eder.
        """
        try:
            if TARGET_COL not in df.columns:
                raise KeyError(
                    f"Target kolon '{TARGET_COL}' DataFrame'de bulunamadı. "
                    f"Mevcut kolonlar: {df.columns.tolist()}"
                )

            # Target'ı çıkar ve null satırları at
            valid_mask = df[TARGET_COL].notna()
            df_clean   = df[valid_mask].copy()
            y          = pd.to_numeric(df_clean[TARGET_COL], errors="coerce").dropna()
            df_clean   = df_clean.loc[y.index]

            dropped = len(df) - len(df_clean)
            if dropped > 0:
                logger.warning(
                    "Target null olan %d satır düşürüldü. Kalan: %d", dropped, len(df_clean)
                )

            # Geçerli numeric feature'ları seç
            num_cols = [
                c for c in NUMERIC_CANDIDATES
                if c in df_clean.columns and c != TARGET_COL
                and pd.api.types.is_numeric_dtype(df_clean[c])
            ]
            # Geçerli categorical feature'ları seç
            cat_cols = [
                c for c in CATEGORICAL_CANDIDATES
                if c in df_clean.columns
            ]

            # Duplikasyon kontrolü (aynı kolon her iki listede olmasın)
            cat_cols = [c for c in cat_cols if c not in num_cols]

            self.feature_names = num_cols + cat_cols
            logger.info(
                "Feature hazırlama: %d numeric + %d categorical = %d toplam",
                len(num_cols), len(cat_cols), len(self.feature_names),
            )
            logger.debug("Numeric   : %s", num_cols)
            logger.debug("Categorical: %s", cat_cols)

            X = df_clean[self.feature_names]
            return X, y

        except Exception as exc:
            logger.error("_prepare_data hatası: %s", exc)
            raise

    # ──────────────────────────────────────────────────────────
    # Train / test ayırma
    # ──────────────────────────────────────────────────────────
    def _split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Stratejik train/test split (random_state config'den alınır)."""
        try:
            test_size   = getattr(
                getattr(self.cfg, "preprocessing", None), "test_size", 0.2
            )
            random_seed = getattr(
                getattr(self.cfg, "project", None), "random_seed", 42
            )
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_seed,
            )
            logger.info(
                "Train/test split: %d/%d (test_size=%.0f%%)",
                len(X_train), len(X_test), test_size * 100,
            )
            return X_train, X_test, y_train, y_test
        except Exception as exc:
            logger.error("_split hatası: %s", exc)
            raise

    # ──────────────────────────────────────────────────────────
    # Pipeline inşası
    # ──────────────────────────────────────────────────────────
    def _build_pipeline(self, X_ref: pd.DataFrame) -> Pipeline:
        """
        ColumnTransformer + XGBRegressor Pipeline'ı oluşturur.
        X_ref ile hangi kolonların numeric/categorical olduğunu teyit eder.
        """
        try:
            from sklearn.impute import SimpleImputer  # lazy import

            num_cols = [
                c for c in self.feature_names
                if c in X_ref.columns and pd.api.types.is_numeric_dtype(X_ref[c])
            ]
            cat_cols = [
                c for c in self.feature_names
                if c in X_ref.columns and c not in num_cols
            ]

            # Numeric transformer: medyan impute → standart ölçekleme
            numeric_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler()),
            ])

            # Categorical transformer: sabit impute → OHE
            categorical_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])

            transformers = []
            if num_cols:
                transformers.append(("num", numeric_transformer, num_cols))
            if cat_cols:
                transformers.append(("cat", categorical_transformer, cat_cols))

            preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder="drop",
                verbose_feature_names_out=True,
            )

            # XGBoost hiperparametreleri config'den al
            xgb_cfg = getattr(
                getattr(self.cfg, "models", None), "delay_predictor", None
            )
            xgb_params = getattr(xgb_cfg, "xgboost", None) if xgb_cfg else None

            if xgb_params:
                xgb_kwargs = xgb_params.to_dict()
                xgb_kwargs.pop("n_jobs", None)  # XGB'de n_jobs → nthread
            else:
                xgb_kwargs = {}

            xgb_model = XGBRegressor(
                n_estimators=xgb_kwargs.get("n_estimators", 300),
                max_depth=xgb_kwargs.get("max_depth", 6),
                learning_rate=xgb_kwargs.get("learning_rate", 0.05),
                subsample=xgb_kwargs.get("subsample", 0.8),
                colsample_bytree=xgb_kwargs.get("colsample_bytree", 0.8),
                min_child_weight=xgb_kwargs.get("min_child_weight", 3),
                reg_alpha=xgb_kwargs.get("reg_alpha", 0.1),
                reg_lambda=xgb_kwargs.get("reg_lambda", 1.0),
                random_state=getattr(
                    getattr(self.cfg, "project", None), "random_seed", 42
                ),
                n_jobs=-1,
                verbosity=0,
                eval_metric="mae",
            )

            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model",        xgb_model),
            ])

            logger.info(
                "Pipeline inşa edildi: %d numeric + %d categorical feature",
                len(num_cols), len(cat_cols),
            )
            return pipeline

        except Exception as exc:
            logger.error("_build_pipeline hatası: %s", exc)
            raise

    # ──────────────────────────────────────────────────────────
    # Değerlendirme
    # ──────────────────────────────────────────────────────────
    def _evaluate(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> dict[str, float]:
        """Train ve test setleri için MAE, RMSE, R² hesaplar."""
        try:
            metrics: dict[str, float] = {}

            for split_name, X_s, y_s in [
                ("train", X_train, y_train),
                ("test",  X_test,  y_test),
            ]:
                preds = self.pipeline.predict(X_s)
                mae   = mean_absolute_error(y_s, preds)
                rmse  = np.sqrt(mean_squared_error(y_s, preds))
                r2    = r2_score(y_s, preds)

                metrics[f"{split_name}_mae"]  = round(float(mae), 4)
                metrics[f"{split_name}_rmse"] = round(float(rmse), 4)
                metrics[f"{split_name}_r2"]   = round(float(r2), 4)

                logger.info(
                    "[%s] MAE=%.2f dk | RMSE=%.2f dk | R²=%.4f",
                    split_name.upper(), mae, rmse, r2,
                )

            # Overfitting kontrolü
            gap = metrics["train_mae"] - metrics["test_mae"]
            if abs(gap) > 5:
                logger.warning(
                    "Potansiyel overfitting: train_mae=%.2f, test_mae=%.2f (fark=%.2f)",
                    metrics["train_mae"], metrics["test_mae"], gap,
                )

            # Alias (dışarıya test metrikleri as primary)
            metrics["mae"]  = metrics["test_mae"]
            metrics["rmse"] = metrics["test_rmse"]
            metrics["r2"]   = metrics["test_r2"]

            return metrics

        except Exception as exc:
            logger.error("_evaluate hatası: %s", exc)
            raise

    # ──────────────────────────────────────────────────────────
    # Cross-validation
    # ──────────────────────────────────────────────────────────
    def _cross_validate(
        self, X: pd.DataFrame, y: pd.Series
    ) -> dict[str, float]:
        """K-Fold CV ile MAE ortalaması ve std'yi hesaplar."""
        try:
            n_splits = getattr(
                getattr(
                    getattr(self.cfg, "models", None), "delay_predictor", None
                ),
                "cross_validation",
                None,
            )
            n_splits = getattr(n_splits, "n_splits", 5) if n_splits else 5

            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            scores = cross_val_score(
                self.pipeline, X, y,
                cv=kf,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
            )
            cv_mae_mean = round(float(-scores.mean()), 4)
            cv_mae_std  = round(float(scores.std()), 4)

            logger.info(
                "[CV %d-Fold] MAE = %.2f ± %.2f dk",
                n_splits, cv_mae_mean, cv_mae_std,
            )
            return {"cv_mae_mean": cv_mae_mean, "cv_mae_std": cv_mae_std}

        except Exception as exc:
            logger.warning("_cross_validate hatası (atlanıyor): %s", exc)
            return {"cv_mae_mean": None, "cv_mae_std": None}

    # ──────────────────────────────────────────────────────────
    # Model kaydetme
    # ──────────────────────────────────────────────────────────
    def _save_model(self) -> Path:
        """Eğitilmiş pipeline'ı joblib ile diske kaydeder."""
        try:
            model_path = self.cfg.get_model_path(MODEL_FILE)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.pipeline, model_path, compress=3)
            self._model_path = model_path
            logger.info("Model kaydedildi: %s", model_path)
            return model_path
        except Exception as exc:
            logger.error("_save_model hatası: %s", exc)
            raise

    # ──────────────────────────────────────────────────────────
    # Feature Importance
    # ──────────────────────────────────────────────────────────
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Eğitilmiş pipeline'dan feature importance değerlerini çıkarır.

        Returns:
            pd.DataFrame: [feature, importance, importance_pct] sütunları,
                          importance'a göre azalan sırada.
        """
        try:
            if self.pipeline is None:
                raise RuntimeError("Model henüz eğitilmedi. Önce train() çağır.")

            xgb_model    = self.pipeline.named_steps["model"]
            preprocessor = self.pipeline.named_steps["preprocessor"]

            importances = xgb_model.feature_importances_

            # sklearn ColumnTransformer'dan gerçek feature isimlerini al
            try:
                feature_names_out = preprocessor.get_feature_names_out()
            except Exception:
                feature_names_out = [f"feature_{i}" for i in range(len(importances))]

            fi_df = pd.DataFrame({
                "feature":    feature_names_out,
                "importance": importances,
            }).sort_values("importance", ascending=False).reset_index(drop=True)

            total = fi_df["importance"].sum()
            fi_df["importance_pct"] = (fi_df["importance"] / total * 100).round(2)

            return fi_df

        except Exception as exc:
            logger.error("get_feature_importance hatası: %s", exc)
            raise

    def _save_feature_importance(self) -> Path:
        """Feature importance DataFrame'ini CSV olarak kaydeder."""
        try:
            fi_df    = self.get_feature_importance()
            fi_path  = self.cfg.paths.outputs.reports / FI_FILE
            fi_path.parent.mkdir(parents=True, exist_ok=True)
            fi_df.to_csv(fi_path, index=False)
            logger.info(
                "Feature importance kaydedildi: %s (top-5: %s)",
                fi_path,
                fi_df["feature"].head(5).tolist(),
            )
            return fi_path
        except Exception as exc:
            logger.warning("_save_feature_importance hatası (atlanıyor): %s", exc)
            return Path("outputs/reports") / FI_FILE

    def _save_metrics(self, metrics: dict[str, Any]) -> None:
        """Eğitim metriklerini JSON olarak kaydeder."""
        try:
            metrics_path = self.cfg.paths.outputs.reports / METRICS_FILE
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, default=str)
            logger.info("Metrikler kaydedildi: %s", metrics_path)
        except Exception as exc:
            logger.warning("_save_metrics hatası (atlanıyor): %s", exc)

    # ──────────────────────────────────────────────────────────
    # Log özeti
    # ──────────────────────────────────────────────────────────
    def _log_summary(self, metrics: dict[str, Any]) -> None:
        logger.info("=" * 60)
        logger.info("EĞİTİM SONUÇLARI")
        logger.info("  Train  MAE : %.2f dk", metrics.get("train_mae", 0))
        logger.info("  Test   MAE : %.2f dk  ← birincil metrik", metrics.get("mae", 0))
        logger.info("  Test   RMSE: %.2f dk", metrics.get("rmse", 0))
        logger.info("  Test   R²  : %.4f",    metrics.get("r2", 0))
        if metrics.get("cv_mae_mean"):
            logger.info(
                "  CV MAE     : %.2f ± %.2f dk",
                metrics["cv_mae_mean"], metrics["cv_mae_std"],
            )
        logger.info("  Model      : %s", metrics.get("model_path", "?"))
        logger.info("=" * 60)


# ──────────────────────────────────────────────────────────────
# Hızlı test
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from src.data_ingestion.csv_reader import CSVReader
    from src.features.preprocessor import run_preprocessing_pipeline

    cfg      = get_config()
    dfs      = CSVReader(cfg).load_all()
    features = run_preprocessing_pipeline(dfs, cfg)

    trainer = DelayModelTrainer(cfg)
    result  = trainer.train(features)

    print("\n=== SONUÇ ===")
    for k, v in result.items():
        print(f"  {k:30s}: {v}")
