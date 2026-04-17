# Proje Handoff — Smart Logistics Hackathon

## ÖNEMLİ: Bu Dosyayı Okuduktan Sonra Ne Yapacaksın

1. Bu dosyayı baştan sona oku
2. Projeyi kendi başına keşfet: `src/`, `frontend/`, `config/`, `data/raw/`, `outputs/` dizinlerini oku
3. Blueprint dosyalarını oku: `01_PROJECT_MASTER_BLUEPRINT.md`, `02_DATA_MODEL_AND_PREPROCESSING.md`, `03_ML_AND_OPTIMIZATION_ENGINE.md`, `04_MVP_AND_API_INTEGRATION.md`
4. Kullanıcıdan sonraki adımı sor

---

## Yarışma Bilgisi (Kullanıcının İlk Mesajından)

**Yarışma adı:** Smart Logistics  
**Amaç:** Gerçek zamanlı teslimat rotası optimizasyonu  

### Yarışmanın İstediği Ürün
> "SMART LOGISTICS is a service that optimizes delivery routes in real time, taking into account traffic, weather, delays, and customer time windows. Its goal is to help logistics companies deliver faster, more reliably, and at lower operational costs by providing couriers and dispatchers with adaptive, continuously updated routes."

**Beklenen özellikler:**
- Rota boyunca gecikme olasılığını tahmin eder
- Optimize edilmiş durak sırası önerir
- Gerçek zamanlı koşullara göre kalkış saati ve rotayı yeniden hesaplar
- Dispatcher-friendly formatta öneri sunar

**Kısıtlar:**
- Öneriler somut ve pratik olmalı, aşırı teknik detay olmamalı
- Rota mantığı yorumlanabilir olmalı: neden bir durak değişti veya kalkış saati ayarlandı açık olmalı
- Görseller kompakt ve hızlı okunabilir olmalı (örn. "Delay in 12 minutes — we recommend reordering stop #3")

**Değerlendirme kriterleri:**
1. Gecikme tahmin doğruluğu (Effectiveness & task alignment)
2. Optimizasyon önerilerinin kalitesi (Effectiveness & task alignment)
3. Dispatcher arayüz netliği (UX & interface)
4. AI'nın gerçekten işe yarıyor olması (Use of ML/AI)
5. MVP kararlılığı ve kod kalitesi (Code quality & architecture)

**Teslim edilecekler:**
- Repository erişimi
- Proje sunumu linki
- Çalışan MVP / tıklanabilir prototip linki

### Yarışma Organizatörü Onayı (WhatsApp)
Soru: "CSV'de A→C gibi doğrudan rota yoksa, bilinen rotalardan `actual/haversine` oranı hesaplayıp yeni segment için uygulamak geçerli mi?"  
Cevap: **"Yes, your approach is absolutely fine and can be used."**

---

## Proje Mimarisi (Blueprint Özeti)

```
/hackathon-smart-logistic
  ├── data/raw/               5 CSV dosyası (routes, route_stops, traffic_segments, weather_observations, historical_delay_stats)
  ├── outputs/models/         xgboost_delay_model.joblib  ← güncel eğitilmiş model
  ├── outputs/reports/        feature_importance.csv, train_metrics.json
  ├── config/                 settings.yaml, config_loader.py
  ├── src/
  │   ├── data_ingestion/     csv_reader.py, api_client.py (OpenWeather + fallback)
  │   ├── features/           preprocessor.py (SpatialTemporalJoiner + FeatureEngineer)
  │   ├── ml/                 trainer.py, predictor.py
  │   ├── optimization/       route_engine.py  ← ana motor
  │   └── api/                fastapi_app.py
  ├── frontend/               streamlit_dashboard.py
  └── main_pipeline.py        eğitim pipeline'ı
```

**Tech stack:** FastAPI (port 8000) + Streamlit (port 8501) + XGBoost + sklearn Pipeline + Haversine

**İki aşamalı strateji:**
- Phase 1: XGBoost, sağlanan CSV dataset ile eğitildi (Sivas, Türkiye, 2025)
- Phase 2: Streamlit MVP'de canlı OpenWeather API + Haversine ile gerçek zamanlı simülasyon

---

## Bu Konuşmada Yapılan Değişiklikler

### ✅ 1. Port Hatası Düzeltildi
`frontend/streamlit_dashboard.py:46` → `API_BASE = "http://127.0.0.1:8001"` → `8000`

### ✅ 2. Optimizer — Tüm Duraklar Dinamik (Mimari İyileştirme)
`src/optimization/route_engine.py`
- Eski: `anchor_idx=0` hardcoded → ilk durak her zaman sabit
- Yeni: `fixed_positions: Optional[frozenset[int]] = None` parametresi → varsayılan tüm duraklar mobil, opsiyonel kilitleme destekleniyor
- `_exact_permutation(all_idxs, fixed_pos, obj)` ve `_two_opt(initial_seq, fixed_pos, obj)` güncellendi
- `src/api/fastapi_app.py` `OptimizeRequest`'e `fixed_positions: Optional[list[int]]` eklendi

### ✅ 3. Data Leakage Temizlendi → Model Yeniden Eğitildi
**Sorun:** Eski modelde bu feature'lar cevabı sızdırıyordu (train R²=1.0, ezber):
- `delay_ratio` = actual_travel_min / planned_travel_min (gerçek veri, inference'ta bilinmez)
- `cumulative_delay_at_stop` = delay_at_stop_min cumsum (hedef değişkenin kendisi)
- `overall_delay_factor` = routes.csv'den actual_duration/planned_duration (gerçek veri)

**Düzeltmeler:**
- `src/features/preprocessor.py`: `_delay_features()` sadeleştirildi, `overall_delay_factor` join'den çıkarıldı
- `src/ml/trainer.py`: 3 leaky feature NUMERIC_CANDIDATES'ten silindi, `n_jobs=1` (XGBoost 3.2 + Python 3.14 uyum fix)
- `main_pipeline.py` stale validation listeleri güncellendi

**Yeni model (outputs/models/xgboost_delay_model.joblib):**
- Test MAE: 4.04 dk | Test R²: 0.9655
- Top features: delay_probability %18.6, incident_severity %16.9, planned_travel_min %12.3, traffic_level_congested %12.1

### ✅ 4. Optimizer Batch Prediction
`ObjectiveFunction._estimate_delays()`: N ayrı `predict()` → 1 `predict_batch()` + `_feature_cols` init'te önhesaplanıyor

### ✅ 5. Küçük Kalite/Hız Düzeltmeleri
- `mobile_count`: O(N) → `len(stop_points) - len(fixed_pos)`
- `_two_opt`: `if fixed_pos and any(...)` guard (boş frozenset'te gereksiz O(N²) döngü önlendi)
- Variable shadowing: fallback'te `mat_idx` → `stop_idx`

---

## Mevcut Açık Sorunlar (Sıradaki Adımlar)

### 🔴 1. Optimizer Hâlâ Çok Yavaş
Log'dan: 5 durak → 60+ saniye, 4 durak → 3 saniye.
5! = 120 permütasyon ama her batch prediction + DataFrame build pahalı.
Çözüm seçenekleri: `MAX_EXACT_PERM_STOPS` eşiğini 10→6 düşür, greedy init ekle, ya da permütasyon timeout.

### 🔴 2. Tortuosity Factor Yanlış
Log'da: `Tortuosity Factor: 1.0000` / `1.0001` — olması gereken ~1.3-1.35.
`TortuosityCalculator.compute()` metodunda bug var, `route_engine.py`'de incelenmeli.

### 🟡 3. A-to-C Mesafe İmputation Eksik
Yarışma onayladı. CSV'de olmayan segment kombinasyonları için `haversine × (actual_km/haversine_km)` oranı uygulanmalı. Şu an eksik mesafeler median ile dolduruluyor.

### 🟡 4. OpenWeather API Key
`config/settings.yaml`'da `"YOUR_OPENWEATHER_API_KEY"` placeholder. Mock data kullanılıyor.

### 🟡 5. Dispatcher UI — "Neden?" Açıklaması
Yarışma kriteri: "route logic must remain interpretable". Feature importance ile her durak değişikliğinin sebebi gösterilmeli (örn. "Stop #3 öne alındı: trafik yoğunluğu azalıyor, +15 dk kazanç").

---

## Proje Nasıl Çalıştırılır

```bash
# Terminal 1
cd c:\Users\Eliaçık\hackathon-smart-logistic
uvicorn src.api.fastapi_app:app --reload --port 8000

# Terminal 2
streamlit run frontend/streamlit_dashboard.py

# Model yeniden eğitmek
python main_pipeline.py
```

---

## Senin Görevin

Yukarıdakileri anladıktan sonra projeyi kendin de incele. Sonra kullanıcıya hangi konuya devam etmek istediğini sor — büyük ihtimalle optimizer hızı veya tortuosity bug'ı.
