"""
frontend/streamlit_dashboard.py

Akıllı Lojistik ve Rota Optimizasyonu — Streamlit Dashboard
============================================================
Çalıştırma:
    streamlit run frontend/streamlit_dashboard.py

Gereksinim:
    FastAPI backend önce ayakta olmalı:
    uvicorn src.api.fastapi_app:app --reload --port 8000
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# ── Proje kökünü path'e ekle ────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ════════════════════════════════════════════════════════════
# Sayfa konfigürasyonu — İLK Streamlit çağrısı olmalı
# ════════════════════════════════════════════════════════════
st.set_page_config(
    layout="wide",
    page_title="Smart Logistics AI",
    page_icon="🚚",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "Akıllı Lojistik ve Rota Optimizasyonu — Hackathon 2025",
    },
)

# ════════════════════════════════════════════════════════════
# Sabitler
# ════════════════════════════════════════════════════════════
API_BASE        = "http://127.0.0.1:8001"
STOPS_CSV       = PROJECT_ROOT / "data" / "raw" / "route_stops.csv"
TIMEOUT_SEC     = 10    # /health, /predict için
TIMEOUT_LONG    = 120   # /optimize için (büyük rotalarda uzun sürebilir)

# ── Renk paleti ─────────────────────────────────────────────
COLOR_SUCCESS = "#00C853"
COLOR_WARNING = "#FF6D00"
COLOR_DANGER  = "#D50000"
COLOR_PRIMARY = "#1565C0"
COLOR_SURFACE = "#F8F9FA"

# ════════════════════════════════════════════════════════════
# Global CSS
# ════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* Genel arka plan */
.stApp { background-color: #F0F4F8; }

/* Başlık kartı */
.hero-card {
    background: linear-gradient(135deg, #1565C0 0%, #0D47A1 60%, #01579B 100%);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    color: white;
    box-shadow: 0 8px 32px rgba(21,101,192,0.3);
}
.hero-card h1 { margin: 0; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.5px; }
.hero-card p  { margin: 6px 0 0; font-size: 1.05rem; opacity: 0.88; }

/* Metrik kartları */
.metric-card {
    background: white;
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    border-left: 5px solid #1565C0;
    margin-bottom: 12px;
}
.metric-card.success { border-left-color: #00C853; }
.metric-card.warning { border-left-color: #FF6D00; }
.metric-card.danger  { border-left-color: #D50000; }

/* Tablo başlığı */
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #1A237E;
    padding: 8px 0 4px;
    border-bottom: 2px solid #E3F2FD;
    margin-bottom: 12px;
}

/* Badge */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
}
.badge-blue   { background: #E3F2FD; color: #1565C0; }
.badge-green  { background: #E8F5E9; color: #2E7D32; }
.badge-orange { background: #FFF3E0; color: #E65100; }
.badge-red    { background: #FFEBEE; color: #C62828; }

/* Optimizasyon karşılaştırma kartı */
.compare-card {
    background: white;
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 3px 14px rgba(0,0,0,0.09);
    height: 100%;
}
.compare-card.original { border-top: 4px solid #90A4AE; }
.compare-card.optimized { border-top: 4px solid #00C853; }

/* Kazanım banner */
.win-banner {
    background: linear-gradient(90deg, #00C853, #00E676);
    border-radius: 12px;
    padding: 18px 28px;
    text-align: center;
    color: white;
    font-weight: 800;
    font-size: 1.3rem;
    margin: 16px 0;
    box-shadow: 0 4px 20px rgba(0,200,83,0.35);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%   { box-shadow: 0 4px 20px rgba(0,200,83,0.35); }
    50%  { box-shadow: 0 4px 32px rgba(0,200,83,0.6);  }
    100% { box-shadow: 0 4px 20px rgba(0,200,83,0.35); }
}

/* Sidebar */
section[data-testid="stSidebar"] { background: #1A237E; }
section[data-testid="stSidebar"] * { color: white !important; }
section[data-testid="stSidebar"] .stSelectbox label { color: #90CAF9 !important; }

/* Buton */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1565C0, #0D47A1);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 14px 32px;
    font-size: 1.05rem;
    font-weight: 700;
    width: 100%;
    transition: all 0.2s;
    box-shadow: 0 4px 14px rgba(21,101,192,0.4);
}
div.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(21,101,192,0.55);
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# Yardımcı fonksiyonlar
# ════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def load_stops_csv() -> pd.DataFrame:
    """route_stops.csv'yi yükler ve önbelleğe alır (5 dk TTL)."""
    if not STOPS_CSV.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(STOPS_CSV, low_memory=False)
        dt_cols = ["planned_arrival", "actual_arrival", "time_window_open", "time_window_close"]
        for col in dt_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df
    except Exception as exc:
        st.error(f"CSV okuma hatası: {exc}")
        return pd.DataFrame()


def api_get(endpoint: str, timeout: int = TIMEOUT_SEC) -> dict | None:
    """GET isteği; bağlantı veya timeout hatasında None döner (sessizce)."""
    try:
        resp = requests.get(f"{API_BASE}{endpoint}", timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except (requests.exceptions.ConnectionError,
            requests.exceptions.Timeout):
        return None
    except Exception:
        return None


def _cached_health() -> dict | None:
    """
    Health check — önbellekleme YOK.
    Sadece başarılı yanıt döner; None ise backend çevrimdışıdır.
    Timeout 3 saniye: sidebar'ı bloklamaz.
    """
    return api_get("/health", timeout=8)


def api_post(endpoint: str, payload: dict, long_timeout: bool = False) -> dict | None:
    """POST isteği; bağlantı hatasında None döner."""
    timeout = TIMEOUT_LONG if long_timeout else TIMEOUT_SEC
    try:
        resp = requests.post(
            f"{API_BASE}{endpoint}",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return None
    except requests.exceptions.Timeout:
        st.error(
            f"⏱ Sunucu {timeout} saniyede yanıt vermedi. "
            "Backend terminalde hata var mı kontrol et."
        )
        return None
    except requests.exceptions.HTTPError as exc:
        detail = exc.response.json().get("detail", str(exc)) if exc.response else str(exc)
        st.error(f"API yanıt hatası ({exc.response.status_code}): {detail}")
        return None
    except Exception as exc:
        st.error(f"Beklenmeyen istek hatası: {exc}")
        return None


def _backend_reachable() -> bool:
    """Backend'e hızlı ping atar; ulaşılabilirse True döner."""
    try:
        requests.get(f"{API_BASE}/health", timeout=3)
        return True
    except Exception:
        return False


def _api_down_warning() -> None:
    """API bağlantısı yoksa gösterilen şık uyarı."""
    st.markdown("""
    <div style="background:#FFF3E0; border-left:5px solid #FF6D00;
                border-radius:10px; padding:18px 22px; margin:12px 0;">
        <b style="font-size:1.1rem;">🔌 Backend Bağlantısı Kurulamadı</b><br><br>
        Lütfen önce FastAPI sunucusunu başlatın:<br>
        <code style="background:#FFE0B2; padding:4px 8px; border-radius:4px; font-size:0.9rem;">
            uvicorn src.api.fastapi_app:app --reload --port 8000
        </code><br><br>
        Sunucu çalışıyorsa sayfayı yenileyin.
    </div>
    """, unsafe_allow_html=True)


def _severity_to_label(score: float) -> tuple[str, str]:
    """Weather severity index → (emoji_label, badge_class)"""
    if score < 0.2:
        return "🟢 Düşük Risk", "badge-green"
    if score < 0.4:
        return "🟡 Orta Risk", "badge-orange"
    if score < 0.6:
        return "🟠 Yüksek Risk", "badge-orange"
    return "🔴 Kritik Risk", "badge-red"


def _format_dt(val) -> str:
    """Timestamp'i HH:MM formatına çevirir."""
    try:
        return pd.Timestamp(val).strftime("%H:%M")
    except Exception:
        return str(val) if pd.notna(val) else "—"


def _stops_to_payload(stops_df: pd.DataFrame, route_id: str) -> dict | None:
    """DataFrame → POST /optimize payload sözlüğü."""
    if stops_df.empty:
        return None
    stops = []
    for _, row in stops_df.iterrows():
        stop = {
            "stop_id":               str(row.get("stop_id", f"STP-{_}")),
            "stop_sequence":         int(row.get("stop_sequence", 1)),
            "latitude":              float(row.get("latitude", 0)),
            "longitude":             float(row.get("longitude", 0)),
            "planned_service_min":   float(row.get("planned_service_min", 10)),
            "planned_travel_min":    float(row.get("planned_travel_min", 20)),
            "distance_from_prev_km": float(row.get("distance_from_prev_km", 0)),
            "delay_probability":     float(row.get("delay_probability", 0.2)),
            "package_count":         int(row.get("package_count", 5)),
            "package_weight_kg":     float(row.get("package_weight_kg", 10)),
            "road_type":             str(row.get("road_type", "highway")).lower(),
        }
        # datetime kolonlar
        for dt_col in ["planned_arrival", "time_window_open", "time_window_close"]:
            val = row.get(dt_col)
            if pd.notna(val) and val is not None:
                try:
                    stop[dt_col] = pd.Timestamp(val).isoformat()
                except Exception:
                    stop[dt_col] = None
        stops.append(stop)

    return {
        "route_id":       route_id,
        "departure_time": stops[0].get("planned_arrival"),
        "stops":          stops,
    }


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════

def render_sidebar(stops_all: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Sidebar'ı render eder. (seçili route_id, o rotanın stop df'i) döner."""
    with st.sidebar:
        st.markdown("## 🚚 Smart Logistics AI")
        st.markdown("---")

        # API sağlık durumu — önbellekli, 3s timeout, hata göstermez
        health = _cached_health()
        if health:
            model_ok = health.get("model_loaded", False)
            w_status = health.get("weather_api", "?")
            st.markdown(f"**Sistem:** {'🟢 Çevrimiçi' if model_ok else '🟡 Model Eksik'}")
            st.markdown(f"**Model:** {'✅ Yüklü' if model_ok else '❌ Yüklenmedi'}")
            st.markdown(f"**Hava API:** {'🌐 Canlı' if w_status == 'live' else '📦 Mock'}")
            st.markdown(f"**Uptime:** {health.get('uptime_sec', 0):.0f} sn")
        else:
            st.markdown("**Sistem:** 🔴 Çevrimdışı")
            st.caption("Backend'i başlatın:\n`python -m uvicorn src.api.fastapi_app:app --reload --port 8000`")

        st.markdown("---")
        st.markdown("### 📋 Rota Seçimi")

        if stops_all.empty or "route_id" not in stops_all.columns:
            st.warning("CSV yüklenemedi veya 'route_id' kolonu yok.")
            return "", pd.DataFrame()

        route_ids = sorted(stops_all["route_id"].unique().tolist())
        selected  = st.selectbox(
            "Test edilecek rotayı seç:",
            options=route_ids,
            help="Dropdown'dan bir rota seçin; detaylar sağda görünür.",
        )

        route_df = stops_all[stops_all["route_id"] == selected].copy()
        route_df = route_df.sort_values("stop_sequence").reset_index(drop=True)

        st.markdown("---")
        st.markdown("### 📊 Rota Özeti")
        st.markdown(f"**Durak sayısı:** {len(route_df)}")
        if "total_distance_km" in stops_all.columns:
            pass
        if "planned_travel_min" in route_df.columns:
            total_planned = route_df["planned_travel_min"].sum()
            st.markdown(f"**Toplam seyahat:** {total_planned:.0f} dk")
        if "package_count" in route_df.columns:
            st.markdown(f"**Toplam paket:** {route_df['package_count'].sum():.0f}")
        if "delay_probability" in route_df.columns:
            avg_prob = route_df["delay_probability"].mean()
            st.markdown(f"**Ort. gecikme olasılığı:** %{avg_prob*100:.1f}")

        st.markdown("---")
        st.caption("Hackathon 2025 · Smart Logistics AI")

    return selected, route_df


# ════════════════════════════════════════════════════════════
# ANA İÇERİK
# ════════════════════════════════════════════════════════════

def render_hero() -> None:
    st.markdown("""
    <div class="hero-card">
        <h1>🚚 Akıllı Lojistik & Rota Optimizasyonu</h1>
        <p>XGBoost + Uzaysal-Zamansal Analiz + Dinamik Rota Motoru ile güçlendirilmiş
           gerçek zamanlı teslimat optimizasyonu</p>
    </div>
    """, unsafe_allow_html=True)


def render_original_table(route_df: pd.DataFrame, route_id: str) -> None:
    """Seçilen rotanın orijinal durak tablosunu gösterir."""
    st.markdown(
        f'<div class="section-title">📍 Orijinal Durak Sırası — {route_id}</div>',
        unsafe_allow_html=True,
    )

    display_cols = []
    col_map = {
        "stop_sequence":         "Sıra",
        "stop_id":               "Durak ID",
        "latitude":              "Enlem",
        "longitude":             "Boylam",
        "planned_arrival":       "Planlanan Varış",
        "time_window_open":      "TW Açılış",
        "time_window_close":     "TW Kapanış",
        "planned_travel_min":    "Seyahat (dk)",
        "planned_service_min":   "Servis (dk)",
        "delay_probability":     "Gecikme Olas.",
        "road_type":             "Yol Tipi",
        "package_count":         "Paket",
    }
    available = {k: v for k, v in col_map.items() if k in route_df.columns}
    display_df = route_df[list(available.keys())].rename(columns=available).copy()

    # Datetime formatlama
    for col in ["Planlanan Varış", "TW Açılış", "TW Kapanış"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(_format_dt)

    # Gecikme olasılığı → yüzde
    if "Gecikme Olas." in display_df.columns:
        display_df["Gecikme Olas."] = display_df["Gecikme Olas."].apply(
            lambda x: f"%{x*100:.0f}" if pd.notna(x) else "—"
        )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Sıra":   st.column_config.NumberColumn(format="%d", width="small"),
            "Enlem":  st.column_config.NumberColumn(format="%.5f"),
            "Boylam": st.column_config.NumberColumn(format="%.5f"),
        },
    )


def render_map(route_df: pd.DataFrame, optimized_ids: list[str] | None = None) -> None:
    """Folium haritasında rotayı gösterir."""
    try:
        import folium
        from streamlit_folium import st_folium

        if "latitude" not in route_df.columns:
            return

        center_lat = route_df["latitude"].mean()
        center_lon = route_df["longitude"].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10,
                       tiles="CartoDB positron")

        # Sıraya göre renkler
        n = len(route_df)
        orig_sorted = route_df.sort_values("stop_sequence").reset_index(drop=True)

        # Orijinal rota — gri çizgi
        orig_coords = [[r["latitude"], r["longitude"]] for _, r in orig_sorted.iterrows()]
        if len(orig_coords) > 1:
            folium.PolyLine(orig_coords, color="#90A4AE", weight=3,
                            opacity=0.6, tooltip="Orijinal Rota").add_to(m)

        # Optimize edilmiş sıra varsa — mavi çizgi
        if optimized_ids:
            id_to_coord = {
                str(r["stop_id"]): [r["latitude"], r["longitude"]]
                for _, r in orig_sorted.iterrows()
                if "stop_id" in orig_sorted.columns
            }
            opt_coords = [id_to_coord[sid] for sid in optimized_ids if sid in id_to_coord]
            if len(opt_coords) > 1:
                folium.PolyLine(opt_coords, color="#1565C0", weight=4,
                                opacity=0.9, tooltip="Optimize Edilmiş Rota").add_to(m)

        # Durak marker'ları
        for i, (_, row) in enumerate(orig_sorted.iterrows()):
            is_anchor = (i == 0)
            color  = "red" if is_anchor else "blue"
            icon   = "home" if is_anchor else "info-sign"
            label  = f"⚓ {row.get('stop_id','?')} (Anchor)" if is_anchor \
                     else f"{i+1}. {row.get('stop_id','?')}"
            folium.Marker(
                [row["latitude"], row["longitude"]],
                popup=folium.Popup(label, max_width=200),
                tooltip=label,
                icon=folium.Icon(color=color, icon=icon, prefix="glyphicon"),
            ).add_to(m)

        st_folium(m, use_container_width=True, height=380)

    except ImportError:
        st.info("Harita için: `pip install folium streamlit-folium`")
    except Exception as exc:
        st.warning(f"Harita render hatası: {exc}")


def render_risk_analysis(route_df: pd.DataFrame) -> None:
    """'Canlı Risk Analizi Yap' butonunu ve sonucunu gösterir."""
    if route_df.empty:
        return

    first_stop = route_df.iloc[0]
    lat = float(first_stop.get("latitude", 39.62))
    lon = float(first_stop.get("longitude", 37.05))

    if st.button("🔍 Canlı Risk Analizi Yap", use_container_width=True):
        with st.spinner("Anlık hava durumu ve risk skoru çekiliyor..."):
            time.sleep(0.3)  # UX: progress hissi
            payload = {
                "latitude":              lat,
                "longitude":             lon,
                "planned_arrival":       datetime.now().isoformat(),
                "road_type":             str(first_stop.get("road_type", "highway")).lower(),
                "planned_travel_min":    float(first_stop.get("planned_travel_min", 20)),
                "planned_service_min":   float(first_stop.get("planned_service_min", 10)),
                "package_count":         int(first_stop.get("package_count", 5)),
                "package_weight_kg":     float(first_stop.get("package_weight_kg", 10)),
                "distance_from_prev_km": float(first_stop.get("distance_from_prev_km", 10)),
                "delay_probability":     float(first_stop.get("delay_probability", 0.2)),
            }
            result = api_post("/predict", payload)

        if result is None:
            _api_down_warning()
            return

        weather    = result.get("weather", {})
        delay_min  = result.get("predicted_delay_min", 0)
        severity   = weather.get("weather_severity_index", 0)
        label, badge_cls = _severity_to_label(severity)
        is_mock    = weather.get("is_mock", True)

        # Risk kartı
        risk_color  = "#FFEBEE" if severity > 0.4 else "#E8F5E9"
        risk_border = COLOR_DANGER if severity > 0.4 else COLOR_SUCCESS

        st.markdown(f"""
        <div style="background:{risk_color}; border-left:5px solid {risk_border};
                    border-radius:12px; padding:20px 24px; margin:12px 0;">
            <b style="font-size:1.1rem;">
                {'⚠️' if severity > 0.4 else '✅'} Canlı Risk Analizi Sonucu
                {'<span style="font-size:0.75rem; background:#E0E0E0; padding:2px 8px; '
                 'border-radius:10px; margin-left:8px;">MOCK</span>' if is_mock else ''}
            </b><br><br>
            <b>📍 Konum:</b> {lat:.4f}, {lon:.4f} &nbsp;|&nbsp;
            <b>🌡 Sıcaklık:</b> {weather.get('temperature_c', '?')}°C &nbsp;|&nbsp;
            <b>🌧 Yağış:</b> {weather.get('precipitation_mm', 0):.1f} mm<br>
            <b>💨 Rüzgar:</b> {weather.get('wind_speed_kmh', '?')} km/h &nbsp;|&nbsp;
            <b>👁 Görüş:</b> {weather.get('visibility_km', '?')} km &nbsp;|&nbsp;
            <b>🛣 Yüzey:</b> {weather.get('road_surface_condition', '?').upper()}<br><br>
            <b>Risk Seviyesi:</b> {label} &nbsp; (Skor: {severity:.2f})<br>
            <b>⏱ Tahmin Edilen Gecikme:</b>
            <span style="font-size:1.2rem; font-weight:800;">
                {delay_min:.1f} dakika
            </span>
        </div>
        """, unsafe_allow_html=True)

        if severity > 0.5:
            st.warning(
                f"⚠️ Canlı Hava: **{weather.get('weather_condition','?').upper()}** — "
                f"Görüş {weather.get('visibility_km','?')} km. "
                f"Gecikme Riski **YÜKSEK**! Sürücüleri uyar."
            )
        elif severity > 0.3:
            st.info(
                f"ℹ️ Hava koşulları **orta** risk oluşturuyor. "
                f"Güzergah dikkatli takip edilmeli."
            )
        else:
            st.success("✅ Hava koşulları uygun. Rota nominal şartlarda ilerleyebilir.")


def render_optimization_result(route_df: pd.DataFrame, route_id: str) -> None:
    """'Rotayı Optimize Et' butonu ve iki kolonlu karşılaştırma sonucu."""
    st.markdown("<br>", unsafe_allow_html=True)
    optimize_clicked = st.button(
        "🚀 Rotayı Optimize Et",
        type="primary",
        use_container_width=True,
        help="AI motoru tüm permutasyonları değerlendirerek en hızlı rotayı bulur.",
    )

    if not optimize_clicked:
        return

    # Payload hazırla
    payload = _stops_to_payload(route_df, route_id)
    if payload is None:
        st.error("Payload oluşturulamadı.")
        return

    # API isteği
    with st.spinner("🧠 AI motoru rota permutasyonlarını analiz ediyor..."):
        progress = st.progress(0, text="Mesafe matrisi hesaplanıyor...")
        time.sleep(0.4); progress.progress(25, text="Tortuosity faktörü belirleniyor...")
        time.sleep(0.3); progress.progress(50, text="ML gecikme tahminleri çalışıyor...")

        result = api_post("/optimize", payload, long_timeout=True)

        progress.progress(85, text="En iyi sıralama seçiliyor...")
        time.sleep(0.3); progress.progress(100, text="Tamamlandı!")
        time.sleep(0.3); progress.empty()

    if result is None:
        # Hata mesajı zaten api_post içinde gösterildi (timeout / http error)
        # Sadece bağlantı hiç kurulamadıysa backend uyarısı göster
        if not _backend_reachable():
            _api_down_warning()
        return

    # ── Kazanım banner ──────────────────────────────────────
    time_saved    = result.get("time_saved_min", 0)
    tw_reduced    = result.get("missed_windows_reduced", 0)
    improvement   = result.get("improvement_pct", 0)
    algorithm     = result.get("algorithm_used", "?")
    tortuosity    = result.get("tortuosity_factor", 1.0)
    elapsed       = result.get("elapsed_ms", 0)

    if time_saved > 0 or tw_reduced > 0:
        st.markdown(f"""
        <div class="win-banner">
            🎯 OPTİMİZASYON TAMAMLANDI — {time_saved:.1f} DAKİKA KAZANILDI!
            {'| ' + str(tw_reduced) + ' TESLİMAT KURTARILDI 🚀' if tw_reduced > 0 else ''}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success("✅ Optimizasyon tamamlandı. Mevcut sıra zaten en optimal!")

    # ── Metrik satırı ───────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            "⏱ Kazanılan Süre",
            f"{time_saved:.1f} dk",
            delta=f"-%{improvement:.1f}" if improvement > 0 else "0%",
            delta_color="normal",
        )
    with m2:
        st.metric(
            "🎯 Kurtarılan Teslimat",
            f"{tw_reduced} pencere",
            delta=(f"+{tw_reduced} iyileşme" if tw_reduced > 0 else "Değişim yok"),
            delta_color="normal" if tw_reduced > 0 else "off",
        )
    with m3:
        st.metric(
            "🌀 Tortuosity Faktörü",
            f"{tortuosity:.3f}",
            help="Rotanın kıvrımlılık katsayısı. 1.0 = düz yol.",
        )
    with m4:
        st.metric(
            "⚡ Hesaplama Süresi",
            f"{elapsed:.0f} ms",
            delta=f"Algoritma: {algorithm.replace('_', ' ').title()}",
            delta_color="off",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── İki kolonlu karşılaştırma ───────────────────────────
    orig_seq   = result.get("original_sequence", [])
    opt_ids    = result.get("optimized_stop_ids", [])
    opt_seq    = result.get("optimized_sequence", [])
    stop_details = result.get("stop_details", [])

    orig_min  = result.get("original_total_min", 0)
    opt_min   = result.get("optimized_total_min", 0)
    orig_tw   = result.get("original_missed_windows", 0)
    opt_tw    = result.get("optimized_missed_windows", 0)
    orig_delay = result.get("original_predicted_delay", 0)
    opt_delay  = result.get("optimized_predicted_delay", 0)

    col_left, col_right = st.columns(2, gap="large")

    # ── Sol: Orijinal ───────────────────────────────────────
    with col_left:
        st.markdown("""
        <div class="compare-card original">
            <b style="font-size:1.05rem; color:#546E7A;">
                ⬜ Orijinal Sıralama
            </b>
        </div>
        """, unsafe_allow_html=True)

        orig_rows = []
        for seq in orig_seq:
            row = route_df[route_df["stop_sequence"] == seq]
            if not row.empty:
                r = row.iloc[0]
                orig_rows.append({
                    "Sıra":      int(seq),
                    "Durak ID":  str(r.get("stop_id", "?")),
                    "Varış":     _format_dt(r.get("planned_arrival")),
                    "Yol Tipi":  str(r.get("road_type", "?")).title(),
                    "Paket":     int(r.get("package_count", 0)),
                })

        if orig_rows:
            st.dataframe(
                pd.DataFrame(orig_rows),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown(f"""
        <div style="background:#ECEFF1; border-radius:8px; padding:12px 16px; margin-top:8px;">
            📏 Toplam Süre: <b>{orig_min:.1f} dk</b><br>
            ⚠️ Kaçırılan TW: <b style="color:#C62828;">{orig_tw}</b><br>
            ⏱ Tahmini Gecikme: <b>{orig_delay:.1f} dk</b>
        </div>
        """, unsafe_allow_html=True)

    # ── Sağ: Optimize ───────────────────────────────────────
    with col_right:
        st.markdown("""
        <div class="compare-card optimized">
            <b style="font-size:1.05rem; color:#2E7D32;">
                ✅ AI Optimize Edilmiş Sıralama
            </b>
        </div>
        """, unsafe_allow_html=True)

        opt_rows = []
        for detail in stop_details:
            orig_s = detail.get("sequence_original", "?")
            changed = (detail.get("position") != orig_s)
            opt_rows.append({
                "Yeni Sıra":      detail.get("position", "?"),
                "Durak ID":       detail.get("stop_id", "?"),
                "Org. Sıra":      orig_s,
                "Değişim":        "🔄" if changed else "—",
            })

        if opt_rows:
            st.dataframe(
                pd.DataFrame(opt_rows),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Değişim": st.column_config.TextColumn(width="small"),
                },
            )

        tw_color = "#1B5E20" if opt_tw < orig_tw else "#C62828"
        st.markdown(f"""
        <div style="background:#E8F5E9; border-radius:8px; padding:12px 16px;
                    margin-top:8px; border:1px solid #A5D6A7;">
            📏 Toplam Süre: <b>{opt_min:.1f} dk</b>
            <span style="color:{COLOR_SUCCESS}; font-weight:700;">
                {f"(−{time_saved:.1f} dk)" if time_saved > 0 else ""}
            </span><br>
            ✅ Kaçırılan TW: <b style="color:{tw_color};">{opt_tw}</b><br>
            ⏱ Tahmini Gecikme: <b>{opt_delay:.1f} dk</b>
            {f'<span style="color:{COLOR_SUCCESS};"> (−{orig_delay-opt_delay:.1f} dk)</span>'
             if opt_delay < orig_delay else ''}
        </div>
        """, unsafe_allow_html=True)

    # ── Harita (optimize edilmiş rota) ──────────────────────
    if opt_ids:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">🗺️ Rota Haritası — Optimize Edilmiş Güzergah</div>',
            unsafe_allow_html=True,
        )
        render_map(route_df, optimized_ids=opt_ids)


# ════════════════════════════════════════════════════════════
# ANA UYGULAMA
# ════════════════════════════════════════════════════════════

def main() -> None:
    # Veri yükle
    with st.spinner("Veri yükleniyor..."):
        stops_all = load_stops_csv()

    # Hero
    render_hero()

    # Sidebar + rota seçimi
    selected_route, route_df = render_sidebar(stops_all)

    if stops_all.empty:
        st.error(
            f"CSV dosyası bulunamadı veya boş: `{STOPS_CSV}`\n\n"
            "CSV'leri `data/raw/` klasörüne kopyaladığından emin ol, "
            "ardından sayfayı yenile."
        )
        return

    if route_df.empty:
        st.warning("Seçilen rotaya ait durak bulunamadı.")
        return

    # ── Üst bölüm: Tablo + Harita yan yana ─────────────────
    tab_table, tab_map = st.tabs(["📋 Durak Listesi", "🗺️ Harita"])

    with tab_table:
        render_original_table(route_df, selected_route)

    with tab_map:
        render_map(route_df)

    st.markdown("---")

    # ── Eylem bölümü ────────────────────────────────────────
    st.markdown(
        '<div class="section-title">⚡ AI Eylem Merkezi</div>',
        unsafe_allow_html=True,
    )
    btn_col1, btn_col2 = st.columns([1, 1], gap="medium")

    with btn_col1:
        render_risk_analysis(route_df)

    with btn_col2:
        # Buton burada; sonuç aşağıda tam genişlikte
        pass

    # Optimize butonu + sonuç tam genişlikte
    render_optimization_result(route_df, selected_route)


# ════════════════════════════════════════════════════════════
# Giriş noktası
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
