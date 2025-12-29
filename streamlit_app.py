import streamlit as st
from streamlit_folium import st_folium
import folium
import pandas as pd
import joblib
import requests
import io
from datetime import datetime, timedelta

# ==================== CONFIGURAÇÃO DA PÁGINA ====================
st.set_page_config(
    page_title="EarthQuake AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🌍 EarthQuake AI")
st.markdown("### Previsão Estatística de Risco Sísmico por Localização")
st.markdown("_Clique no mapa para estimar a magnitude média histórica de terremotos em qualquer lugar do mundo_")
st.caption("Modelo treinado com **3.4 milhões** de eventos USGS (1990–2025) · HistGradientBoostingRegressor")

# ==================== CARREGA APENAS O MODELO DE MAGNITUDE ====================
@st.cache_resource
def load_magnitude_model():
    return joblib.load('model_magnitude_predictor.pkl')

model_mag = load_magnitude_model()

# ==================== MAPA INTERATIVO (SATÉLITE LINDO) ====================
st.header("🗺️ Clique no mapa para analisar o risco sísmico")

m = folium.Map(
    location=[0, 0],
    zoom_start=2,
    tiles="Esri WorldImagery",
    attr="Esri"
)

folium.TileLayer(
    tiles="OpenStreetMap",
    name="Ruas (claro)",
    show=False
).add_to(m)

folium.LayerControl().add_to(m)

map_data = st_folium(m, width=1200, height=500, key="main_map")

# ==================== LOCALIZAÇÃO SELECIONADA ====================
if map_data and map_data.get("last_clicked"):
    lat = round(map_data["last_clicked"]["lat"], 4)
    lon = round(map_data["last_clicked"]["lng"], 4)
    location_name = "Localização clicada"
else:
    lat = -23.55
    lon = -46.63
    location_name = "São Paulo, Brasil (padrão – clique no mapa para mudar)"

st.subheader(f"📍 {location_name}: {lat}, {lon}")

col1, col2, col3 = st.columns(3)
col1.metric("Latitude", lat)
col2.metric("Longitude", lon)
col3.metric("Profundidade Padrão", "10 km")

# ==================== PREDIÇÃO DE MAGNITUDE ====================
input_data = pd.DataFrame([{
    'latitude': lat,
    'longitude': lon,
    'depth': 10.0,
    'year': 2025,
    'month': 12,
    'day': 29,
    'hour': 12
}])

pred_mag = float(model_mag.predict(input_data)[0])

st.metric("**Magnitude Média Histórica Estimada**", f"{pred_mag:.2f}")

# Classificação de risco
if pred_mag < 4.0:
    risco = "🟢 Baixo"
    explicacao = "Região com poucos terremotos ou apenas micro-sismos. Risco muito baixo."
elif pred_mag < 4.8:
    risco = "🟡 Moderado"
    explicacao = "Terremotos sentidos ocasionalmente. Danos muito raros."
elif pred_mag < 5.5:
    risco = "🟠 Médio-Alto"
    explicacao = "Possibilidade de terremotos moderados. Atenção em construções antigas."
else:
    risco = "🔴 Alto"
    explicacao = "Região tectonicamente ativa. Histórico de terremotos fortes."

st.markdown(f"**Nível de Risco Relativo:** {risco}")
st.info(explicacao)

st.warning("⚠️ **Importante**: Esta é uma estimativa estatística baseada em padrões históricos. Grandes terremotos (M>7) são raros e **não podem ser previstos com precisão**.")

# ==================== ESCALA DE MAGNITUDE ====================
st.header("📊 Escala de Magnitude – O que significa?")
st.markdown("""
| Magnitude | Efeito típico                          | Frequência     |
|----------|----------------------------------------|----------------|
| < 4.0    | Geralmente não sentido                 | Muito comum    |
| 4.0–4.9  | Sentido, sem danos                     | Comum          |
| 5.0–5.9  | Danos leves a moderados                | Moderado       |
| 6.0–6.9  | Danos significativos                   | Raro           |
| ≥ 7.0    | Graves a catastróficos                 | Muito raro     |
""")

# ==================== TENDÊNCIA HISTÓRICA (GRÁFICO ESTÁTICO) ====================
st.header("📈 Tendência Histórica de Atividade Sísmica – Américas")

st.image(
    'forecast_americas.png',
    caption="Histórico recente + projeção simples baseada em média móvel e tendência linear (últimos 10 anos)",
    use_container_width=True  # <-- CORRIGIDO: era use_column_width
)

st.info(
    "O aumento gradual no número de eventos registrados reflete principalmente **melhorias na rede de detecção sísmica global** "
    "ao longo dos anos, e não necessariamente um aumento real na atividade tectônica."
)

# ==================== ALERTAS EM TEMPO REAL (AUTO-UPDATE A CADA 1 MINUTO) ====================
st.header("🚨 Alertas Globais – Terremotos M ≥ 6.0 (Últimos 30 Dias)")

alert_placeholder = st.empty()
status_placeholder = st.empty()

with alert_placeholder.container():
    status_placeholder.info("🔄 Carregando dados em tempo real da USGS...")

@st.cache_data(ttl=60)
def load_earthquake_alerts():
    try:
        end_time = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
        start_time = (datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m-%dT%H:%M:%S')

        url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        params = {
            'format': 'csv',
            'starttime': start_time,
            'endtime': end_time,
            'minmagnitude': 6.0,
            'orderby': 'time-desc',
            'limit': 20
        }

        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()

        df = pd.read_csv(io.StringIO(response.text))

        if df.empty:
            return None, "🌿 Nenhum terremoto M ≥ 6.0 nos últimos 30 dias — período calmo globalmente!"

        df = df[['time', 'mag', 'place', 'depth']].head(10).copy()
        df.columns = ['Data/Hora (UTC)', 'Magnitude', 'Local', 'Profundidade (km)']
        df['Magnitude'] = df['Magnitude'].round(1)

        return df, f"✅ Atualizado agora: {datetime.utcnow().strftime('%d/%m/%2025 %H:%M')} UTC"

    except Exception:
        return None, "⚠️ Falha ao carregar dados da USGS. Tentando novamente em 1 minuto..."

alerts_df, message = load_earthquake_alerts()

with alert_placeholder.container():
    status_placeholder.success(message)
    if alerts_df is not None:
        st.dataframe(alerts_df, use_container_width=True, hide_index=True)  # <-- CORRIGIDO aqui também

# ==================== RODAPÉ ====================
st.markdown("---")
st.markdown(
    """
    **EarthQuake AI** – Projeto portfólio 2025  
    Modelo: HistGradientBoostingRegressor (scikit-learn)  
    Dados: USGS Earthquake Catalog (1990–2025) + API em tempo real  
    Feito com ❤️ e Streamlit · Alertas atualizados a cada minuto  
    """
)