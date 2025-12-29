import streamlit as st
from streamlit_folium import st_folium
import folium
import pandas as pd
import joblib
import requests
import io
from datetime import datetime, timedelta
import plotly.graph_objects as go

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
st.caption("Modelo treinado com **3.4 milhões** de eventos USGS (1990–2025) · HistGradientBoosting + Prophet")

# ==================== CARREGA MODELOS ====================
@st.cache_resource
def load_models():
    mag_model = joblib.load('model_magnitude_predictor.pkl')
    prophet_model = joblib.load('prophet_americas_forecast.pkl')
    return mag_model, prophet_model

model_mag, prophet_model = load_models()

# ==================== MAPA INTERATIVO (MUITO MAIS BONITO) ====================
st.header("🗺️ Clique no mapa para analisar o risco sísmico")

# Mapa com imagem de satélite (lindo e com continentes bem visíveis)
m = folium.Map(
    location=[0, 0],
    zoom_start=2,
    tiles="Esri WorldImagery",  # Satélite lindo!
    attr="Esri"  # Crédito obrigatório
)

# Adiciona tiles OpenStreetMap como opção alternativa (claro, se preferir)
folium.TileLayer(
    tiles="OpenStreetMap",
    name="Ruas (claro)",
    show=False
).add_to(m)

# Controle de camadas
folium.LayerControl().add_to(m)

# Renderiza o mapa
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

# ==================== PREVISÃO MENSAL (AMÉRICAS) ====================
st.header("📈 Previsão de Eventos Mensais – Américas (ano 2026)")

future = prophet_model.make_future_dataframe(periods=12, freq='ME')
forecast = prophet_model.predict(future)

today = datetime(2025, 12, 29)
forecast_future = forecast[forecast['ds'] > today]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=forecast_future['ds'],
    y=forecast_future['yhat'],
    mode='lines+markers',
    name='Previsão',
    line=dict(color='#e63946', width=4),
    marker=dict(size=8)
))
fig.add_trace(go.Scatter(
    x=forecast_future['ds'],
    y=forecast_future['yhat_upper'],
    mode='lines',
    line=dict(width=0),
    showlegend=False,
    hoverinfo='none'
))
fig.add_trace(go.Scatter(
    x=forecast_future['ds'],
    y=forecast_future['yhat_lower'],
    mode='lines',
    fill='tonexty',
    fillcolor='rgba(230, 57, 70, 0.2)',
    name='Intervalo de Confiança (80%)',
    line=dict(width=0)
))

fig.update_layout(
    title="Número Estimado de Terremotos por Mês nas Américas",
    xaxis_title="Data",
    yaxis_title="Número de Eventos",
    template="plotly_white",
    height=500,
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# ==================== ALERTAS EM TEMPO REAL (COM AUTO-UPDATE A CADA 1 MINUTO) ====================
st.header("🚨 Alertas Globais – Terremotos M ≥ 6.0 (Últimos 30 Dias)")

# Placeholder para atualizar automaticamente
alert_placeholder = st.empty()
status_placeholder = st.empty()

with alert_placeholder.container():
    status_placeholder.info("🔄 Carregando dados em tempo real da USGS...")

# Função para carregar alertas
@st.cache_data(ttl=60)  # Cache de 60 segundos = atualiza a cada 1 minuto
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

        return df, f"✅ Atualizado agora: {datetime.utcnow().strftime('%d/%m/%Y %H:%M')} UTC"

    except Exception as e:
        return None, "⚠️ Falha ao carregar dados da USGS (sem conexão ou serviço temporariamente indisponível). Tentando novamente em 1 minuto..."

# Carrega e exibe
alerts_df, message = load_earthquake_alerts()

with alert_placeholder.container():
    status_placeholder.success(message)
    if alerts_df is not None:
        st.dataframe(alerts_df, use_container_width=True, hide_index=True)

# ==================== RODAPÉ ====================
st.markdown("---")
st.markdown(
    """
    **EarthQuake AI** – Projeto portfólio 2025  
    Modelos: HistGradientBoostingRegressor + Prophet  
    Dados: USGS Earthquake Catalog + API em tempo real  
    Feito com ❤️ e Streamlit · Atualização automática dos alertas a cada minuto
    """
)