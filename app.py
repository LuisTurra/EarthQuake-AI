import streamlit as st
from streamlit_folium import st_folium
import folium
import pandas as pd
import joblib
import requests
import io
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Config página
st.set_page_config(page_title="EarthQuake AI", layout="wide")

st.title("🌍 EarthQuake AI")
st.markdown("### Previsão de Risco Sísmico por Localização")
st.markdown("_Clique no mapa para estimar o risco em qualquer lugar do mundo · Modelo IA treinado com 3.4M eventos USGS_")

# Carrega o modelo de magnitude
@st.cache_resource
def load_model():
    return joblib.load('model_magnitude_predictor.pkl')

model = load_model()

# ==================== MAPA FOLIUM CLICÁVEL ====================
st.header("🗺️ Mapa Interativo – Clique para ver o risco")

m = folium.Map(location=[0, 0], zoom_start=2, tiles="OpenStreetMap")

# Marker padrão no Brasil
folium.Marker(
    location=[-23.55, -46.63],
    popup="São Paulo (padrão)",
    icon=folium.Icon(color="blue")
).add_to(m)

map_data = st_folium(m, width=800, height=500, key="folium_map")

# ==================== RISCO ESTIMADO ====================
st.header("🔴 Risco Sísmico Estimado")

if map_data and map_data.get("last_clicked"):
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]
    st.success(f"📍 Localização clicada: {lat:.4f}, {lon:.4f}")
else:
    lat = -23.55
    lon = -46.63
    st.info("📍 Clique no mapa para selecionar (padrão: São Paulo, Brasil)")

col1, col2 = st.columns(2)
with col1:
    st.metric("Latitude", f"{lat:.4f}")
with col2:
    st.metric("Longitude", f"{lon:.4f}")

# Previsão com LightGBM
input_data = pd.DataFrame([{
    'latitude': lat,
    'longitude': lon,
    'depth': 10,
    'year': 2025,
    'month': 12,
    'day': 27,
    'hour': 12
}])

pred_mag = model.predict(input_data)[0]

st.metric("Magnitude Média Estimada (padrão histórico)", f"{pred_mag:.2f}")

if pred_mag < 4.0:
    risco = "🟢 Baixo"
    explicacao = "Terremotos geralmente imperceptíveis. Baixo risco de danos."
elif pred_mag < 5.5:
    risco = "🟡 Moderado"
    explicacao = "Terremotos sentidos ocasionalmente. Risco moderado em eventos raros."
else:
    risco = "🔴 Alto"
    explicacao = "Região com histórico de terremotos mais fortes. Maior potencial de danos."

st.markdown(f"**Nível de Risco Relativo:** {risco}")
st.info(explicacao)
st.warning("Nota: Grandes terremotos (M>7) são raros e não previsíveis com precisão. Esta é uma estimativa estatística baseada em milhões de eventos históricos.")

# ==================== INFO SOBRE MAGNITUDE ====================
st.header("📊 O que significa cada magnitude?")
st.markdown("""
- **< 4.0** → Geralmente **não sentido** ou muito leve.
- **4.0 – 4.9** → Sentido por muitos. Raros danos.
- **5.0 – 5.9** → Pode causar danos leves a moderados.
- **6.0 – 6.9** → Danos moderados a graves.
- **≥ 7.0** → Terremoto grave ou catastrófico.

Fonte: U.S. Geological Survey (USGS)
""")

# ==================== PREVISÃO DE EVENTOS ====================
st.header("📈 Previsão de Eventos Mensais – Américas (Próximos 12 Meses)")

prophet_model = joblib.load('prophet_americas_forecast.pkl')
future = prophet_model.make_future_dataframe(periods=12, freq='ME')
forecast = prophet_model.predict(future)
forecast_future = forecast[forecast['ds'] > datetime.today()]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=forecast_future['ds'],
    y=forecast_future['yhat'],
    mode='lines+markers',
    name='Previsão',
    line=dict(color='#e63946', width=4)
))
fig.add_trace(go.Scatter(
    x=forecast_future['ds'],
    y=forecast_future['yhat_upper'],
    mode='lines',
    line=dict(width=0),
    showlegend=False
))
fig.add_trace(go.Scatter(
    x=forecast_future['ds'],
    y=forecast_future['yhat_lower'],
    mode='lines',
    fill='tonexty',
    fillcolor='rgba(230,57,70,0.2)',
    name='Intervalo de confiança'
))

fig.update_layout(
    title="Número estimado de terremotos por mês",
    xaxis_title="Mês",
    yaxis_title="Eventos",
    template="simple_white",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# ==================== HOT ZONES ESTÁTICA ====================
st.header("⚠️ Hot Zones – Média Histórica (1990–2025)")
hot_data = pd.DataFrame([
    {"Região": "Américas", "Nº de Eventos": "3.025.497", "Magnitude Média": "4.82"},
    {"Região": "Ásia/Oceania", "Nº de Eventos": "266.361", "Magnitude Média": "4.75"},
    {"Região": "Europa/África", "Nº de Eventos": "132.030", "Magnitude Média": "4.45"},
    {"Região": "Outros/Oceano", "Nº de Eventos": "9.240", "Magnitude Média": "4.60"}
])
st.table(hot_data)

# ==================== ALERTAS EM TEMPO REAL (ROBUSTO) ====================
st.header("🚨 Alertas – Terremotos M > 6.0 (Últimos 30 Dias)")

try:
    # Datas em formato ISO completo (UTC)
    end_time = datetime.utcnow().isoformat(timespec='seconds')
    start_time = (datetime.utcnow() - timedelta(days=30)).isoformat(timespec='seconds')

    url = f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime={start_time}&endtime={end_time}&minmagnitude=6.0&orderby=time-desc&limit=20"

    response = requests.get(url, timeout=15)
    response.raise_for_status()  # Erro se não 200

    csv_text = response.text.strip()
    if len(csv_text) < 100 or 'time' not in csv_text:  # Header mínimo
        raise ValueError("Resposta vazia ou inválida")

    alerts_df = pd.read_csv(io.StringIO(csv_text))
    if not alerts_df.empty:
        alerts_df = alerts_df[['time', 'magnitude', 'place', 'depth']].head(10)
        alerts_df.rename(columns={
            'time': 'Data/Hora (UTC)',
            'magnitude': 'Magnitude',
            'place': 'Local',
            'depth': 'Profundidade (km)'
        }, inplace=True)
        st.table(alerts_df)
    else:
        st.success("🌿 Nenhum terremoto acima de M6.0 nos últimos 30 dias – período relativamente calmo!")
except requests.exceptions.RequestException:
    st.warning("Problema de conexão com a USGS (timeout ou rede). Tente recarregar.")
except Exception:
    st.warning("Dados temporariamente indisponíveis da USGS. Normal em picos – recarregue em alguns minutos.")

st.caption("Projeto portfólio 2025 · LightGBM + Prophet · Dados em tempo real da USGS")