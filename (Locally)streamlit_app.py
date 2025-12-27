import streamlit as st
from streamlit_folium import st_folium
import folium
import pandas as pd
import joblib
import duckdb
from datetime import datetime
import plotly.graph_objects as go
import requests
from io import StringIO
from datetime import datetime, timedelta

# Config página
st.set_page_config(page_title="EarthQuake AI", layout="wide")

st.title("🌍 EarthQuake AI")
st.markdown("### Previsão de Risco Sísmico – Clique no mapa para saber o risco em qualquer lugar")

# Carrega o modelo de magnitude
@st.cache_resource
def load_model():
    return joblib.load('model_magnitude_predictor.pkl')

model = load_model()

# ==================== MAPA COM FOLIUM (CLIQUE PERFEITO) ====================
st.header("🗺️ Mapa Interativo – Clique em qualquer lugar do mundo")

# Mapa inicial centrado no mundo (ou Brasil se quiser)
m = folium.Map(location=[0, 0], zoom_start=2, tiles="OpenStreetMap")

# Adiciona marker temporário (opcional – só pra beleza)
folium.Marker(
    location=[-23.55, -46.63],
    popup="São Paulo (exemplo padrão)",
    icon=folium.Icon(color="blue", icon="info-sign")
).add_to(m)

# Renderiza o mapa e captura o clique
map_data = st_folium(m, width=800, height=500, key="map")

# ==================== RISCO NO PONTO CLICADO ====================
st.header("🔴 Risco Sísmico na Localização Selecionada")

if map_data and map_data.get("last_clicked"):
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]
    st.success(f"📍 Localização clicada: Latitude {lat:.4f}, Longitude {lon:.4f}")
else:
    lat = -23.55
    lon = -46.63
    st.info("📍 Clique no mapa para selecionar um local (padrão: São Paulo, Brasil)")

col1, col2 = st.columns(2)
with col1:
    st.metric("Latitude", f"{lat:.4f}")
with col2:
    st.metric("Longitude", f"{lon:.4f}")

# Previsão IA
input_data = pd.DataFrame([{
    'latitude': lat,
    'longitude': lon,
    'depth': 10,
    'year': 2025,
    'month': 12,
    'day': 27,
    'hour': 12
}])
# Previsão
pred_mag = model.predict(input_data)[0]

st.metric("Magnitude Média Estimada (baseado em padrões históricos)", f"{pred_mag:.2f}")

if pred_mag < 4.0:
    risco = "🟢 Baixo"
    explicacao = "Região com terremotos tipicamente pequenos e imperceptíveis. Baixo risco de danos."
elif pred_mag < 5.5:
    risco = "🟡 Moderado"
    explicacao = "Terremotos sentidos ocasionalmente. Risco moderado de danos leves em eventos raros."
else:
    risco = "🔴 Alto"
    explicacao = "Região com histórico de terremotos mais fortes. Maior potencial de danos em eventos significativos."

st.markdown(f"**Nível de Risco Relativo:** {risco}")
st.info(explicacao)
st.warning("Nota: Grandes terremotos (M>7) são raros em todo o mundo e não podem ser previstos com precisão. Esta é uma estimativa estatística baseada em milhões de eventos históricos da USGS.")
# ==================== INFO SOBRE MAGNITUDE ====================
st.header("📊 O que significa cada magnitude?")
st.markdown("""
- **< 4.0** → Geralmente **não sentido** ou muito leve.
- **4.0 – 4.9** → Sentido por muitos. Raros danos.
- **5.0 – 5.9** → Pode causar danos leves a moderados.
- **6.0 – 6.9** → Danos moderados a graves.
- **≥ 7.0** → Terremoto grave ou catastrófico.

Fonte: USGS
""")

# ==================== PREVISÃO DE EVENTOS ====================
st.header("📈 Previsão de Eventos Mensais – Américas")

prophet_model = joblib.load('prophet_americas_forecast.pkl')
future = prophet_model.make_future_dataframe(periods=12, freq='ME')
forecast = prophet_model.predict(future)
forecast_future = forecast[forecast['ds'] > datetime.today()]

fig_forecast = go.Figure()
fig_forecast.add_trace(go.Scatter(
    x=forecast_future['ds'],
    y=forecast_future['yhat'],
    mode='lines+markers',
    name='Previsão',
    line=dict(color='#e63946', width=4)
))
fig_forecast.add_trace(go.Scatter(
    x=forecast_future['ds'],
    y=forecast_future['yhat_upper'],
    mode='lines',
    line=dict(width=0),
    showlegend=False
))
fig_forecast.add_trace(go.Scatter(
    x=forecast_future['ds'],
    y=forecast_future['yhat_lower'],
    mode='lines',
    fill='tonexty',
    fillcolor='rgba(230,57,70,0.2)',
    name='Intervalo de confiança'
))

fig_forecast.update_layout(
    title="Número estimado de terremotos por mês",
    xaxis_title="Mês",
    yaxis_title="Eventos",
    template="simple_white",
    height=500
)

st.plotly_chart(fig_forecast, use_container_width=True)

# ==================== ALERTAS EM TEMPO REAL ====================
st.header("🚨 Alertas – Terremotos M > 6.0 (Últimos 30 Dias)")

# Datas UTC em ISO completo
end_time = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
start_time = (datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m-%dT%H:%M:%SZ')

url = f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime={start_time}&endtime={end_time}&minmagnitude=6.0&orderby=time-desc&limit=20"

st.info(f"Debug: Usando URL {url} – cole no navegador pra testar direto.")

try:
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    csv_text = response.text.strip()
    if 'time' not in csv_text:
        raise ValueError("Resposta sem dados válidos")

    alerts_df = pd.read_csv(StringIO(csv_text))
    if not alerts_df.empty:
        alerts_df = alerts_df[['time', 'mag', 'place', 'depth']].head(10)
        alerts_df.rename(columns={
            'time': 'Data/Hora (UTC)',
            'mag': 'Magnitude',
            'place': 'Local',
            'depth': 'Profundidade (km)'
        }, inplace=True)
        st.table(alerts_df)
    else:
        st.success("🌿 Nenhum terremoto acima de M6.0 nos últimos 30 dias – período calmo!")
except Exception as e:
    st.warning(f"Erro ao carregar: {str(e)}. API USGS pode estar lenta – tente recarregar ou ver o URL no navegador.")

st.caption("Projeto portfólio 2025 · Clique no mapa para previsão instantânea · LightGBM + Prophet + DuckDB")