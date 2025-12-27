import duckdb
import pandas as pd
from lightgbm import LGBMRegressor
from prophet import Prophet
import numpy as np
from sklearn.metrics import mean_absolute_error
import joblib  # Pra salvar os modelos
import os

con = duckdb.connect(database='earthquake.duckdb', read_only=True)

print("Carregando dados limpos para ML...")
df = con.execute("SELECT * FROM earthquakes").df()

print(f"Dados carregados: {len(df):,} linhas")

# ==================== ML #1: Predição de Magnitude ====================
print("\nTreinando modelo #1: Predição de Magnitude (LightGBM)")

features = ['latitude', 'longitude', 'depth', 'year', 'month', 'day', 'hour']
target = 'magnitude'

# Amostra pra treinamento rápido 
train_df = df.sample(frac=0.7, random_state=42)
val_df = df.drop(train_df.index)

X_train = train_df[features]
y_train = train_df[target]
X_val = val_df[features]
y_val = val_df[target]

model_mag = LGBMRegressor(n_estimators=500, learning_rate=0.1, random_state=42)
model_mag.fit(X_train, y_train)

pred_val = model_mag.predict(X_val)
mae = mean_absolute_error(y_val, pred_val)
print(f"Erro médio absoluto (MAE) na validação: {mae:.3f} (quanto menor, melhor – <0.5 é ótimo pra magnitude)")

# Salva o modelo
joblib.dump(model_mag, 'model_magnitude_predictor.pkl')
print("Modelo de predição de magnitude salvo!")

# Exemplo de uso: "O que esperar se ocorrer um terremoto aqui?"
exemplo = pd.DataFrame([{
    'latitude': -23.55,   
    'longitude': -46.63,
    'depth': 10,
    'year': 2025,
    'month': 12,
    'day': 27,
    'hour': 15
}])
pred_sp = model_mag.predict(exemplo)[0]
print(f"Previsão de magnitude esperada em São Paulo (exemplo): {pred_sp:.2f}")

# ==================== ML #2: Forecast com Prophet (por continente) ====================
print("\nTreinando modelo #2: Forecast de eventos por mês (Prophet)")

# Agrega eventos mensais por continente
monthly = con.execute("""
SELECT 
    DATE_TRUNC('month', earthquake_time) AS ds,
    continent_simple,
    COUNT(*) AS y,
    AVG(magnitude) AS avg_mag
FROM earthquakes
GROUP BY ds, continent_simple
ORDER BY ds
""").df()

# Treina um Prophet por continente (exemplo: Américas – o maior)
americas = monthly[monthly['continent_simple'] == 'Américas'][['ds', 'y']].copy()
americas.rename(columns={'y': 'y'}, inplace=True)  

prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
prophet_model.fit(americas)

future = prophet_model.make_future_dataframe(periods=12, freq='M')  # Previsão 12 meses
forecast = prophet_model.predict(future)

print("Previsão de eventos mensais nas Américas (próximos 6 meses):")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(6))

# Salva modelo Prophet
joblib.dump(prophet_model, 'prophet_americas_forecast.pkl')
print("Modelo Prophet salvo!")

# Plota e salva gráfico
fig = prophet_model.plot(forecast)
fig.savefig('forecast_americas.png')
print("Gráfico salvo como forecast_americas.png")

print("\nMachine Learning concluído!")