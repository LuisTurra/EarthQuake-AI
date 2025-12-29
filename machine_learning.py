import duckdb
import polars as pl
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# Conecta ao banco DuckDB
con = duckdb.connect(database='earthquake.duckdb', read_only=True)

print("Carregando dados limpos para ML com Polars...")
df_pl = con.execute("SELECT * FROM earthquakes").pl()

print(f"Dados carregados: {len(df_pl):,} linhas")

# ==================== ML #1: Predição de Magnitude ====================
print("\nTreinando modelo #1: Predição de Magnitude (HistGradientBoosting)")

features_mag = ['latitude', 'longitude', 'depth', 'year', 'month', 'day', 'hour']
target = 'magnitude'

# Split treino/validação
df_pl = df_pl.with_row_index(name="temp_id")
train_pl = df_pl.sample(fraction=0.7, shuffle=True, seed=42)
val_pl = df_pl.filter(~pl.col("temp_id").is_in(train_pl["temp_id"]))

print(f"Treino: {len(train_pl):,} linhas | Validação: {len(val_pl):,} linhas")

X_train = train_pl.drop("temp_id").select(features_mag).to_pandas()
y_train = train_pl[target].to_numpy()
X_val = val_pl.drop("temp_id").select(features_mag).to_pandas()
y_val = val_pl[target].to_numpy()

model_mag = HistGradientBoostingRegressor(
    max_iter=500,
    learning_rate=0.1,
    max_depth=8,
    random_state=42,
    loss='absolute_error',
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10
)

print("Treinando o modelo de predição de magnitude...")
model_mag.fit(X_train, y_train)

pred_val = model_mag.predict(X_val)
mae = mean_absolute_error(y_val, pred_val)
print(f"MAE na validação (magnitude): {mae:.3f}")

# Salva modelo de magnitude
joblib.dump(model_mag, 'model_magnitude_predictor.pkl')
print("Modelo de predição de magnitude salvo → model_magnitude_predictor.pkl")

# Exemplo São Paulo
exemplo = pl.DataFrame({
    'latitude': [-23.55],
    'longitude': [-46.63],
    'depth': [10.0],
    'year': [2025],
    'month': [12],
    'day': [29],
    'hour': [15]
})
pred_sp = model_mag.predict(exemplo.select(features_mag).to_pandas())[0]
print(f"Previsão exemplo São Paulo: {pred_sp:.2f}")

# ==================== ML #2: Forecast Mensal (sem Prophet) ====================
print("\nTreinando modelo #2: Previsão de eventos mensais nas Américas (HistGradientBoosting)")

# Série temporal mensal - Américas
monthly_pl = (
    df_pl
    .sort("earthquake_time")
    .group_by_dynamic("earthquake_time", every="1mo", closed="left")
    .agg(
        y=pl.len()  # número de terremotos no mês
    )
    .filter(pl.col("earthquake_time").dt.year() >= 1990)  # segurança
)

# Engenharia de features para sazonalidade
monthly_df = monthly_pl.to_pandas()
monthly_df['month'] = monthly_df['earthquake_time'].dt.month
monthly_df['year'] = monthly_df['earthquake_time'].dt.year
monthly_df['month_sin'] = np.sin(2 * np.pi * monthly_df['month'] / 12)
monthly_df['month_cos'] = np.cos(2 * np.pi * monthly_df['month'] / 12)

features_forecast = ['year', 'month_sin', 'month_cos']
X_ts = monthly_df[features_forecast]
y_ts = monthly_df['y']

model_forecast = HistGradientBoostingRegressor(
    max_iter=300,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

print("Treinando modelo de forecast mensal...")
model_forecast.fit(X_ts, y_ts)

# Avaliação simples (últimos 12 meses como "validação")
pred_ts = model_forecast.predict(X_ts)
print(f"MAE aproximado no histórico mensal: {mean_absolute_error(y_ts, pred_ts):.1f} eventos")

# Salva modelo de forecast
joblib.dump(model_forecast, 'model_monthly_forecast.pkl')
print("Modelo de forecast mensal salvo → model_monthly_forecast.pkl")

# ==================== Gera gráfico estático da previsão (opcional) ====================
# Previsão para os próximos 12 meses (a partir de jan/2026)
future_dates = pd.date_range(start="2026-01-01", periods=12, freq="MS")
future_df = pd.DataFrame({
    'ds': future_dates,
    'month': future_dates.month,
    'year': future_dates.year
})
future_df['month_sin'] = np.sin(2 * np.pi * future_df['month'] / 12)
future_df['month_cos'] = np.cos(2 * np.pi * future_df['month'] / 12)

X_future = future_df[features_forecast]
pred_future = model_forecast.predict(X_future)

# Gráfico completo: histórico recente + previsão
recent_hist = monthly_df.tail(24)  # últimos 24 meses
plt.figure(figsize=(12, 6))
plt.plot(recent_hist['earthquake_time'], recent_hist['y'], 
         label="Histórico (Américas)", color="#1f77b4", linewidth=2)
plt.plot(future_df['ds'], pred_future, 
         label="Previsão 2026", color="#e63946", linewidth=3, marker="o")
plt.title("Número de Terremotos Mensais nas Américas – Histórico + Previsão 2026")
plt.xlabel("Data")
plt.ylabel("Número de Eventos")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('forecast_americas.png', dpi=150)
plt.close()
print("Gráfico salvo → forecast_americas.png")

print("\nTreinamento concluído com sucesso!")
print("Arquivos gerados (prontos para Streamlit Cloud):")
print("  • model_magnitude_predictor.pkl")
print("  • model_monthly_forecast.pkl")
print("  • forecast_americas.png")
