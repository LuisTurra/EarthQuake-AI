import duckdb
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Conecta ao banco DuckDB
con = duckdb.connect(database='earthquake.duckdb', read_only=True)

print("Carregando dados limpos para ML com Polars...")
df_pl = con.execute("SELECT * FROM earthquakes").pl()

print(f"Dados carregados: {len(df_pl):,} linhas")

# ==================== ML #1: Predição de Magnitude ====================
print("\nTreinando modelo #1: Predição de Magnitude (HistGradientBoosting + Polars)")

features = ['latitude', 'longitude', 'depth', 'year', 'month', 'day', 'hour']
target = 'magnitude'

# Adiciona ID temporário para split exato
df_pl = df_pl.with_row_index(name="temp_id")

# Split treino/validação
train_pl = df_pl.sample(fraction=0.7, shuffle=True, seed=42)
val_pl = df_pl.filter(~pl.col("temp_id").is_in(train_pl["temp_id"]))

print(f"Treino: {len(train_pl):,} linhas | Validação: {len(val_pl):,} linhas")

# Prepara dados para o sklearn (remove temp_id e converte)
X_train = train_pl.drop("temp_id").select(features).to_pandas()
y_train = train_pl[target].to_numpy()

X_val = val_pl.drop("temp_id").select(features).to_pandas()
y_val = val_pl[target].to_numpy()

# Modelo HistGradientBoosting (compatível com Streamlit Cloud)
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

# Avaliação
pred_val = model_mag.predict(X_val)
mae = mean_absolute_error(y_val, pred_val)
print(f"Erro médio absoluto (MAE) na validação: {mae:.3f}")
print("(Valores ~0.4-0.6 são normais para predição de magnitude)")

# Salva modelo
joblib.dump(model_mag, 'model_magnitude_predictor.pkl')
print("Modelo de predição de magnitude salvo!")

# Exemplo: predição em São Paulo
exemplo_pl = pl.DataFrame({
    'latitude': [-23.55],
    'longitude': [-46.63],
    'depth': [10.0],
    'year': [2025],
    'month': [12],
    'day': [29],
    'hour': [15]
})

pred_sp = model_mag.predict(exemplo_pl.select(features).to_pandas())[0]
print(f"Previsão de magnitude esperada em São Paulo (exemplo): {pred_sp:.2f}")

# ==================== ML #2: Forecast com Prophet ====================
print("\nPreparando série temporal mensal com Polars...")

# CORREÇÃO PRINCIPAL: ordenar por earthquake_time ANTES do group_by_dynamic
monthly_pl = (
    df_pl
    .sort("earthquake_time")  # <--- ESSENCIAL para group_by_dynamic funcionar
    .group_by_dynamic("earthquake_time", every="1mo", closed="left")
    .agg(
        continent_simple=pl.col("continent_simple").mode().first(),  # mantém continente
        y=pl.len(),                  # <--- CORRIGIDO: pl.count() → pl.len()
        avg_mag=pl.col("magnitude").mean()
    )
)

# Filtra apenas Américas (maior volume de dados)
americas_pl = monthly_pl.filter(pl.col("continent_simple") == "Américas")

# Prepara para Prophet
americas_df = americas_pl.select([
    pl.col("earthquake_time").alias("ds"),
    pl.col("y").cast(pl.Float64).alias("y")
]).to_pandas()

print(f"Série temporal pronta: {len(americas_df)} meses de dados (Américas)")

# Treina Prophet
prophet_model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    seasonality_mode='multiplicative',
    changepoint_prior_scale=0.05
)

print("Treinando modelo Prophet...")
prophet_model.fit(americas_df)

# Previsão futura
future = prophet_model.make_future_dataframe(periods=12, freq='MS')  # MS = início do mês
forecast = prophet_model.predict(future)

print("\nPrevisão de terremotos mensais nas Américas (próximos 6 meses):")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(6).round({
    'yhat': 0, 'yhat_lower': 0, 'yhat_upper': 0
}))

# Salva modelo e gráfico
joblib.dump(prophet_model, 'prophet_americas_forecast.pkl')
print("Modelo Prophet salvo!")

fig = prophet_model.plot(forecast)
plt.title("Previsão Mensal de Terremotos - Américas")
fig.savefig('forecast_americas.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Gráfico salvo como 'forecast_americas.png'")

print("\nMachine Learning concluído com sucesso!")
print("Arquivos gerados:")
print("  - model_magnitude_predictor.pkl")
print("  - prophet_americas_forecast.pkl")
print("  - forecast_americas.png")
print("\nPronto para usar no Streamlit!")