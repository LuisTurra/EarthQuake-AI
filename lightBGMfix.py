import duckdb
import pandas as pd
from lightgbm import LGBMRegressor
import joblib

con = duckdb.connect(database='earthquake.duckdb', read_only=True)

df = con.execute("SELECT latitude, longitude, depth, year, month, day, hour, magnitude FROM earthquakes").df()

# Usa 70% dos dados pra treinamento (mantém precisão alta)
train_df = df.sample(frac=0.7, random_state=42)
X_train = train_df[['latitude', 'longitude', 'depth', 'year', 'month', 'day', 'hour']]
y_train = train_df['magnitude']

model = LGBMRegressor(n_estimators=500, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'model_magnitude_predictor.pkl')
print("Modelo re-treinado com LightGBM novo – números reais mantidos e compatível com Cloud!")