import duckdb
import pandas as pd
from lightgbm import LGBMRegressor
import joblib

con = duckdb.connect(database='earthquake.duckdb', read_only=True)

df = con.execute("SELECT latitude, longitude, depth, year, month, day, hour, magnitude FROM earthquakes").df()

X = df[['latitude', 'longitude', 'depth', 'year', 'month', 'day', 'hour']]
y = df['magnitude']

model = LGBMRegressor(n_estimators=500, learning_rate=0.1, random_state=42)
model.fit(X, y)

joblib.dump(model, 'model_magnitude_predictor.pkl')
print("Modelo salvo ")