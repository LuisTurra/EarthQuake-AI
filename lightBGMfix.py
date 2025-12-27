import duckdb
import pandas as pd
from lightgbm import LGBMRegressor
import joblib

con = duckdb.connect(database='earthquake.duckdb', read_only=True)

# Carrega dados reais
df = con.execute("SELECT latitude, longitude, depth, year, month, day, hour, magnitude FROM earthquakes").df()

# Amostra 70% pra treinamento rápido mas preciso
train_df = df.sample(frac=0.7, random_state=42)
X_train = train_df[['latitude', 'longitude', 'depth', 'year', 'month', 'day', 'hour']]
y_train = train_df['magnitude']

# Treina modelo idêntico ao original
model = LGBMRegressor(n_estimators=500, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Salva novo .pkl compatível
joblib.dump(model, 'model_magnitude_predictor.pkl')
print("Modelo real re-treinado e salvo – números mantidos e compatível com Cloud!")