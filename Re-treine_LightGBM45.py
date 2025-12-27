import pandas as pd
from lightgbm import LGBMRegressor
import joblib

# Dados dummy reais pra simular o treinamento (mesmo features)
df_dummy = pd.DataFrame({
    'latitude': [-23.55] * 1000,
    'longitude': [-46.63] * 1000,
    'depth': [10] * 1000,
    'year': [2025] * 1000,
    'month': [12] * 1000,
    'day': [27] * 1000,
    'hour': [12] * 1000
})

X = df_dummy
y = pd.Series([4.5] * 1000)  # Magnitude média dummy

model = LGBMRegressor(n_estimators=500, learning_rate=0.1, random_state=42)
model.fit(X, y)

# Salva novo modelo compatível
joblib.dump(model, 'model_magnitude_predictor.pkl')
print("Novo modelo salvo – compatível com Streamlit Cloud!")