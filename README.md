# 🌍 EarthQuake AI – Analisador Global de Risco Sísmico com Inteligência Artificial

**Projeto de portfólio – Data Science & Machine Learning**

Um aplicativo interativo que permite ao usuário:
- **Clique em qualquer lugar do mundo** para estimar o risco sísmico local com base em padrões históricos.
- Ver **previsão de eventos futuros** usando Prophet.
- Consultar **alertas em tempo real** da USGS (terremotos M > 6.0 últimos 30 dias).
- Entender o significado de cada magnitude de forma clara.

![Demo do App](https://luisturra-deteccao-fraude-cartao-credito-streamlit-app-99a0gz.streamlit.app/)  


## 🚀 Tecnologias Utilizadas

- **Python** + **Streamlit** (app web interativo)
- **Folium** (mapa clicável com OpenStreetMap)
- **HistGradientBoost** (modelo de regressão para estimativa de magnitude média histórica)
- **HistGradientBoost** (forecast de eventos mensais)
- **DuckDB** (processamento local eficiente de 3.4M eventos USGS)
- **API USGS** (dados em tempo real)

## 📊 Dados

- Fonte: U.S. Geological Survey (USGS) Earthquake Catalog
- Período: 1990 – 2025
- Total: +3.4 milhões de eventos globais
- Processamento: limpeza, feature engineering e treinamento local com DuckDB

## 🛠️ Como Rodar Localmente

1. Clone o repositório:
   ```bash
   git clone https://github.com/SEU_USUARIO/EarthQuake-AI.git
   cd EarthQuake-AI