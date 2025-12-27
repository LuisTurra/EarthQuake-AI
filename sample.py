import duckdb
import pandas as pd

# Conecta no banco
con = duckdb.connect(database='earthquake.duckdb', read_only=True)

# Pega amostra de 10 linhas
sample = con.execute("SELECT * FROM earthquakes_raw LIMIT 10").df()

# Configura Pandas pra mostrar TODAS as colunas e sem truncar texto
pd.set_option('display.max_columns', None)   # Mostra todas as colunas
pd.set_option('display.width', None)         # Largura automática
pd.set_option('display.max_colwidth', 50)    # Máximo 50 caracteres por célula (aumenta se quiser mais)
pd.set_option('display.expand_frame_repr', False)  # Não quebra linha

print("\nAmostra completa das primeiras 10 linhas (todas as colunas visíveis):")
print(sample)