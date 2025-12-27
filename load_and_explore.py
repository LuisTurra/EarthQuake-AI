import duckdb
import polars as pl
import pandas as pd

# Caminho do seu CSV 
csv_path = "dataset/Earthquakes_USGS.csv"  

# Conecta DuckDB 
con = duckdb.connect(database='earthquake.duckdb', read_only=False)

# Carrega o CSV direto no DuckDB 
print("Carregando o CSV... isso pode levar alguns minutos na primeira vez")
con.execute(f"""
CREATE OR REPLACE TABLE earthquakes_raw AS 
SELECT * FROM read_csv_auto('{csv_path}', header=true)
""")

# Teste básico
rows = con.execute("SELECT COUNT(*) FROM earthquakes_raw").fetchone()[0]
print(f"Total de eventos carregados: {rows:,}")

# Colunas disponíveis
columns = con.execute("DESCRIBE earthquakes_raw").df()
print("\nColunas:")
print(columns)

# Amostra dos dados
sample = con.execute("SELECT * FROM earthquakes_raw LIMIT 10").df()
print("\nPrimeiras 10 linhas:")
print(sample)

# Estatísticas rápidas (ex: período coberto)
period = con.execute("""
SELECT 
    MIN(time) AS data_minima,
    MAX(time) AS data_maxima,
    COUNT(*) AS total_eventos
FROM earthquakes_raw
""").df()
print("\nPeríodo e total:")
print(period)