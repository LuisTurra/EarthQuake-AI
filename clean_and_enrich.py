import duckdb

# Conecta no mesmo banco de antes (pra usar a tabela raw)
con = duckdb.connect(database='earthquake.duckdb', read_only=False)

print("Criando tabela limpa e enriquecida...")

con.execute("""
CREATE OR REPLACE TABLE earthquakes_clean AS
SELECT
    time::TIMESTAMP AS earthquake_time,
    latitude,
    longitude,
    depth,
    mag AS magnitude,
    place,
    title,
    -- Filtra dados válidos
    CASE 
        WHEN mag IS NOT NULL 
         AND mag >= -1 AND mag <= 10  -- Magnitude realista
         AND latitude BETWEEN -90 AND 90
         AND longitude BETWEEN -180 AND 180
         AND depth >= 0
        THEN 1 ELSE 0 
    END AS valid_row,

    -- Energia liberada em joules (fórmula Richter aproximada)
    CASE 
        WHEN mag IS NOT NULL THEN POW(10, (1.5 * mag + 4.8))
        ELSE NULL 
    END AS energy_joules,

    -- Features temporais
    EXTRACT(YEAR FROM time) AS year,
    EXTRACT(MONTH FROM time) AS month,
    EXTRACT(DAY FROM time) AS day,
    EXTRACT(HOUR FROM time) AS hour,

    -- Continente simples (aproximação por coordenadas – bom pra dashboard)
    CASE
        WHEN longitude BETWEEN -180 AND -20 AND latitude BETWEEN -60 AND 75 THEN 'Américas'
        WHEN longitude BETWEEN -20 AND 60 AND latitude BETWEEN -35 AND 70 THEN 'Europa/África'
        WHEN longitude BETWEEN 60 AND 180 AND latitude BETWEEN -50 AND 70 THEN 'Ásia/Oceania'
        ELSE 'Outros/Oceano'
    END AS continent_simple

FROM earthquakes_raw
WHERE 
    time >= '1990-01-01'  -- Filtra 1990 em diante
    AND time <= '2025-12-31'
""")

# Filtra só linhas válidas na tabela final
con.execute("""
CREATE OR REPLACE TABLE earthquakes AS
SELECT 
    * EXCLUDE (valid_row)  -- Remove coluna auxiliar
FROM earthquakes_clean
WHERE valid_row = 1
""")

# Estatísticas da tabela limpa
stats = con.execute("""
SELECT 
    COUNT(*) AS total_eventos,
    MIN(earthquake_time) AS data_min,
    MAX(earthquake_time) AS data_max,
    AVG(magnitude) AS media_magnitude,
    MAX(magnitude) AS max_magnitude,
    MIN(energy_joules) AS min_energy,
    MAX(energy_joules) AS max_energy
FROM earthquakes
""").df()

print("\nTabela 'earthquakes' criada com sucesso!")
print(stats)

# Quantos por continente (pra ver se a regra tá boa)
continentes = con.execute("""
SELECT continent_simple, COUNT(*) AS eventos
FROM earthquakes
GROUP BY continent_simple
ORDER BY eventos DESC
""").df()
print("\nEventos por continente (aproximado):")
print(continentes)

# Amostra final
sample_clean = con.execute("SELECT * FROM earthquakes LIMIT 10").df()
print("\nAmostra da tabela final:")
print(sample_clean)