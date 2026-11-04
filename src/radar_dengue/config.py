"""Configurações centrais do projeto."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
GEOJSON = ROOT / "data" / "geo" / "sp_municipios.geojson"
MODELS = ROOT / "models"
FIGURES = ROOT / "reports" / "figures"

INFODENGUE_URL = "https://info.dengue.mat.br/api/alertcity"
YEAR_START = 2015
YEAR_END = 2026

# Horizontes de previsão, em semanas à frente
HORIZONS = [1, 2, 3, 4]

# Semana epidemiológica de corte entre treino e teste
# Treino: até o fim de 2023 | Teste: 2024 em diante (inclui a epidemia recorde de 2024)
TEST_START = "2024-01-01"
