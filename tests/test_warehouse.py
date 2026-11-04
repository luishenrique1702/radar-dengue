import pandas as pd
import pytest

from radar_dengue.warehouse import build_tables, connect, queries, run_query
from tests.test_features import fake_data


@pytest.fixture
def con():
    raw = fake_data(n_weeks=120)
    raw["nivel"] = 1
    geos = raw["geocode"].unique()
    prev = pd.DataFrame({
        "geocode": geos, "horizonte": 1, "data_alvo": pd.Timestamp("2024-05-05"),
        "casos_previstos": [40.0, 60.0], "casos_min": [30.0, 45.0], "casos_max": [50.0, 80.0],
        "incidencia_100k": [80.0, 120.0], "nivel_risco": ["Alto", "Muito alto"],
    })
    teste = pd.DataFrame({
        "geocode": list(geos) * 2, "horizonte": [1, 1, 2, 2],
        "casos_reais": [10.0, 20.0, 12.0, 18.0], "casos_previstos": [11.0, 19.0, 10.0, 20.0],
        "baseline": [8.0, 25.0, 9.0, 24.0],
    })
    c = connect(build_tables(raw, prev, teste))
    yield c
    c.close()


def test_tabelas_criadas(con):
    nomes = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert {"municipios", "casos_semanais", "previsoes", "previsoes_teste"} <= nomes


@pytest.mark.parametrize("arquivo", queries(), ids=lambda p: p.stem)
def test_todas_as_consultas_rodam(con, arquivo):
    df = run_query(con, arquivo)
    assert isinstance(df, pd.DataFrame)


def test_erro_por_porte_calcula_reducao(con):
    df = run_query(con, next(q for q in queries("05")))
    linha = df[df["horizonte"] == 1].iloc[0]
    assert linha["erro_modelo"] == pytest.approx(1.0)
    assert linha["erro_baseline"] == pytest.approx(3.5)
