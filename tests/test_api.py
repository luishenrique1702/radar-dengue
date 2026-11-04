from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "data/processed/previsao_atual.parquet").exists(),
                                reason="previsões ausentes")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from radar_dengue.api import app
    return TestClient(app)


def test_saude(client):
    r = client.get("/saude")
    assert r.status_code == 200
    assert r.json()["municipios"] == 645


def test_previsao_por_nome_sem_acento(client):
    r = client.get("/previsao/sao paulo")
    assert r.status_code == 200
    corpo = r.json()
    assert corpo["municipio"] == "São Paulo"
    assert [p["horizonte"] for p in corpo["previsoes"]] == [1, 2, 3, 4]
    for p in corpo["previsoes"]:
        niveis = {i["nivel"]: i for i in p["intervalos"]}
        assert niveis[50]["minimo"] >= niveis[95]["minimo"]
        assert niveis[95]["minimo"] <= p["casos_previstos"] <= niveis[95]["maximo"]


def test_previsao_por_codigo_ibge(client):
    assert client.get("/previsao/3509502").json()["municipio"] == "Campinas"


def test_municipio_inexistente(client):
    assert client.get("/previsao/Gotham").status_code == 404


def test_alertas_ordenados(client):
    r = client.get("/alertas", params={"horizonte": 4})
    assert r.status_code == 200
    inc = [a["incidencia_100k"] for a in r.json()]
    assert inc == sorted(inc, reverse=True)
    assert all(a["nivel_risco"] in ("Alto", "Muito alto") for a in r.json())


def test_horizonte_invalido(client):
    assert client.get("/alertas", params={"horizonte": 9}).status_code == 422


def test_busca_municipios(client):
    nomes = [m["municipio"] for m in client.get("/municipios", params={"busca": "ribeirao"}).json()]
    assert "Ribeirão Preto" in nomes
