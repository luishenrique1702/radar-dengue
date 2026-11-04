import pandas as pd
import pytest

from radar_dengue.validate import DataValidationError, validate_raw
from tests.test_features import fake_data


def base_valida(n_cidades: int = 645) -> pd.DataFrame:
    cidades = [str(3500000 + i) for i in range(n_cidades)]
    return fake_data(n_weeks=10, cities=cidades)


def test_dados_validos_passam():
    df = base_valida()
    hoje = df["data_iniSE"].max() + pd.Timedelta(days=7)
    assert validate_raw(df, hoje=hoje) == []


def test_coluna_ausente():
    with pytest.raises(DataValidationError, match="Colunas ausentes"):
        validate_raw(base_valida().drop(columns=["casos"]))


def test_poucos_municipios():
    with pytest.raises(DataValidationError, match="municípios"):
        validate_raw(base_valida(n_cidades=100))


def test_casos_negativos():
    df = base_valida()
    df.loc[0, "casos"] = -1
    with pytest.raises(DataValidationError, match="negativas"):
        validate_raw(df)


def test_dados_desatualizados_geram_aviso():
    df = base_valida()
    avisos = validate_raw(df, hoje=df["data_iniSE"].max() + pd.Timedelta(days=60))
    assert any("desatualizados" in a for a in avisos)
