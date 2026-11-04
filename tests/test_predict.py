import numpy as np
import pandas as pd

from radar_dengue.predict import correct_recent_weeks, forecast
from tests.test_features import fake_data


class ModeloConstante:
    """Modelo de mentira: prevê sempre variação zero (casos iguais aos de hoje)."""

    def predict(self, X):
        return np.zeros(len(X))


def test_previsao_tem_uma_linha_por_municipio_e_horizonte():
    raw = fake_data()
    prev = forecast(raw, models={h: ModeloConstante() for h in [1, 2, 3, 4]})
    assert len(prev) == raw["geocode"].nunique() * 4
    assert (prev["casos_previstos"] >= 0).all()
    assert prev["nivel_risco"].isin(["Baixo", "Moderado", "Alto", "Muito alto"]).all()


def test_previsao_aponta_para_o_futuro():
    raw = fake_data()
    prev = forecast(raw, models={1: ModeloConstante()})
    assert (prev["data_alvo"] > raw["data_iniSE"].max()).all()


def test_correcao_das_semanas_recentes():
    raw = fake_data()
    raw.loc[raw["data_iniSE"] == raw["data_iniSE"].max(), ["casos", "casos_est"]] = [3, 40.0]
    corrigido = correct_recent_weeks(raw)
    ultima = corrigido[corrigido["data_iniSE"] == raw["data_iniSE"].max()]
    assert (ultima["casos"] == 40).all()
    # semanas antigas não mudam
    antigas = raw["data_iniSE"] < raw["data_iniSE"].max() - pd.Timedelta(weeks=4)
    pd.testing.assert_series_equal(corrigido.loc[antigas, "casos"], raw.loc[antigas, "casos"],
                                   check_dtype=False)
