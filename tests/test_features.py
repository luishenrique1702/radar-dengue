import numpy as np
import pandas as pd
import pytest

from radar_dengue.features import FEATURES, add_targets, build_features, risk_level


def fake_data(n_weeks: int = 80, cities=("3500105", "3550308")) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for c in cities:
        datas = pd.date_range("2022-01-02", periods=n_weeks, freq="7D")
        for i, d in enumerate(datas):
            rows.append({
                "geocode": c, "municipio": f"Cidade {c}", "data_iniSE": d,
                "SE": d.isocalendar().year * 100 + d.isocalendar().week,
                "casos": int(rng.poisson(20)), "casos_est": 20.0, "pop": 50_000, "Rt": 1.0,
                "receptivo": 0, "tempmin": 18.0, "tempmed": 22.0, "tempmax": 27.0,
                "umidmin": 50.0, "umidmed": 70.0, "umidmax": 90.0 if i % 9 else np.nan,
            })
    return pd.DataFrame(rows)


def test_todas_as_features_existem():
    d = build_features(fake_data())
    assert set(FEATURES) <= set(d.columns)


def test_lag_nao_mistura_municipios():
    d = build_features(fake_data())
    primeira = d.groupby("geocode").head(1)
    assert primeira["log_casos_lag1"].isna().all()


def test_lag_usa_apenas_o_passado():
    d = build_features(fake_data()).sort_values(["geocode", "data_iniSE"])
    cidade = d[d["geocode"] == "3500105"]
    esperado = np.log1p(cidade["casos"]).shift(1)
    pd.testing.assert_series_equal(cidade["log_casos_lag1"], esperado, check_names=False)


def test_alvo_e_o_futuro():
    d = add_targets(build_features(fake_data()), [2])
    cidade = d[d["geocode"] == "3500105"].reset_index(drop=True)
    assert cidade.loc[0, "y_h2"] == pytest.approx(np.log1p(cidade.loc[2, "casos"]))
    assert cidade.loc[0, "data_alvo_h2"] == cidade.loc[2, "data_iniSE"]


def test_clima_sem_buracos():
    d = build_features(fake_data())
    assert d["umidmax"].notna().all()


@pytest.mark.parametrize("inc, nivel", [(0, "Baixo"), (9.9, "Baixo"), (10, "Moderado"),
                                        (49, "Moderado"), (75, "Alto"), (250, "Muito alto")])
def test_niveis_de_risco(inc, nivel):
    assert risk_level([inc])[0] == nivel
