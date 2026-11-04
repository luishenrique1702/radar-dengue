"""Comparação com SARIMAX ajustado cidade por cidade.

Para cada município com mais de 100 mil habitantes, ajusta um modelo SARIMAX em log(1 + casos):
- parte ARIMA (p, d, q) escolhida por AIC entre algumas ordens comuns;
- sazonalidade anual por termos de Fourier (regressão harmônica dinâmica), a forma usual de tratar
  períodos longos como 52 semanas, em que um SARIMA com componente sazonal de período 52 é instável;
- opcionalmente, temperatura e umidade médias das 4 semanas que terminam 4 semanas antes, que já
  são conhecidas no momento de qualquer previsão de até 4 semanas.

Os parâmetros são estimados com dados até 2023 e mantidos fixos no teste; a cada semana, o filtro
de Kalman incorpora a observação nova e o modelo prevê de 1 a 4 semanas à frente.

Uso:
    python -m radar_dengue.sarimax
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from radar_dengue.config import DATA_RAW, HORIZONS, MODELS, TEST_START

ORDENS = [(1, 0, 0), (2, 0, 0), (1, 0, 1), (2, 0, 1), (0, 1, 1), (1, 1, 1)]
POP_MINIMA = 100_000


def fourier(semanas: pd.Series, k: int = 3) -> pd.DataFrame:
    w = semanas.to_numpy()
    cols = {}
    for j in range(1, k + 1):
        cols[f"sen{j}"] = np.sin(2 * np.pi * j * w / 52.18)
        cols[f"cos{j}"] = np.cos(2 * np.pi * j * w / 52.18)
    return pd.DataFrame(cols, index=semanas.index)


def city_frame(g: pd.DataFrame, clima: bool) -> tuple[pd.Series, pd.DataFrame]:
    g = g.sort_values("data_iniSE").set_index("data_iniSE").asfreq("W-SUN")
    y = np.log1p(g["casos"].astype(float).ffill())
    semana = pd.Series(g.index.isocalendar().week.to_numpy(), index=g.index)
    x = fourier(semana)
    if clima:
        for c in ["tempmed", "umidmed"]:
            v = g[c].ffill().bfill()
            x[c] = v.shift(4).rolling(4, min_periods=1).mean().bfill()
    return y, x


def fit_city(y: pd.Series, x: pd.DataFrame):
    """Escolhe a ordem por AIC e devolve o melhor ajuste."""
    melhor = None
    for ordem in ORDENS:
        try:
            m = sm.tsa.SARIMAX(y, exog=x, order=ordem, trend="c" if ordem[1] == 0 else "n",
                               enforce_stationarity=True, enforce_invertibility=True)
            r = m.fit(disp=False, maxiter=200)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if np.isfinite(r.aic) and (melhor is None or r.aic < melhor.aic):
            melhor = r
    return melhor


def rolling_forecasts(res, y: pd.Series, x: pd.DataFrame, origens: pd.DatetimeIndex) -> pd.DataFrame:
    """Previsões de 1 a 4 semanas a partir de cada origem, com parâmetros fixos."""
    completo = res.apply(y, exog=x)  # filtra a série inteira com os parâmetros do treino
    linhas = []
    pos = {d: i for i, d in enumerate(y.index)}
    hmax = max(HORIZONS)
    for origem in origens:
        i = pos[origem]
        if i + hmax >= len(y):
            break
        parcial = completo.apply(y.iloc[: i + 1], exog=x.iloc[: i + 1])
        fc = parcial.forecast(hmax, exog=x.iloc[i + 1: i + 1 + hmax])
        for h in HORIZONS:
            # expm1 da média do log é a mediana na escala de casos: a previsão ótima para o MAE
            linhas.append({"data_iniSE": origem, "horizonte": h,
                           "previsto": float(np.expm1(fc.iloc[h - 1]).clip(0))})
    return pd.DataFrame(linhas)


def run(raw: pd.DataFrame, clima: bool) -> pd.DataFrame:
    pop = raw.groupby("geocode")["pop"].max()
    cidades = pop[pop >= POP_MINIMA].index
    saida = []
    for geo in cidades:
        g = raw[raw["geocode"] == geo]
        y, x = city_frame(g, clima)
        treino = y.index < TEST_START
        res = fit_city(y[treino], x[treino])
        if res is None:
            continue
        origens = y.index[~treino]
        prev = rolling_forecasts(res, y, x, origens)
        prev["geocode"] = geo
        prev["ordem"] = str(res.model.order)
        saida.append(prev)
    return pd.concat(saida, ignore_index=True)


def main() -> None:
    warnings.filterwarnings("ignore")
    raw = pd.read_parquet(DATA_RAW / "infodengue_sp.parquet")
    resultados = {}
    for clima in (False, True):
        nome = "SARIMAX com clima" if clima else "SARIMAX (sazonalidade de Fourier)"
        prev = run(raw, clima)
        prev.to_parquet(MODELS / f"sarimax{'_clima' if clima else ''}.parquet", index=False)
        resultados[nome] = {"cidades": int(prev["geocode"].nunique()),
                            "ordens": prev.drop_duplicates("geocode")["ordem"].value_counts().to_dict()}
        print(nome, resultados[nome])
    with open(MODELS / "sarimax_info.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
