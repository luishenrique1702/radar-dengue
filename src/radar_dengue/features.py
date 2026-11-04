"""Engenharia de atributos para previsão semanal de dengue.

Todos os atributos de uma linha usam apenas informação disponível até a semana t,
para que o modelo nunca "veja o futuro" (vazamento de dados).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CLIMATE = ["tempmin", "tempmed", "tempmax", "umidmin", "umidmed", "umidmax"]

FEATURES = [
    "log_casos_lag0", "log_casos_lag1", "log_casos_lag2", "log_casos_lag3",
    "log_casos_lag52", "log_media4", "log_media8", "crescimento_4s",
    "inc100k_lag0", "Rt", "receptivo",
    "tempmed_media4", "tempmin_media4", "tempmax_media4",
    "umidmed_media4", "umidmin_media4", "umidmax_media4",
    "tempmed_lag4", "umidmed_lag4",
    "sem_sin", "sem_cos", "log_pop", "cresc_estado_4s",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recebe a base semanal (um registro por município e semana) e devolve atributos."""
    d = df.sort_values(["geocode", "data_iniSE"]).copy()
    g = d.groupby("geocode", group_keys=False)

    # Clima: preenche falhas pontuais com o valor da semana anterior do mesmo município
    d[CLIMATE] = g[CLIMATE].transform(lambda s: s.ffill().bfill())

    d["log_casos"] = np.log1p(d["casos"])
    for lag in [0, 1, 2, 3, 52]:
        d[f"log_casos_lag{lag}"] = g["log_casos"].shift(lag)
    d["log_media4"] = g["log_casos"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    d["log_media8"] = g["log_casos"].transform(lambda s: s.rolling(8, min_periods=1).mean())
    d["crescimento_4s"] = d["log_casos_lag0"] - g["log_casos"].shift(4)
    d["inc100k_lag0"] = d["casos"] / d["pop"] * 1e5

    for c in CLIMATE:
        d[f"{c}_media4"] = g[c].transform(lambda s: s.rolling(4, min_periods=1).mean())
    d["tempmed_lag4"] = g["tempmed"].shift(4)
    d["umidmed_lag4"] = g["umidmed"].shift(4)

    semana = d["SE"] % 100
    d["sem_sin"] = np.sin(2 * np.pi * semana / 52)
    d["sem_cos"] = np.cos(2 * np.pi * semana / 52)
    d["log_pop"] = np.log(d["pop"])

    # Tendência do estado inteiro: ajuda a captar epidemias que se espalham entre cidades
    # (usamos a variação, e não o nível, para não depender do tamanho da epidemia)
    estado = np.log1p(d.groupby("data_iniSE")["casos"].sum().sort_index())
    cresc = (estado - estado.shift(4)).rename("cresc_estado_4s")
    d = d.merge(cresc, left_on="data_iniSE", right_index=True, how="left")
    return d


def add_targets(d: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Alvo: log(1 + casos) h semanas à frente, e a data a que ele se refere."""
    g = d.groupby("geocode", group_keys=False)
    for h in horizons:
        d[f"y_h{h}"] = g["log_casos"].shift(-h)
        d[f"data_alvo_h{h}"] = g["data_iniSE"].shift(-h)
    return d


def risk_level(inc100k: pd.Series | np.ndarray) -> np.ndarray:
    """Classifica a incidência semanal prevista (casos por 100 mil habitantes)."""
    bins = [-np.inf, 10, 50, 100, np.inf]
    labels = np.array(["Baixo", "Moderado", "Alto", "Muito alto"])
    idx = np.digitize(np.asarray(inc100k, dtype=float), bins[1:-1])
    return labels[idx]
