"""Engenharia de atributos para previsão semanal de dengue.

Todos os atributos de uma linha usam apenas informação disponível até a semana t,
para que o modelo nunca "veja o futuro" (vazamento de dados).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CLIMATE = ["tempmin", "tempmed", "tempmax", "umidmin", "umidmed", "umidmax"]

FEATURES_BASE = [
    "log_casos_lag0", "log_casos_lag1", "log_casos_lag2", "log_casos_lag3",
    "log_casos_lag52", "log_media4", "log_media8", "crescimento_4s",
    "inc100k_lag0", "Rt", "receptivo",
    "tempmed_media4", "tempmin_media4", "tempmax_media4",
    "umidmed_media4", "umidmin_media4", "umidmax_media4",
    "tempmed_lag4", "umidmed_lag4",
    "sem_sin", "sem_cos", "log_pop", "cresc_estado_4s",
]
FEATURES_RT = ["Rt_var2", "Rt_media3"]
FEATURES_VIZINHOS = ["viz_log_casos", "viz_inc100k", "viz_cresc_4s"]
# Vizinhos e tendência do Rt foram testados em radar_dengue.experimentos e ficaram fora do
# modelo: na validação 2021–2023, anteciparam 37% dos inícios de epidemia com 4 semanas, contra
# 32%, mas no backtest de 2019 a 2026 o erro ficou igual na média, pior em 4 das 8 temporadas
# e pior na epidemia de 2024.
FEATURES = FEATURES_BASE


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

    # Tendência do Rt: um Rt que sobe indica aceleração antes de os casos subirem muito
    g = d.groupby("geocode", group_keys=False)
    d["Rt_var2"] = d["Rt"] - g["Rt"].shift(2)
    d["Rt_media3"] = g["Rt"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    return add_neighbor_features(d)


def add_neighbor_features(d: pd.DataFrame) -> pd.DataFrame:
    """Casos nos municípios que fazem fronteira, na semana t e a tendência em 4 semanas.

    A dengue se espalha entre cidades vizinhas; os vizinhos podem sinalizar o início de uma
    epidemia antes de ela aparecer nos casos do próprio município. Municípios sem vizinhos na
    malha (ou fora dela) ficam com valores ausentes, que o LightGBM trata.
    """
    from radar_dengue.geo import neighbors

    viz = neighbors()
    codigos = d["geocode"].astype(str)
    casos = d.assign(geocode=codigos).pivot_table(
        index="data_iniSE", columns="geocode", values="casos", aggfunc="sum")
    pop = d.assign(geocode=codigos).pivot_table(
        index="data_iniSE", columns="geocode", values="pop", aggfunc="mean")
    cols = list(casos.columns)
    pos = {c: i for i, c in enumerate(cols)}
    m = np.zeros((len(cols), len(cols)))
    for a in cols:
        for b in viz.get(a, []):
            if b in pos:
                m[pos[a], pos[b]] = 1.0
    sem_viz = m.sum(1) == 0
    soma = casos.fillna(0).to_numpy() @ m.T
    soma_pop = pop.ffill().bfill().fillna(0).to_numpy() @ m.T
    soma[:, sem_viz] = np.nan
    log_viz = pd.DataFrame(np.log1p(soma), index=casos.index, columns=cols)
    inc_viz = pd.DataFrame(soma / np.where(soma_pop > 0, soma_pop, np.nan) * 1e5,
                           index=casos.index, columns=cols)
    longo = pd.DataFrame({
        "viz_log_casos": log_viz.stack(future_stack=True),
        "viz_inc100k": inc_viz.stack(future_stack=True),
        "viz_cresc_4s": (log_viz - log_viz.shift(4)).stack(future_stack=True),
    }).reset_index().rename(columns={"level_1": "geocode_str"})
    longo.columns = ["data_iniSE", "geocode_str", "viz_log_casos", "viz_inc100k", "viz_cresc_4s"]
    out = d.assign(geocode_str=codigos).merge(longo, on=["data_iniSE", "geocode_str"], how="left")
    return out.drop(columns="geocode_str").set_index(d.index)


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
