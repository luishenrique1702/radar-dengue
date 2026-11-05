"""Intervalos de previsão calibrados (regressão quantílica + calibração conformal).

Os modelos quantílicos do LightGBM estimam os limites inferior e superior. Sozinhos, eles
costumam ficar estreitos demais em períodos atípicos. A calibração conformal (CQR) mede, num
período separado, o quanto os intervalos erraram e alarga cada limite na medida certa para que
"80%" cubra de fato cerca de 80% dos valores reais.

Referência: Romano, Patterson e Candès (2019), "Conformalized Quantile Regression".
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from radar_dengue.features import FEATURES

NIVEIS = {50: (0.25, 0.75), 80: (0.10, 0.90), 95: (0.025, 0.975)}
PARAMS_Q = dict(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=50,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42,
                verbose=-1)


def fit_quantiles(df: pd.DataFrame, h: int) -> dict:
    """Um modelo por quantil, prevendo a variação de log(1 + casos) em relação à semana atual."""
    alvo = df[f"y_h{h}"] - df["log_casos_lag0"]
    modelos = {}
    for lo, hi in NIVEIS.values():
        for a in (lo, hi):
            if a not in modelos:
                modelos[a] = lgb.LGBMRegressor(**PARAMS_Q, objective="quantile", alpha=a).fit(
                    df[FEATURES], alvo)
    return modelos


def predict_bounds(modelos: dict, df: pd.DataFrame, nivel: int, ajuste: float = 0.0):
    """Limites em log(1 + casos), já com o ajuste conformal somado nas duas pontas."""
    lo_a, hi_a = NIVEIS[nivel]
    base = df["log_casos_lag0"].to_numpy()
    lo = modelos[lo_a].predict(df[FEATURES]) + base - ajuste
    hi = modelos[hi_a].predict(df[FEATURES]) + base + ajuste
    return np.minimum(lo, hi), np.maximum(lo, hi)


def conformal_adjustment(modelos: dict, calib: pd.DataFrame, h: int, nivel: int) -> float:
    """Quanto alargar o intervalo (em log) para atingir a cobertura nominal na calibração."""
    y = calib[f"y_h{h}"].to_numpy()
    lo, hi = predict_bounds(modelos, calib, nivel)
    escores = np.maximum(lo - y, y - hi)
    n = len(escores)
    q = min(1.0, (nivel / 100) * (1 + 1 / n))
    return float(np.quantile(escores, q))


def coverage(modelos: dict, df: pd.DataFrame, h: int, nivel: int, ajuste) -> float:
    y = df[f"y_h{h}"].to_numpy()
    lo, hi = predict_bounds(modelos, df, nivel, ajuste)
    return float(np.mean((y >= lo) & (y <= hi)))


def size_groups(pop: pd.Series) -> np.ndarray:
    """Porte do município, usado na calibração separada por grupo."""
    from radar_dengue.geo import size_class
    return size_class(pop).astype(str).to_numpy()


def adjustment_by_size(modelos: dict, calib: pd.DataFrame, h: int, nivel: int) -> dict:
    """Calibração conformal separada por porte (Mondrian conformal).

    Cada porte recebe o seu próprio alargamento. Com um ajuste único, as cidades pequenas, que
    são a maioria das linhas, definem o ajuste, e as grandes ficam com intervalos estreitos.
    """
    grupos = size_groups(calib["pop"])
    return {g: conformal_adjustment(modelos, calib[grupos == g], h, nivel) for g in np.unique(grupos)}


def adjustments_for(df: pd.DataFrame, por_porte: dict) -> np.ndarray:
    return pd.Series(size_groups(df["pop"])).map(por_porte).to_numpy(dtype=float)


def coverage_by_size(modelos: dict, df: pd.DataFrame, h: int, nivel: int, ajuste) -> dict:
    y = df[f"y_h{h}"].to_numpy()
    lo, hi = predict_bounds(modelos, df, nivel, ajuste)
    dentro = (y >= lo) & (y <= hi)
    grupos = size_groups(df["pop"])
    return {g: float(dentro[grupos == g].mean()) for g in np.unique(grupos)}


def evaluate(d: pd.DataFrame, h: int, consolidated, test_start: str,
             calib_start: str = "2023-01-01") -> dict:
    """Ajusta até 2022, calibra em 2023 e mede a cobertura no período de teste (2024+)."""
    base = consolidated(d, h)
    alvo = base[f"data_alvo_h{h}"]
    ajuste_df = base[alvo < calib_start]
    calib = base[(alvo >= calib_start) & (alvo < test_start)]
    teste = base[base["data_iniSE"] >= test_start]
    modelos = fit_quantiles(ajuste_df, h)
    resultado = {}
    for nivel in NIVEIS:
        q = conformal_adjustment(modelos, calib, h, nivel)
        q_porte = adjustment_by_size(modelos, calib, h, nivel)
        a_porte = adjustments_for(teste, q_porte)
        resultado[str(nivel)] = {
            "ajuste_log": q,
            "ajuste_log_por_porte": q_porte,
            "cobertura_sem_calibracao": coverage(modelos, teste, h, nivel, 0.0),
            "cobertura_calibrada": coverage(modelos, teste, h, nivel, q),
            "cobertura_calibrada_por_porte": coverage(modelos, teste, h, nivel, a_porte),
            "cobertura_por_porte": {
                "ajuste_unico": coverage_by_size(modelos, teste, h, nivel, q),
                "ajuste_por_porte": coverage_by_size(modelos, teste, h, nivel, a_porte),
            },
        }
    return resultado
