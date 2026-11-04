"""Compara o LightGBM com métodos alternativos no período de teste.

Uso:
    python -m radar_dengue.compare
"""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm

from radar_dengue.config import DATA_RAW, HORIZONS, MODELS
from radar_dengue.features import FEATURES, add_targets, build_features
from radar_dengue.model import PARAMS, split


def mae(real, prev) -> float:
    return float(np.mean(np.abs(np.asarray(real) - np.clip(np.asarray(prev, float), 0, None))))


def main() -> None:
    raw = pd.read_parquet(DATA_RAW / "infodengue_sp.parquet")
    d = add_targets(build_features(raw), HORIZONS)
    resultado = {}
    for h in HORIZONS:
        tr, te = split(d, h)
        y = f"y_h{h}"
        real = np.expm1(te[y]).to_numpy()
        atual = te["log_casos_lag0"].to_numpy()
        lin = sm.OLS(tr[y] - tr["log_casos_lag0"], sm.add_constant(tr[FEATURES])).fit()
        gbm = lgb.LGBMRegressor(**PARAMS).fit(tr[FEATURES], tr[y] - tr["log_casos_lag0"])
        resultado[f"h{h}"] = {
            "Repetir o mesmo período do ano anterior": mae(real, np.expm1(te["log_casos_lag52"])),
            "Regressão linear": mae(
                real, np.expm1(lin.predict(sm.add_constant(te[FEATURES])) + atual)),
            "Repetir a semana atual (baseline)": mae(real, np.expm1(atual)),
            "LightGBM (modelo escolhido)": mae(real, np.expm1(gbm.predict(te[FEATURES]) + atual)),
        }
        print(h, {k: round(v, 1) for k, v in resultado[f"h{h}"].items()})
    with open(MODELS / "comparacao.json", "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
