"""Gera a previsão das próximas semanas para todos os municípios.

Uso:
    python -m radar_dengue.predict
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from radar_dengue.config import DATA_PROCESSED, DATA_RAW, HORIZONS, MODELS
from radar_dengue.features import FEATURES, build_features, risk_level
from radar_dengue.monitor import log_forecast
from radar_dengue.validate import validate_raw


def correct_recent_weeks(raw: pd.DataFrame, weeks: int = 4) -> pd.DataFrame:
    """Nas semanas mais recentes, as notificações ainda estão chegando.

    Usamos a estimativa corrigida do próprio InfoDengue (casos_est, um "nowcast")
    para não subestimar a situação atual.
    """
    d = raw.copy()
    limite = d["data_iniSE"].max() - pd.Timedelta(weeks=weeks - 1)
    recente = d["data_iniSE"] >= limite
    d["casos"] = d["casos"].astype(float)
    d.loc[recente, "casos"] = d.loc[recente, ["casos", "casos_est"]].max(axis=1).round()
    return d


def load_models() -> dict:
    """Carrega, para cada horizonte, o modelo central e os quantílicos (se existirem)."""
    models = {}
    for h in HORIZONS:
        m = {"central": joblib.load(MODELS / f"lgbm_h{h}.joblib")}
        for q in ["q10", "q90"]:
            arq = MODELS / f"lgbm_h{h}_{q}.joblib"
            if arq.exists():
                m[q] = joblib.load(arq)
        models[h] = m
    return models


def forecast(raw: pd.DataFrame, models: dict | None = None) -> pd.DataFrame:
    """Previsão para as próximas semanas a partir da semana mais recente dos dados.

    `models` mapeia horizonte -> modelo, ou horizonte -> {"central", "q10", "q90"}.
    """
    d = build_features(correct_recent_weeks(raw))
    last = d[d["data_iniSE"] == d["data_iniSE"].max()].copy()
    models = models or load_models()
    atual = last["log_casos_lag0"].to_numpy()
    pop = last["pop"].to_numpy()

    out = []
    for h, m in models.items():
        m = m if isinstance(m, dict) else {"central": m}
        casos = np.expm1(m["central"].predict(last[FEATURES]) + atual).clip(0)
        inc = casos / pop * 1e5
        bloco = pd.DataFrame({
            "geocode": last["geocode"].values,
            "municipio": last["municipio"].values,
            "semana_base": last["data_iniSE"].values,
            "horizonte": h,
            "data_alvo": last["data_iniSE"].values + np.timedelta64(7 * h, "D"),
            "casos_previstos": casos.round(1),
            "incidencia_100k": inc.round(2),
            "nivel_risco": risk_level(inc),
            "pop": pop,
        })
        if "q10" in m and "q90" in m:
            lo = np.expm1(m["q10"].predict(last[FEATURES]) + atual).clip(0)
            hi = np.expm1(m["q90"].predict(last[FEATURES]) + atual).clip(0)
            # garante a ordem inferior <= central <= superior
            bloco["casos_min"] = np.minimum(lo, casos).round(1)
            bloco["casos_max"] = np.maximum(hi, casos).round(1)
        out.append(bloco)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    raw = pd.read_parquet(DATA_RAW / "infodengue_sp.parquet")
    validate_raw(raw)
    prev = forecast(raw)
    prev.to_parquet(DATA_PROCESSED / "previsao_atual.parquet", index=False)
    log_forecast(prev)

    # Série histórica enxuta para o dashboard
    hist = raw[["geocode", "municipio", "data_iniSE", "casos", "casos_est", "pop",
                "tempmed", "umidmed"]]
    hist.to_parquet(DATA_PROCESSED / "historico.parquet", index=False)
    print(prev.groupby(["horizonte", "nivel_risco"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
