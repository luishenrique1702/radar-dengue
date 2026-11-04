"""Gera a previsão das próximas semanas para todos os municípios.

Uso:
    python -m radar_dengue.predict
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from radar_dengue.config import DATA_PROCESSED, DATA_RAW, HORIZONS, MODELS
from radar_dengue.features import FEATURES, build_features, risk_level
from radar_dengue.intervals import NIVEIS, predict_bounds
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
    """Carrega, para cada horizonte, o modelo central, os quantílicos e o ajuste conformal."""
    ajustes = {}
    arq_int = MODELS / "intervalos.json"
    if arq_int.exists():
        ajustes = json.loads(arq_int.read_text(encoding="utf-8"))
    models = {}
    for h in HORIZONS:
        m = {"central": joblib.load(MODELS / f"lgbm_h{h}.joblib")}
        arq = MODELS / f"quantis_h{h}.joblib"
        if arq.exists():
            m["quantis"] = joblib.load(arq)
            m["ajustes"] = {int(n): v["ajuste_log"] for n, v in ajustes.get(f"h{h}", {}).items()}
        models[h] = m
    return models


def forecast(raw: pd.DataFrame, models: dict | None = None) -> pd.DataFrame:
    """Previsão para as próximas semanas a partir da semana mais recente dos dados.

    `models` mapeia horizonte -> modelo, ou horizonte -> {"central", "quantis", "ajustes"}.
    """
    d = build_features(correct_recent_weeks(raw))
    last = d[d["data_iniSE"] == d["data_iniSE"].max()].copy()
    models = models or load_models()
    atual = last["log_casos_lag0"].to_numpy()
    pop = last["pop"].to_numpy()

    out = []
    for h, m in models.items():
        m = m if isinstance(m, dict) else {"central": m}
        central = m["central"]
        if getattr(central, "objective", None) == "poisson":  # offset na escala de log
            casos = np.exp(central.predict(last[FEATURES], raw_score=True) + atual)
        else:
            casos = np.expm1(central.predict(last[FEATURES]) + atual).clip(0)
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
        if "quantis" in m:
            for nivel in NIVEIS:
                ajuste = m.get("ajustes", {}).get(nivel, 0.0)
                lo, hi = predict_bounds(m["quantis"], last, nivel, ajuste)
                # garante a ordem inferior <= central <= superior
                bloco[f"casos_min_{nivel}"] = np.minimum(np.expm1(lo).clip(0), casos).round(1)
                bloco[f"casos_max_{nivel}"] = np.maximum(np.expm1(hi).clip(0), casos).round(1)
            # o intervalo de 80% é o padrão usado no monitoramento e no SQL
            bloco["casos_min"] = bloco["casos_min_80"]
            bloco["casos_max"] = bloco["casos_max_80"]
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
