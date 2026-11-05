"""Backtest com origem móvel: o modelo é avaliado em várias temporadas, e não só em 2024–2026.

Para cada ano de teste Y, o modelo central é treinado só com alvos anteriores a Y e avaliado nas
semanas de Y. Os intervalos seguem o esquema de produção: quantis ajustados até Y-2, calibração
conformal em Y-1 e avaliação em Y. Nada de Y entra no treino ou na calibração.

Além do erro absoluto (MAE), a previsão probabilística é avaliada pelo WIS (weighted interval
score), a métrica do CDC e do desafio Infodengue-Mosqlimate. O WIS combina a mediana e os
intervalos de 50%, 80% e 95%: premia intervalos estreitos e pune os que não cobrem o valor real,
e aproxima o CRPS. Valores menores são melhores. O baseline probabilístico repete a semana atual
e usa como incerteza os quantis empíricos da variação observada no treino.

Uso:
    python -m radar_dengue.backtest
"""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from radar_dengue.config import DATA_RAW, HORIZONS, MODELS
from radar_dengue.features import FEATURES, add_targets, build_features
from radar_dengue.intervals import NIVEIS, PARAMS_Q, conformal_adjustment, fit_quantiles, predict_bounds
from radar_dengue.model import consolidated, fit_central, predict_central

ANOS = list(range(2019, 2027))


def interval_score(y, lo, hi, alpha: float) -> np.ndarray:
    """Interval score de um intervalo central de nível 1 - alpha (Gneiting e Raftery, 2007)."""
    return (hi - lo) + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)


def wis(y, mediana, limites: dict) -> np.ndarray:
    """WIS de Bracher et al. (2021). `limites` mapeia o nível (50, 80, 95) para (lo, hi)."""
    total = 0.5 * np.abs(y - mediana)
    for nivel, (lo, hi) in limites.items():
        alpha = 1 - nivel / 100
        total = total + (alpha / 2) * interval_score(y, lo, hi, alpha)
    return total / (len(limites) + 0.5)


def baseline_bounds(treino: pd.DataFrame, teste: pd.DataFrame, h: int) -> dict:
    """Persistência com a incerteza dos quantis empíricos da variação em log no treino."""
    variacao = (treino[f"y_h{h}"] - treino["log_casos_lag0"]).to_numpy()
    base = teste["log_casos_lag0"].to_numpy()
    limites = {}
    for nivel, (a_lo, a_hi) in NIVEIS.items():
        q_lo, q_hi = np.quantile(variacao, [a_lo, a_hi])
        limites[nivel] = (np.expm1(base + q_lo).clip(0), np.expm1(base + q_hi).clip(0))
    return limites


def season(d: pd.DataFrame, h: int, ano: int) -> dict:
    base = consolidated(d, h)
    alvo = base[f"data_alvo_h{h}"]
    inicio, fim = f"{ano}-01-01", f"{ano + 1}-01-01"
    teste = base[(base["data_iniSE"] >= inicio) & (alvo < fim)]
    if teste.empty:
        return {}
    treino = base[alvo < inicio]
    y = np.expm1(teste[f"y_h{h}"].to_numpy())
    persistencia = np.expm1(teste["log_casos_lag0"].to_numpy())

    # Ponto: modelo central treinado com tudo antes de Y
    prev = predict_central(fit_central(treino, h), teste)

    # Intervalos: quantis até Y-2, calibração em Y-1 (como em radar_dengue.intervals)
    ajuste = base[alvo < f"{ano - 1}-01-01"]
    calib = base[(alvo >= f"{ano - 1}-01-01") & (alvo < inicio)]
    modelos = fit_quantiles(ajuste, h)
    alvo_med = ajuste[f"y_h{h}"] - ajuste["log_casos_lag0"]
    mediana_m = lgb.LGBMRegressor(**PARAMS_Q, objective="quantile", alpha=0.5).fit(
        ajuste[FEATURES], alvo_med)
    mediana = np.expm1(mediana_m.predict(teste[FEATURES]) + teste["log_casos_lag0"]).clip(0)
    limites, cobertura = {}, {}
    for nivel in NIVEIS:
        q = conformal_adjustment(modelos, calib, h, nivel)
        lo, hi = predict_bounds(modelos, teste, nivel, q)
        limites[nivel] = (np.expm1(lo).clip(0), np.expm1(hi).clip(0))
        cobertura[str(nivel)] = float(np.mean((y >= limites[nivel][0]) & (y <= limites[nivel][1])))

    lim_base = baseline_bounds(treino, teste, h)
    return {
        "ano": ano, "horizonte": h, "n": len(teste), "casos_semana_media": float(y.mean()),
        "mae": float(np.mean(np.abs(y - prev))),
        "mae_baseline": float(np.mean(np.abs(y - persistencia))),
        "wis": float(wis(y, mediana, limites).mean()),
        "wis_baseline": float(wis(y, persistencia, lim_base).mean()),
        **{f"cobertura_{k}": v for k, v in cobertura.items()},
    }


def main() -> None:
    raw = pd.read_parquet(DATA_RAW / "infodengue_sp.parquet")
    d = add_targets(build_features(raw), HORIZONS)
    linhas = []
    for ano in ANOS:
        for h in HORIZONS:
            r = season(d, h, ano)
            if r:
                linhas.append(r)
                print(f"{ano} h={h}: MAE {r['mae']:.1f} (baseline {r['mae_baseline']:.1f}) | "
                      f"WIS {r['wis']:.1f} (baseline {r['wis_baseline']:.1f}) | "
                      f"cobertura 95% {r['cobertura_95']:.1%}")
    with open(MODELS / "backtest.json", "w", encoding="utf-8") as f:
        json.dump(linhas, f, indent=2)


if __name__ == "__main__":
    main()
