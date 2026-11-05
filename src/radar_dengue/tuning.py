"""Ajuste de hiperparâmetros do modelo central, com o mesmo modelo usado em produção.

Cada configuração é avaliada com o objetivo Poisson e a semana atual como offset, nos quatro
horizontes, por validação temporal dentro do período de treino: treina com alvos até o fim do
ano anterior e valida em 2021, 2022 e 2023. O período de teste (2024+) nunca entra na escolha.

A nota de cada configuração é a razão entre o seu erro absoluto médio e o do baseline (repetir a
semana atual), na média dos anos e horizontes. A razão evita que o horizonte de 4 semanas, que
tem erros maiores em casos, domine a escolha.

Uso:
    python -m radar_dengue.tuning
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from radar_dengue.config import DATA_RAW, HORIZONS, MODELS
from radar_dengue.features import add_targets, build_features
from radar_dengue.model import consolidated, fit_central, predict_central

GRADE = [
    dict(num_leaves=15, min_child_samples=200, learning_rate=0.05, n_estimators=500),
    dict(num_leaves=31, min_child_samples=100, learning_rate=0.05, n_estimators=400),
    dict(num_leaves=63, min_child_samples=50, learning_rate=0.03, n_estimators=600),
    dict(num_leaves=63, min_child_samples=200, learning_rate=0.02, n_estimators=1000),
    dict(num_leaves=127, min_child_samples=50, learning_rate=0.03, n_estimators=600),
    dict(num_leaves=127, min_child_samples=100, learning_rate=0.02, n_estimators=1000),
    dict(num_leaves=255, min_child_samples=100, learning_rate=0.03, n_estimators=400),
    dict(num_leaves=255, min_child_samples=200, learning_rate=0.03, n_estimators=600),
    dict(num_leaves=511, min_child_samples=100, learning_rate=0.03, n_estimators=400),
]
ANOS_VALIDACAO = [2021, 2022, 2023]


def validation_errors(d: pd.DataFrame, params: dict) -> list[dict]:
    """MAE do modelo e do baseline em cada ano de validação e horizonte."""
    linhas = []
    for h in HORIZONS:
        base = consolidated(d, h)
        alvo = base[f"data_alvo_h{h}"]
        for ano in ANOS_VALIDACAO:
            tr = base[alvo < f"{ano}-01-01"]
            va = base[(base["data_iniSE"] >= f"{ano}-01-01") & (alvo < f"{ano + 1}-01-01")]
            casos = np.expm1(va[f"y_h{h}"].to_numpy())
            prev = predict_central(fit_central(tr, h, params), va)
            linhas.append({
                "horizonte": h, "ano": ano,
                "mae": float(np.mean(np.abs(casos - prev))),
                "mae_baseline": float(np.mean(np.abs(casos - np.expm1(va["log_casos_lag0"])))),
            })
    return linhas


def main() -> None:
    raw = pd.read_parquet(DATA_RAW / "infodengue_sp.parquet")
    d = add_targets(build_features(raw), HORIZONS)
    arq = MODELS / "tuning.json"
    anterior = json.load(open(arq, encoding="utf-8")) if arq.exists() else []
    saida = [r for r in anterior if isinstance(r, dict) and r.get("params") in GRADE]
    feitos = [r["params"] for r in saida]
    for params in [p for p in GRADE if p not in feitos]:
        erros = pd.DataFrame(validation_errors(d, params))
        razao = float((erros["mae"] / erros["mae_baseline"]).mean())
        saida.append({"params": params, "razao_baseline": razao,
                      "mae_por_horizonte": erros.groupby("horizonte")["mae"].mean().round(3)
                      .rename(lambda h: f"h{h}").to_dict()})
        print(f"{params} -> {razao:.4f}")
    saida.sort(key=lambda r: r["razao_baseline"])
    with open(arq, "w", encoding="utf-8") as f:
        json.dump(saida, f, indent=2)
    print(f"Melhor: {saida[0]['params']}")


if __name__ == "__main__":
    main()
