"""Experimento: os casos dos municípios vizinhos e a tendência do Rt ajudam a prever o início
das epidemias?

Compara quatro conjuntos de atributos na validação temporal (treino até o ano anterior,
validação em 2021, 2022 e 2023), sem usar o período de teste. Além do erro geral, mede o que
motivou o experimento: as semanas de início de epidemia, em que o município está abaixo de 50
casos por 100 mil habitantes e, h semanas depois, passa desse nível. Nessas semanas o modelo
atual subestima os casos.

Uso:
    python -m radar_dengue.experimentos
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from radar_dengue.config import DATA_RAW, HORIZONS, MODELS
from radar_dengue.features import (
    FEATURES_BASE,
    FEATURES_RT,
    FEATURES_VIZINHOS,
    add_targets,
    build_features,
)
from radar_dengue.model import consolidated, fit_central, predict_central
from radar_dengue.tuning import ANOS_VALIDACAO

CONJUNTOS = {
    "atual": FEATURES_BASE,
    "+ tendência do Rt": FEATURES_BASE + FEATURES_RT,
    "+ vizinhos": FEATURES_BASE + FEATURES_VIZINHOS,
    "+ vizinhos e Rt": FEATURES_BASE + FEATURES_VIZINHOS + FEATURES_RT,
}
LIMIAR_EPIDEMIA = 50  # casos por 100 mil habitantes na semana ("Alto" no dashboard)


def evaluate(d: pd.DataFrame, features: list[str]) -> list[dict]:
    linhas = []
    for h in HORIZONS:
        base = consolidated(d, h)
        alvo = base[f"data_alvo_h{h}"]
        for ano in ANOS_VALIDACAO:
            tr = base[alvo < f"{ano}-01-01"]
            va = base[(base["data_iniSE"] >= f"{ano}-01-01") & (alvo < f"{ano + 1}-01-01")]
            y = np.expm1(va[f"y_h{h}"].to_numpy())
            prev = predict_central(fit_central(tr, h, features=features), va)
            pop = va["pop"].to_numpy()
            inicio = (va["inc100k_lag0"].to_numpy() < LIMIAR_EPIDEMIA) & \
                     (y / pop * 1e5 >= LIMIAR_EPIDEMIA)
            linhas.append({
                "horizonte": h, "ano": ano,
                "mae": float(np.mean(np.abs(y - prev))),
                "mae_inicio": float(np.mean(np.abs(y[inicio] - prev[inicio]))),
                "vies_inicio": float(np.mean(prev[inicio] - y[inicio])),
                "inicios_antecipados": float(np.mean(prev[inicio] / pop[inicio] * 1e5
                                                     >= LIMIAR_EPIDEMIA)),
                "n_inicio": int(inicio.sum()),
            })
    return linhas


def main() -> None:
    raw = pd.read_parquet(DATA_RAW / "infodengue_sp.parquet")
    d = add_targets(build_features(raw), HORIZONS)
    saida = {}
    for nome, feats in CONJUNTOS.items():
        r = pd.DataFrame(evaluate(d, feats))
        resumo = r.groupby("horizonte")[["mae", "mae_inicio", "vies_inicio",
                                         "inicios_antecipados"]].mean()
        saida[nome] = {f"h{h}": v for h, v in resumo.round(4).to_dict("index").items()}
        print(nome)
        print(resumo.round(3).to_string())
    with open(MODELS / "experimento_vizinhos.json", "w", encoding="utf-8") as f:
        json.dump(saida, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
