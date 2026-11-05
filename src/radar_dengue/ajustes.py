"""Experimentos de pós-processamento, avaliados no backtest com origem móvel.

1. Gera, para cada temporada de 2019 a 2026 e cada horizonte, as previsões linha a linha do modelo
   central (LightGBM Poisson), da binomial negativa e dos modelos quantílicos, sempre treinados só
   com o passado (o mesmo esquema de `radar_dengue.backtest`). Ficam em
   data/processed/backtest_previsoes.parquet e só são recalculadas com `--refazer`.
2. Testa, em cima dessas previsões:
   - **época sem dengue:** puxar a previsão para a semana atual quando a incidência é baixa;
   - **ensemble:** média geométrica ponderada do LightGBM com a binomial negativa;
   - **calibração conformal adaptativa** dos intervalos (Gibbs e Candès, 2021).

Os ajustes são escolhidos nas temporadas de 2019 a 2022 e avaliados nas de 2023 a 2026, que incluem a
epidemia de 2024.

Uso:
    python -m radar_dengue.ajustes [--refazer]
"""

from __future__ import annotations

import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm

from radar_dengue.backtest import ANOS, wis
from radar_dengue.compare import NB_FEATURES, nb_features
from radar_dengue.config import DATA_PROCESSED, DATA_RAW, HORIZONS, MODELS
from radar_dengue.features import FEATURES, add_targets, build_features
from radar_dengue.geo import size_class
from radar_dengue.intervals import NIVEIS, PARAMS_Q, adjustment_by_size, adjustments_for, fit_quantiles
from radar_dengue.model import consolidated, fit_central, predict_central

ARQ = DATA_PROCESSED / "backtest_previsoes.parquet"
ANOS_AJUSTE = [2019, 2020, 2021, 2022]
ANOS_AVALIACAO = [2023, 2024, 2025, 2026]


def fit_nb(tr: pd.DataFrame, h: int):
    return sm.GLM(np.expm1(tr[f"y_h{h}"]).round(), sm.add_constant(tr[NB_FEATURES]),
                  family=sm.families.NegativeBinomial(alpha=0.3),
                  offset=tr["log_casos_lag0"].to_numpy()).fit()


def season_rows(d: pd.DataFrame, h: int, ano: int) -> pd.DataFrame:
    base = consolidated(d, h)
    alvo = base[f"data_alvo_h{h}"]
    inicio, fim = f"{ano}-01-01", f"{ano + 1}-01-01"
    teste = base[(base["data_iniSE"] >= inicio) & (alvo < fim)]
    treino = base[alvo < inicio]
    atual = teste["log_casos_lag0"].to_numpy()
    out = pd.DataFrame({
        "ano": ano, "horizonte": h, "geocode": teste["geocode"].values,
        "semana_base": teste["data_iniSE"].values, "data_alvo": teste[f"data_alvo_h{h}"].values,
        "pop": teste["pop"].values, "inc100k_lag0": teste["inc100k_lag0"].values,
        "real": np.expm1(teste[f"y_h{h}"]).round().values, "persistencia": np.expm1(atual),
        "lgbm": predict_central(fit_central(treino, h), teste),
        "nb": fit_nb(treino, h).predict(sm.add_constant(teste[NB_FEATURES]), offset=atual),
    })
    # quantis: ajuste até Y-2, calibração conformal por porte em Y-1 (como em produção)
    ajuste = base[alvo < f"{ano - 1}-01-01"]
    calib = base[(alvo >= f"{ano - 1}-01-01") & (alvo < inicio)]
    modelos = fit_quantiles(ajuste, h)
    var = ajuste[f"y_h{h}"] - ajuste["log_casos_lag0"]
    modelos[0.5] = lgb.LGBMRegressor(**PARAMS_Q, objective="quantile", alpha=0.5).fit(
        ajuste[FEATURES], var)
    for a, m in modelos.items():
        out[f"q{a}"] = m.predict(teste[FEATURES]) + atual  # em log(1 + casos)
    for nivel in NIVEIS:
        out[f"ajuste_{nivel}"] = adjustments_for(teste, adjustment_by_size(modelos, calib, h, nivel))
    return out


def generate(d: pd.DataFrame) -> pd.DataFrame:
    partes = []
    for ano in ANOS:
        for h in HORIZONS:
            partes.append(season_rows(d, h, ano))
            print(f"previsões {ano} h={h}: {len(partes[-1]):,} linhas", flush=True)
    p = pd.concat(partes, ignore_index=True)
    p["porte"] = size_class(p["pop"]).astype(str)
    return p


def mae(y, p) -> float:
    return float(np.mean(np.abs(y - p)))


def shrink(p: pd.DataFrame, limiar: float, peso: float, coluna: str = "lgbm") -> np.ndarray:
    """Abaixo de `limiar` casos por 100 mil, mistura (em log) a previsão com a semana atual."""
    baixo = p["inc100k_lag0"].to_numpy() < limiar
    lp = np.log1p(p[coluna].to_numpy())
    lpers = np.log1p(p["persistencia"].to_numpy())
    return np.expm1(np.where(baixo, peso * lp + (1 - peso) * lpers, lp))


def ensemble(p: pd.DataFrame, peso_lgbm: float) -> np.ndarray:
    return np.expm1(peso_lgbm * np.log1p(p["lgbm"]) + (1 - peso_lgbm) * np.log1p(p["nb"]))


def summary_by(p: pd.DataFrame, colunas: dict) -> pd.DataFrame:
    """Erro relativo ao baseline por ano (média dos horizontes)."""
    linhas = []
    for (ano, h), g in p.groupby(["ano", "horizonte"]):
        base = mae(g["real"], g["persistencia"])
        linhas.append({"ano": ano, "horizonte": h,
                       **{n: mae(g["real"], c.loc[g.index]) / base for n, c in colunas.items()}})
    return pd.DataFrame(linhas)


def point_experiments(p: pd.DataFrame) -> dict:
    ajuste = p[p["ano"].isin(ANOS_AJUSTE)]
    # 1. época sem dengue: grade de limiar e peso, escolhida em 2019–2022
    grade = []
    for limiar in [0.5, 1, 2, 5, 10]:
        for peso in [0.0, 0.25, 0.5, 0.75]:
            c = pd.Series(shrink(ajuste, limiar, peso), index=ajuste.index)
            grade.append({"limiar": limiar, "peso": peso,
                          "razao": summary_by(ajuste, {"x": c})["x"].mean()})
    grade = pd.DataFrame(grade)
    base_ajuste = summary_by(ajuste, {"x": ajuste["lgbm"]})["x"].mean()
    melhor_s = grade.loc[grade["razao"].idxmin()].to_dict()
    # 2. ensemble com a binomial negativa
    pesos = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    ens = {w: summary_by(ajuste, {"x": pd.Series(ensemble(ajuste, w), index=ajuste.index)})["x"].mean()
           for w in pesos}
    melhor_w = min(ens, key=ens.get)

    cols = {"LightGBM (atual)": p["lgbm"],
            "época sem dengue": pd.Series(shrink(p, melhor_s["limiar"], melhor_s["peso"]), index=p.index),
            "ensemble com NB": pd.Series(ensemble(p, melhor_w), index=p.index)}
    combinado = pd.Series(shrink(p.assign(ens=cols["ensemble com NB"]), melhor_s["limiar"],
                                 melhor_s["peso"], "ens"), index=p.index)
    cols["os dois"] = combinado
    res = summary_by(p, cols)
    aval = res[res["ano"].isin(ANOS_AVALIACAO)]
    # erro nos meses sem dengue (junho a outubro) e nas semanas de alta incidência
    meses = pd.to_datetime(p["data_alvo"]).dt.month
    inverno = p["ano"].isin(ANOS_AVALIACAO) & meses.between(6, 10)
    alta = p["ano"].isin(ANOS_AVALIACAO) & (p["inc100k_lag0"] >= 50)
    recortes = {}
    for nome, m in [("junho a outubro", inverno), ("incidência ≥ 50/100 mil", alta)]:
        g = p[m]
        recortes[nome] = {n: mae(g["real"], c[m]) / mae(g["real"], g["persistencia"])
                          for n, c in cols.items()}
    return {
        "epoca_sem_dengue": {"escolhido": melhor_s, "razao_ajuste_atual": base_ajuste,
                             "grade": grade.to_dict("records")},
        "ensemble": {"peso_lgbm": melhor_w, "razao_ajuste": ens},
        "avaliacao_por_ano": aval.groupby("ano")[list(cols)].mean().round(4).to_dict("index"),
        "avaliacao_por_horizonte": aval.groupby("horizonte")[list(cols)].mean().round(4).to_dict("index"),
        "avaliacao_media": aval[list(cols)].mean().round(4).to_dict(),
        "recortes": recortes,
    }


def interval_bounds(p: pd.DataFrame, nivel: int, extra) -> tuple[np.ndarray, np.ndarray]:
    lo_a, hi_a = NIVEIS[nivel]
    aj = p[f"ajuste_{nivel}"].to_numpy() + extra
    lo, hi = p[f"q{lo_a}"].to_numpy() - aj, p[f"q{hi_a}"].to_numpy() + aj
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    return np.expm1(lo).clip(0), np.expm1(hi).clip(0)


def adaptive_extra(g: pd.DataFrame, nivel: int, gama: float, estreita: bool = True) -> np.ndarray:
    """Conformal adaptativo (ACI) sobre o alargamento, em log, atualizado toda semana.

    Na semana t, o alargamento extra é e_t = e_{t-1} + gama × (taxa de erro observada − alfa). A
    taxa observada é a fração de municípios fora do intervalo nas previsões cujo alvo já está
    consolidado: com atraso de notificação, a previsão feita em s só é conferida em s + h + 4.
    Com `estreita`, o ajuste também pode ser negativo e estreitar intervalos largos demais.
    """
    alfa = 1 - nivel / 100
    h = int(g["horizonte"].iloc[0])
    semanas = np.sort(g["semana_base"].unique())
    lo0, hi0 = interval_bounds(g, nivel, 0.0)
    fora = pd.Series((g["real"].to_numpy() < lo0) | (g["real"].to_numpy() > hi0), index=g.index)
    extra_semana, e = {}, 0.0
    erros_por_semana = fora.groupby(g["semana_base"].values).mean()
    hist = []
    for s in semanas:
        extra_semana[s] = e
        hist.append(s)
        # retorno disponível: previsões feitas há pelo menos h + 4 semanas
        conferida = s - np.timedelta64(7 * (h + 4), "D")
        if conferida in erros_por_semana.index:
            # erro que o intervalo ajustado daquela semana teria tido
            m = g["semana_base"].to_numpy() == conferida
            lo, hi = interval_bounds(g[m], nivel, extra_semana.get(conferida, 0.0))
            taxa = float(np.mean((g["real"].to_numpy()[m] < lo) | (g["real"].to_numpy()[m] > hi)))
            e = e + gama * (taxa - alfa) if estreita else max(0.0, e + gama * (taxa - alfa))
    return g["semana_base"].map(extra_semana).to_numpy(dtype=float)


def interval_experiments(p: pd.DataFrame) -> dict:
    gamas = [0.0, 0.02, 0.05, 0.1, 0.2]
    linhas = []
    for (ano, h), g in p.groupby(["ano", "horizonte"]):
        y = g["real"].to_numpy()
        med = np.expm1(g["q0.5"].to_numpy()).clip(0)
        for gama in gamas:
            limites, cob = {}, {}
            for nivel in NIVEIS:
                extra = adaptive_extra(g, nivel, gama) if gama > 0 else 0.0
                lo, hi = interval_bounds(g, nivel, extra)
                limites[nivel] = (lo, hi)
                cob[nivel] = float(np.mean((y >= lo) & (y <= hi)))
            linhas.append({"ano": ano, "horizonte": h, "gama": gama,
                           "wis": float(wis(y, med, limites).mean()),
                           **{f"cobertura_{n}": v for n, v in cob.items()}})
        print(f"intervalos {ano} h={h}", flush=True)
    r = pd.DataFrame(linhas)
    # escolhe pelo desvio de cada temporada em relação ao nível nominal (e não pela média), para
    # premiar intervalos certos tanto nos anos de epidemia quanto nos anos tranquilos
    anual = (r[r["ano"].isin(ANOS_AJUSTE)].groupby(["gama", "ano"])
             [["cobertura_80", "cobertura_95"]].mean())
    desvio = (anual["cobertura_80"] - 0.8).abs() + (anual["cobertura_95"] - 0.95).abs()
    escolha = desvio.groupby("gama").mean()
    return {"gama_escolhido": float(escolha.idxmin()),
            "por_ano": r.groupby(["gama", "ano"])[["wis", "cobertura_80", "cobertura_95"]]
            .mean().round(4).reset_index().to_dict("records"),
            "avaliacao": r[r["ano"].isin(ANOS_AVALIACAO)].groupby("gama")[
                ["wis", "cobertura_50", "cobertura_80", "cobertura_95"]].mean().round(4)
            .reset_index().to_dict("records")}


def main() -> None:
    if "--refazer" in sys.argv or not ARQ.exists():
        raw = pd.read_parquet(DATA_RAW / "infodengue_sp.parquet")
        d = nb_features(add_targets(build_features(raw), HORIZONS))
        generate(d).to_parquet(ARQ, index=False)
    p = pd.read_parquet(ARQ)
    saida = {"pontual": point_experiments(p)} if "--so-intervalos" not in sys.argv else {}
    if saida:
        print(json.dumps(saida["pontual"]["avaliacao_por_ano"], indent=1, ensure_ascii=False))
        print("média 2023–2026:", saida["pontual"]["avaliacao_media"])
    saida["intervalos"] = interval_experiments(p)
    print(json.dumps(saida["intervalos"]["avaliacao"], indent=1))
    arq = MODELS / "ajustes.json"
    anterior = json.loads(arq.read_text(encoding="utf-8")) if arq.exists() else {}
    with open(arq, "w", encoding="utf-8") as f:
        json.dump({**anterior, **saida}, f, indent=2, ensure_ascii=False, default=float)


if __name__ == "__main__":
    main()
