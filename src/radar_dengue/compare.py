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
from radar_dengue.model import PARAMS, fit_central, predict_central, split


def mae(real, prev) -> float:
    return float(np.mean(np.abs(np.asarray(real) - np.clip(np.asarray(prev, float), 0, None))))


# Binomial negativa com atributos invariantes à escala (variações, não níveis). Com níveis de
# log(casos) como atributos, ela "explodia" nas cidades grandes durante a epidemia de 2024.
NB_FEATURES = ["d1", "dm4", "dm8", "crescimento_4s", "sem_sin", "sem_cos", "cresc_estado_4s",
               "log_rt", "log_inc", "tempmed_media4", "umidmed_media4"]


def nb_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["d1"] = d["log_casos_lag0"] - d["log_casos_lag1"]
    d["dm4"] = d["log_casos_lag0"] - d["log_media4"]
    d["dm8"] = d["log_casos_lag0"] - d["log_media8"]
    d["log_rt"] = np.log(d["Rt"].clip(0.1, 5))
    d["log_inc"] = np.log1p(d["inc100k_lag0"])
    return d


def main() -> None:
    raw = pd.read_parquet(DATA_RAW / "infodengue_sp.parquet")
    d = nb_features(add_targets(build_features(raw), HORIZONS))
    resultado = {}
    for h in HORIZONS:
        tr, te = split(d, h)
        y = f"y_h{h}"
        real = np.expm1(te[y]).to_numpy()
        atual = te["log_casos_lag0"].to_numpy()
        lin = sm.OLS(tr[y] - tr["log_casos_lag0"], sm.add_constant(tr[FEATURES].fillna(0))).fit()
        nb = sm.GLM(np.expm1(tr[y]).round(), sm.add_constant(tr[NB_FEATURES]),
                    family=sm.families.NegativeBinomial(alpha=0.3),
                    offset=tr["log_casos_lag0"].to_numpy()).fit()
        gbm_log = lgb.LGBMRegressor(**PARAMS).fit(tr[FEATURES], tr[y] - tr["log_casos_lag0"])
        resultado[f"h{h}"] = {
            "Repetir o mesmo período do ano anterior": mae(real, np.expm1(te["log_casos_lag52"])),
            "Regressão linear": mae(
                real, np.expm1(lin.predict(sm.add_constant(te[FEATURES].fillna(0))) + atual)),
            "Repetir a semana atual (baseline)": mae(real, np.expm1(atual)),
            "Binomial negativa": mae(
                real, nb.predict(sm.add_constant(te[NB_FEATURES]), offset=atual)),
            "LightGBM em escala log": mae(real, np.expm1(gbm_log.predict(te[FEATURES]) + atual)),
            "LightGBM Poisson (modelo escolhido)": mae(real, predict_central(fit_central(tr, h), te)),
        }
        print(h, {k: round(v, 1) for k, v in resultado[f"h{h}"].items()})
    with open(MODELS / "comparacao.json", "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    alerta = alert_comparison(d)
    with open(MODELS / "alerta_infodengue.json", "w", encoding="utf-8") as f:
        json.dump(alerta, f, indent=2, ensure_ascii=False)
    for h, r in alerta.items():
        print(h, "recall oficial", round(r["oficial"]["recall"], 3), "-> combinado",
              round(r["combinado"]["recall"], 3), "| precisão", round(r["oficial"]["precisao"], 3),
              "->", round(r["combinado"]["precisao"], 3))


def alert_comparison(d: pd.DataFrame) -> dict:
    """Compara o modelo com o alerta oficial do InfoDengue.

    Evento: o município estar no nível 4 do InfoDengue (incidência acima do limiar epidêmico
    do próprio município) daqui a h semanas. O alerta oficial "de hoje" é comparado com o
    modelo e com a combinação dos dois: alerta oficial + alerta do modelo para municípios que
    ainda não estão em nível 4.
    """
    g = d.groupby("geocode")
    for h in HORIZONS:
        d[f"nivel_h{h}"] = g["nivel"].shift(-h)
    saida = {}
    for h in HORIZONS:
        tr, te = split(d, h)
        modelo = fit_central(tr, h)
        inc_tr = predict_central(modelo, tr) / tr["pop"].to_numpy() * 1e5
        inc_te = predict_central(modelo, te) / te["pop"].to_numpy() * 1e5
        ev_tr, ev_te = (tr[f"nivel_h{h}"] == 4).to_numpy(), (te[f"nivel_h{h}"] == 4).to_numpy()
        fora_tr, fora_te = (tr["nivel"] != 4).to_numpy(), (te["nivel"] != 4).to_numpy()

        # limiar de incidência prevista escolhido no treino, para alertar novas epidemias
        def f1(a, ev):
            tp = (a & ev).sum()
            p, r = tp / max(a.sum(), 1), tp / max(ev.sum(), 1)
            return 2 * p * r / max(p + r, 1e-9)
        limiar = max([10, 20, 30, 50, 75, 100, 150],
                     key=lambda t: f1((inc_tr >= t) & fora_tr, ev_tr & fora_tr))

        oficial = ~fora_te
        extra = (inc_te >= limiar) & fora_te
        comb = oficial | extra
        novos = ev_te & fora_te

        def pr(a, ev=ev_te):
            tp = (a & ev).sum()
            return {"precisao": float(tp / max(a.sum(), 1)), "recall": float(tp / max(ev.sum(), 1)),
                    "alertas": int(a.sum())}
        saida[f"h{h}"] = {
            "eventos": int(ev_te.sum()), "limiar_incidencia": limiar,
            "oficial": pr(oficial), "combinado": pr(comb),
            "novas_epidemias": int(novos.sum()),
            "novas_detectadas_pelo_modelo": float((extra & novos).sum() / max(novos.sum(), 1)),
            "precisao_alertas_novos": float((extra & ev_te).sum() / max(extra.sum(), 1)),
        }
    return saida


def sarimax_comparison(d: pd.DataFrame) -> pd.DataFrame:
    """Compara SARIMAX, binomial negativa, LightGBM e baseline nas mesmas cidades e semanas.

    Usa as previsões salvas por `python -m radar_dengue.sarimax` (cidades com mais de 100 mil
    habitantes) e restringe os outros métodos exatamente às mesmas linhas.
    """
    arquivos = {"SARIMAX (sazonalidade de Fourier)": MODELS / "sarimax.parquet",
                "SARIMAX com clima": MODELS / "sarimax_clima.parquet"}
    sar = {nome: pd.read_parquet(arq) for nome, arq in arquivos.items() if arq.exists()}
    if not sar:
        return pd.DataFrame()
    linhas = []
    for h in HORIZONS:
        tr, te = split(d, h)
        cidades = set.intersection(*(set(s["geocode"]) for s in sar.values()))
        te = te[te["geocode"].isin(cidades)].copy()
        y = f"y_h{h}"
        te["real"] = np.expm1(te[y])
        te["LightGBM Poisson (modelo escolhido)"] = predict_central(fit_central(tr, h), te)
        nb = sm.GLM(np.expm1(tr[y]).round(), sm.add_constant(tr[NB_FEATURES]),
                    family=sm.families.NegativeBinomial(alpha=0.3),
                    offset=tr["log_casos_lag0"].to_numpy()).fit()
        te["Binomial negativa"] = nb.predict(sm.add_constant(te[NB_FEATURES]),
                                             offset=te["log_casos_lag0"].to_numpy())
        te["Repetir a semana atual (baseline)"] = np.expm1(te["log_casos_lag0"])
        for nome, s in sar.items():
            s = s[s["horizonte"] == h][["geocode", "data_iniSE", "previsto"]]
            te = te.merge(s.rename(columns={"previsto": nome}), on=["geocode", "data_iniSE"],
                          how="inner")
        metodos = ["Repetir a semana atual (baseline)", *sar, "Binomial negativa",
                   "LightGBM Poisson (modelo escolhido)"]
        for m in metodos:
            linhas.append({"horizonte": h, "método": m, "MAE": mae(te["real"], te[m]),
                           "linhas": len(te), "cidades": te["geocode"].nunique()})
    return pd.DataFrame(linhas)


if __name__ == "__main__":
    main()
