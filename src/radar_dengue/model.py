"""Treino e avaliação dos modelos de previsão (um LightGBM por horizonte).

Uso:
    python -m radar_dengue.model
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from radar_dengue.config import DATA_PROCESSED, DATA_RAW, HORIZONS, MODELS, TEST_START
from radar_dengue.features import FEATURES, add_targets, build_features, risk_level
from radar_dengue.validate import validate_raw

QUANTILES = {"q10": 0.1, "q90": 0.9}
REGISTRY = MODELS / "registro"

PARAMS = dict(
    n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=50,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    random_state=42, verbose=-1,
)


def consolidated(d: pd.DataFrame, h: int) -> pd.DataFrame:
    """Linhas com alvo disponível e já consolidado.

    As últimas 4 semanas ainda recebem notificações atrasadas e não servem como "verdade".
    """
    y, data_alvo = f"y_h{h}", f"data_alvo_h{h}"
    base = d.dropna(subset=[y, "log_casos_lag52"])
    limite = d["data_iniSE"].max() - pd.Timedelta(weeks=4)
    return base[base[data_alvo] <= limite]


def split(d: pd.DataFrame, h: int):
    """Divisão temporal: treino com alvos antes de 2024; teste com dados de 2024 em diante."""
    data_alvo = f"data_alvo_h{h}"
    base = consolidated(d, h)
    train = base[base[data_alvo] < TEST_START]
    test = base[base["data_iniSE"] >= TEST_START]
    return train, test


def metrics(y_true_log, y_pred_log, pop) -> dict:
    casos, prev = np.expm1(y_true_log), np.expm1(y_pred_log).clip(0)
    nivel_real = risk_level(casos / pop * 1e5)
    nivel_prev = risk_level(prev / pop * 1e5)
    alto_real = np.isin(nivel_real, ["Alto", "Muito alto"])
    alto_prev = np.isin(nivel_prev, ["Alto", "Muito alto"])
    tp = (alto_real & alto_prev).sum()
    return {
        "mae_casos": float(np.mean(np.abs(casos - prev))),
        "rmse_log": float(np.sqrt(np.mean((y_true_log - y_pred_log) ** 2))),
        "acerto_nivel": float(np.mean(nivel_real == nivel_prev)),
        "recall_alerta": float(tp / max(alto_real.sum(), 1)),
        "precisao_alerta": float(tp / max(alto_prev.sum(), 1)),
    }


def main() -> None:
    raw = pd.read_parquet(DATA_RAW / "infodengue_sp.parquet")
    validate_raw(raw)
    d = add_targets(build_features(raw), HORIZONS)
    MODELS.mkdir(exist_ok=True)

    resultados, previsoes_teste, importancias = {}, [], {}
    for h in HORIZONS:
        train, test = split(d, h)
        model = lgb.LGBMRegressor(**PARAMS)
        # Alvo = variação em relação à semana atual. Isso permite ao modelo
        # acompanhar epidemias maiores que qualquer uma vista no treino.
        model.fit(train[FEATURES], train[f"y_h{h}"] - train["log_casos_lag0"])
        pred = model.predict(test[FEATURES]) + test["log_casos_lag0"].to_numpy()

        # Baseline ingênuo: "daqui a h semanas, os casos serão iguais aos de hoje"
        baseline = test["log_casos_lag0"].to_numpy()
        m_model = metrics(test[f"y_h{h}"].to_numpy(), pred, test["pop"].to_numpy())
        m_base = metrics(test[f"y_h{h}"].to_numpy(), baseline, test["pop"].to_numpy())
        resultados[f"h{h}"] = {"modelo": m_model, "baseline": m_base,
                               "n_treino": len(train), "n_teste": len(test)}
        print(f"h={h}: MAE modelo {m_model['mae_casos']:.2f} | baseline "
              f"{m_base['mae_casos']:.2f} | recall alerta {m_model['recall_alerta']:.2f}")

        previsoes_teste.append(pd.DataFrame({
            "geocode": test["geocode"].values, "municipio": test["municipio"].values,
            "semana_base": test["data_iniSE"].values,
            "data_alvo": test[f"data_alvo_h{h}"].values, "horizonte": h,
            "casos_reais": np.expm1(test[f"y_h{h}"]).round().values,
            "casos_previstos": np.expm1(pred).clip(0).round(1),
            "baseline": np.expm1(baseline).round(),
        }))
        ganho = model.booster_.feature_importance("gain").tolist()
        importancias[f"h{h}"] = dict(zip(FEATURES, ganho, strict=True))

        # Modelos finais para uso no dashboard: retreinados com todos os dados consolidados.
        # Além da previsão central, dois modelos quantílicos formam o intervalo de 80%.
        full = consolidated(d, h)
        alvo = full[f"y_h{h}"] - full["log_casos_lag0"]
        joblib.dump(lgb.LGBMRegressor(**PARAMS).fit(full[FEATURES], alvo),
                    MODELS / f"lgbm_h{h}.joblib")
        for nome, a in QUANTILES.items():
            mq = lgb.LGBMRegressor(**PARAMS, objective="quantile", alpha=a)
            joblib.dump(mq.fit(full[FEATURES], alvo), MODELS / f"lgbm_h{h}_{nome}.joblib")

    DATA_PROCESSED.mkdir(exist_ok=True)
    pd.concat(previsoes_teste).to_parquet(DATA_PROCESSED / "previsoes_teste.parquet", index=False)
    with open(MODELS / "metricas.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    with open(MODELS / "importancias.json", "w", encoding="utf-8") as f:
        json.dump(importancias, f, indent=2)
    register(resultados, raw)


def register(resultados: dict, raw: pd.DataFrame) -> None:
    """Registra a versão treinada: data, período dos dados e métricas de teste.

    Cada retreino vira uma linha em models/registro/versoes.csv. Assim dá para comparar versões
    e perceber se um retreino piorou o modelo antes de usá-lo.
    """
    REGISTRY.mkdir(parents=True, exist_ok=True)
    linha = {"versao": datetime.now(UTC).strftime("%Y%m%d-%H%M"),
             "dados_ate": raw["data_iniSE"].max().date().isoformat(),
             "parametros": json.dumps(PARAMS, sort_keys=True)}
    for h, r in resultados.items():
        linha[f"mae_{h}"] = round(r["modelo"]["mae_casos"], 3)
        linha[f"mae_baseline_{h}"] = round(r["baseline"]["mae_casos"], 3)
    arq = REGISTRY / "versoes.csv"
    hist = pd.read_csv(arq) if arq.exists() else pd.DataFrame()
    pd.concat([hist, pd.DataFrame([linha])], ignore_index=True).to_csv(arq, index=False)
    print(f"Versão {linha['versao']} registrada em {arq}")


if __name__ == "__main__":
    main()
