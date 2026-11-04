"""Monitoramento do modelo em produção.

Toda previsão publicada é guardada. Quando as semanas previstas ficam consolidadas
(4 semanas depois, por causa do atraso de notificação), comparamos o previsto com o real.

Uso:
    python -m radar_dengue.monitor            # avalia e mostra o desempenho recente
    python -m radar_dengue.monitor --checar   # sai com erro se o modelo estiver pior que o baseline
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from radar_dengue.config import DATA_PROCESSED, DATA_RAW, ROOT

MONITOR = ROOT / "data" / "monitoring"
LOG = MONITOR / "previsoes_log.parquet"
DESEMPENHO = MONITOR / "desempenho.parquet"
CHAVE = ["semana_base", "geocode", "horizonte"]
COLS = CHAVE + ["data_alvo", "casos_previstos", "casos_min", "casos_max", "origem"]


def log_forecast(prev: pd.DataFrame, origem: str = "produção") -> None:
    """Acrescenta a previsão ao histórico, substituindo uma execução anterior da mesma semana."""
    MONITOR.mkdir(parents=True, exist_ok=True)
    novo = prev.assign(origem=origem).reindex(columns=COLS)
    if LOG.exists():
        novo = pd.concat([pd.read_parquet(LOG), novo], ignore_index=True)
    novo.drop_duplicates(CHAVE, keep="last").to_parquet(LOG, index=False)


def seed_from_backtest(semanas: int = 26) -> None:
    """Preenche o histórico com as previsões do período de teste, marcadas como retroativas.

    Na primeira publicação ainda não há previsões "ao vivo" consolidadas. As do período de teste
    foram feitas por modelos que não viram aquelas semanas, então servem como ponto de partida
    honesto, desde que identificadas como retroativas.
    """
    teste = pd.read_parquet(DATA_PROCESSED / "previsoes_teste.parquet")
    inicio = teste["semana_base"].max() - pd.Timedelta(weeks=semanas)
    teste = teste[teste["semana_base"] > inicio]
    log_forecast(teste[["semana_base", "geocode", "horizonte", "data_alvo", "casos_previstos"]],
                 origem="retroativo")


def evaluate(raw: pd.DataFrame) -> pd.DataFrame:
    """Desempenho por semana de previsão e horizonte, só com semanas-alvo já consolidadas."""
    log = pd.read_parquet(LOG)
    limite = raw["data_iniSE"].max() - pd.Timedelta(weeks=4)
    real = raw[["geocode", "data_iniSE", "casos"]]
    m = log[log["data_alvo"] <= limite].merge(
        real.rename(columns={"data_iniSE": "data_alvo", "casos": "real"}),
        on=["geocode", "data_alvo"], how="inner",
    ).merge(
        real.rename(columns={"data_iniSE": "semana_base", "casos": "baseline"}),
        on=["geocode", "semana_base"], how="left",
    )
    m["erro_modelo"] = (m["real"] - m["casos_previstos"]).abs()
    m["erro_baseline"] = (m["real"] - m["baseline"]).abs()
    m["dentro"] = np.where(m["casos_min"].notna(),
                           (m["real"] >= m["casos_min"]) & (m["real"] <= m["casos_max"]), np.nan)
    out = (m.groupby(["semana_base", "horizonte", "origem"])
           .agg(mae_modelo=("erro_modelo", "mean"), mae_baseline=("erro_baseline", "mean"),
                cobertura=("dentro", "mean"), municipios=("geocode", "nunique"))
           .reset_index())
    out["razao"] = out["mae_modelo"] / out["mae_baseline"]
    out.to_parquet(DESEMPENHO, index=False)
    return out


def check(desempenho: pd.DataFrame, horizonte: int = 4, semanas: int = 4,
          tolerancia: float = 1.2) -> bool:
    """True se o modelo esteve pior que o baseline (acima da tolerância) nas últimas semanas."""
    d = desempenho[desempenho["horizonte"] == horizonte].sort_values("semana_base").tail(semanas)
    return len(d) == semanas and bool((d["razao"] > tolerancia).all())


def main() -> None:
    raw = pd.read_parquet(DATA_RAW / "infodengue_sp.parquet")
    if not LOG.exists() or not (pd.read_parquet(LOG)["origem"] == "retroativo").any():
        seed_from_backtest()
    des = evaluate(raw)
    recentes = des.sort_values("semana_base").groupby("horizonte").tail(4)
    print(recentes[["semana_base", "horizonte", "mae_modelo", "mae_baseline", "origem"]]
          .round({"mae_modelo": 2, "mae_baseline": 2}).to_string(index=False))
    if check(des):
        print("ALERTA: modelo pior que o baseline por 4 semanas seguidas. Investigar e retreinar.")
        if "--checar" in sys.argv:
            sys.exit(1)


if __name__ == "__main__":
    main()
