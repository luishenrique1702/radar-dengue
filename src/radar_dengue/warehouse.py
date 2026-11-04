"""Banco analítico em DuckDB e consultas SQL do projeto.

Uso:
    python -m radar_dengue.warehouse            # cria data/radar.duckdb e roda as consultas
    python -m radar_dengue.warehouse 04         # roda só a consulta que começa com "04"
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd

from radar_dengue.config import DATA_PROCESSED, DATA_RAW, ROOT
from radar_dengue.geo import municipality_table, size_class

DB_PATH = ROOT / "data" / "radar.duckdb"
SQL_DIR = ROOT / "sql"
RESULTS_DIR = ROOT / "reports" / "sql"


def build_tables(raw: pd.DataFrame, previsoes: pd.DataFrame | None = None,
                 previsoes_teste: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Monta as tabelas do banco a partir dos dados do pipeline."""
    pop_max = raw.groupby("geocode")["pop"].max().rename("pop_max")
    municipios = (municipality_table()
                  .merge(raw[["geocode", "municipio"]].drop_duplicates("geocode"), on="geocode")
                  .merge(pop_max, left_on="geocode", right_index=True))
    municipios["porte"] = size_class(municipios["pop_max"]).astype(str)
    municipios["regiao"] = municipios["regiao"].astype(str)

    casos = raw.rename(columns={"data_iniSE": "data_inise", "SE": "se"})[
        ["geocode", "data_inise", "se", "casos", "casos_est", "pop", "Rt", "nivel",
         "tempmed", "umidmed"]].rename(columns={"Rt": "rt"})

    tabelas = {"municipios": municipios, "casos_semanais": casos}
    if previsoes is not None:
        tabelas["previsoes"] = previsoes
    if previsoes_teste is not None:
        tabelas["previsoes_teste"] = previsoes_teste
    return tabelas


def connect(tabelas: dict[str, pd.DataFrame], path: str | Path = ":memory:"):
    """Cria (ou recria) o banco com as tabelas informadas."""
    con = duckdb.connect(str(path))
    for nome, df in tabelas.items():
        con.register("_tmp", df)
        con.execute(f"CREATE OR REPLACE TABLE {nome} AS SELECT * FROM _tmp")
        con.unregister("_tmp")
    return con


def queries(prefixo: str = "") -> list[Path]:
    return sorted(p for p in SQL_DIR.glob("*.sql") if p.name.startswith(prefixo))


def run_query(con, arquivo: Path) -> pd.DataFrame:
    return con.execute(arquivo.read_text(encoding="utf-8")).df()


def main() -> None:
    prefixo = sys.argv[1] if len(sys.argv) > 1 else ""
    raw = pd.read_parquet(DATA_RAW / "infodengue_sp.parquet")
    prev = pd.read_parquet(DATA_PROCESSED / "previsao_atual.parquet")
    teste = pd.read_parquet(DATA_PROCESSED / "previsoes_teste.parquet")
    con = connect(build_tables(raw, prev, teste), DB_PATH)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for arq in queries(prefixo):
        df = run_query(con, arq)
        df.to_csv(RESULTS_DIR / f"{arq.stem}.csv", index=False)
        print(f"\n== {arq.stem} ({len(df)} linhas)\n{df.head(8).to_string(index=False)}")
    con.close()


if __name__ == "__main__":
    main()
