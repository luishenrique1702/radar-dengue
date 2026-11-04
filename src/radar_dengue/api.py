"""API de previsões do Radar Dengue.

Rodar localmente:
    uvicorn radar_dengue.api:app --reload
Documentação interativa em http://localhost:8000/docs
"""

from __future__ import annotations

import json
import unicodedata
from functools import lru_cache
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from radar_dengue.config import DATA_PROCESSED, MODELS

NivelRisco = Literal["Baixo", "Moderado", "Alto", "Muito alto"]

app = FastAPI(
    title="Radar Dengue SP",
    version="1.0",
    description=(
        "Previsão semanal de casos de dengue para os 645 municípios do estado de São Paulo, "
        "de 1 a 4 semanas à frente, com intervalos de previsão calibrados de 50%, 80% e 95%. "
        "Fonte dos dados: InfoDengue (Fiocruz/FGV)."
    ),
)


class Intervalo(BaseModel):
    nivel: int = Field(description="Probabilidade nominal do intervalo, em %")
    minimo: float
    maximo: float


class Previsao(BaseModel):
    horizonte: int = Field(description="Semanas à frente")
    semana: str = Field(description="Início da semana prevista (AAAA-MM-DD)")
    casos_previstos: float
    incidencia_100k: float
    nivel_risco: NivelRisco
    intervalos: list[Intervalo]


class PrevisaoMunicipio(BaseModel):
    geocode: str
    municipio: str
    populacao: int
    semana_base: str = Field(description="Última semana com dados usada na previsão")
    previsoes: list[Previsao]


class Alerta(BaseModel):
    geocode: str
    municipio: str
    semana: str
    casos_previstos: float
    incidencia_100k: float
    nivel_risco: NivelRisco


class Municipio(BaseModel):
    geocode: str
    municipio: str


def _normalizar(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return sem_acento.lower().strip()


@lru_cache(maxsize=1)
def _previsoes() -> pd.DataFrame:
    p = pd.read_parquet(DATA_PROCESSED / "previsao_atual.parquet")
    p["geocode"] = p["geocode"].astype(str)
    p["_nome"] = p["municipio"].map(_normalizar)
    return p


def _encontrar(municipio: str) -> pd.DataFrame:
    p = _previsoes()
    chave = _normalizar(municipio)
    linhas = p[(p["geocode"] == municipio) | (p["_nome"] == chave)]
    if linhas.empty:
        raise HTTPException(404, f"Município não encontrado: {municipio}. "
                                 "Use o nome (ex.: Campinas) ou o código IBGE (ex.: 3509502).")
    return linhas.sort_values("horizonte")


@app.get("/saude", summary="Verifica se a API está no ar")
def saude() -> dict:
    p = _previsoes()
    return {"status": "ok", "semana_base": str(pd.Timestamp(p["semana_base"].iloc[0]).date()),
            "municipios": int(p["geocode"].nunique())}


@app.get("/municipios", response_model=list[Municipio], summary="Lista ou busca municípios")
def municipios(busca: str | None = Query(None, description="Trecho do nome, sem acento")):
    p = _previsoes().drop_duplicates("geocode")
    if busca:
        p = p[p["_nome"].str.contains(_normalizar(busca), regex=False)]
    return [Municipio(geocode=g, municipio=m) for g, m in zip(p["geocode"], p["municipio"],
                                                                strict=True)]


@app.get("/previsao/{municipio}", response_model=PrevisaoMunicipio,
         summary="Previsão de 1 a 4 semanas para um município")
def previsao(municipio: str):
    linhas = _encontrar(municipio)
    primeira = linhas.iloc[0]
    itens = []
    for _, r in linhas.iterrows():
        intervalos = [Intervalo(nivel=n, minimo=float(r[f"casos_min_{n}"]),
                                maximo=float(r[f"casos_max_{n}"]))
                      for n in (50, 80, 95) if f"casos_min_{n}" in r]
        itens.append(Previsao(
            horizonte=int(r["horizonte"]), semana=str(pd.Timestamp(r["data_alvo"]).date()),
            casos_previstos=float(r["casos_previstos"]),
            incidencia_100k=float(r["incidencia_100k"]), nivel_risco=r["nivel_risco"],
            intervalos=intervalos))
    return PrevisaoMunicipio(
        geocode=primeira["geocode"], municipio=primeira["municipio"],
        populacao=int(primeira["pop"]),
        semana_base=str(pd.Timestamp(primeira["semana_base"]).date()), previsoes=itens)


@app.get("/alertas", response_model=list[Alerta],
         summary="Municípios com risco alto ou muito alto previsto")
def alertas(horizonte: int = Query(2, ge=1, le=4, description="Semanas à frente"),
            casos_minimos: float = Query(5, ge=0, description="Ignora municípios com menos casos "
                                                             "previstos que isso")):
    p = _previsoes()
    p = p[(p["horizonte"] == horizonte) & p["nivel_risco"].isin(["Alto", "Muito alto"])
          & (p["casos_previstos"] >= casos_minimos)].sort_values("incidencia_100k",
                                                                 ascending=False)
    return [Alerta(geocode=r["geocode"], municipio=r["municipio"],
                   semana=str(pd.Timestamp(r["data_alvo"]).date()),
                   casos_previstos=float(r["casos_previstos"]),
                   incidencia_100k=float(r["incidencia_100k"]), nivel_risco=r["nivel_risco"])
            for _, r in p.iterrows()]


@app.get("/modelo", summary="Desempenho do modelo no período de teste (2024–2026)")
def modelo() -> dict:
    metricas = json.loads((MODELS / "metricas.json").read_text(encoding="utf-8"))
    return {
        "modelo": "LightGBM com objetivo Poisson, um por horizonte",
        "erro_medio_casos": {h: {"modelo": round(v["modelo"]["mae_casos"], 2),
                                 "baseline": round(v["baseline"]["mae_casos"], 2)}
                             for h, v in metricas.items()},
    }
