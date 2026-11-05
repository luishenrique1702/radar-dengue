"""Atributos geográficos dos municípios: centróide, região aproximada e porte."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from radar_dengue.config import GEOJSON

PORTES = ["<20 mil", "20–100 mil", "100–500 mil", ">500 mil"]
REGIOES = ["Oeste", "Centro", "Leste"]


def _centroid(geometry: dict) -> tuple[float, float]:
    pts: list = []

    def walk(c):
        if isinstance(c[0], (int, float)):
            pts.append(c)
        else:
            for x in c:
                walk(x)

    walk(geometry["coordinates"])
    a = np.asarray(pts)
    return float(a[:, 0].mean()), float(a[:, 1].mean())


def municipality_table() -> pd.DataFrame:
    """Código IBGE, longitude e latitude do centróide e região aproximada.

    A região é uma aproximação por faixas de longitude (oeste < -50°, leste > -48°),
    suficiente para comparar o interior oeste com o leste (capital, Vale do Paraíba e litoral).
    """
    with open(GEOJSON, encoding="utf-8") as f:
        geo = json.load(f)
    rows = [(f["properties"]["id"], *_centroid(f["geometry"])) for f in geo["features"]]
    t = pd.DataFrame(rows, columns=["geocode", "lon", "lat"])
    t["regiao"] = pd.cut(t["lon"], [-180, -50, -48, 180], labels=REGIOES)
    return t


def _vertices(coords) -> set[tuple[float, float]]:
    """Vértices de um polígono (ou multipolígono), arredondados a 4 casas decimais."""
    if isinstance(coords[0], (int, float)):
        return {(round(coords[0], 4), round(coords[1], 4))}
    return set().union(*(_vertices(c) for c in coords))


def neighbors(min_vertices: int = 2) -> dict[str, list[str]]:
    """Municípios que fazem fronteira com cada município.

    Dois municípios são vizinhos quando compartilham pelo menos `min_vertices` vértices da
    malha do IBGE (com 4 casas decimais, cerca de 10 m). Um vértice só em comum é um encontro
    de pontas, e não uma fronteira. A mediana é de 5 vizinhos; só Ilhabela não tem nenhum.
    """
    with open(GEOJSON, encoding="utf-8") as f:
        geo = json.load(f)
    por_vertice: dict[tuple, set] = {}
    for feat in geo["features"]:
        for p in _vertices(feat["geometry"]["coordinates"]):
            por_vertice.setdefault(p, set()).add(feat["properties"]["id"])
    contagem: dict[str, dict[str, int]] = {}
    for municipios in por_vertice.values():
        for a in municipios:
            for b in municipios - {a}:
                contagem.setdefault(a, {}).setdefault(b, 0)
                contagem[a][b] += 1
    return {a: sorted(b for b, n in viz.items() if n >= min_vertices)
            for a, viz in contagem.items()}


def size_class(pop: pd.Series) -> pd.Series:
    return pd.cut(pop, [0, 20e3, 100e3, 500e3, np.inf], labels=PORTES)
