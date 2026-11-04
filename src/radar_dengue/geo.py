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


def size_class(pop: pd.Series) -> pd.Series:
    return pd.cut(pop, [0, 20e3, 100e3, 500e3, np.inf], labels=PORTES)
