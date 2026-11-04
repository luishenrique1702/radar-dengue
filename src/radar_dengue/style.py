"""Estilo visual comum aos gráficos dos notebooks."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from radar_dengue.config import GEOJSON

OBSERVADO = "#2a78d6"
PREVISTO = "#eb6834"
BASELINE = "#898781"
TERCEIRA = "#1baf7a"
RAMPA = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
REGIAO_CORES = {"Oeste": "#2a78d6", "Centro": "#eb6834", "Leste": "#1baf7a"}


def setup() -> None:
    plt.rcParams.update({
        "figure.figsize": (10, 4), "figure.dpi": 110, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": "#e1e0d9", "grid.linewidth": 0.8, "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": "#52514e", "xtick.color": "#898781", "ytick.color": "#898781",
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "font.size": 10, "legend.frameon": False, "lines.linewidth": 2,
    })


def plot_map(ax, values: dict, cmap, norm, edge: str = "#fcfcfb") -> None:
    """Desenha os municípios de SP coloridos por `values` (geocode -> valor)."""
    with open(GEOJSON, encoding="utf-8") as f:
        geo = json.load(f)
    polys, cores = [], []
    for feat in geo["features"]:
        gid = feat["properties"]["id"]
        g = feat["geometry"]
        partes = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for parte in partes:
            polys.append(parte[0])
            cores.append(cmap(norm(values.get(gid, 0))))
    ax.add_collection(PolyCollection(polys, facecolors=cores, edgecolors=edge, linewidths=0.2))
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")
