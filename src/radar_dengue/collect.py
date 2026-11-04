"""Coleta de dados semanais de dengue (InfoDengue) para todos os municípios de SP.

Uso:
    python -m radar_dengue.collect
"""

from __future__ import annotations

import io
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from radar_dengue.config import (
    DATA_PROCESSED,
    DATA_RAW,
    GEOJSON,
    INFODENGUE_URL,
    YEAR_END,
    YEAR_START,
)
from radar_dengue.validate import validate_raw

KEEP_COLS = [
    "data_iniSE", "SE", "casos", "casos_est", "p_inc100k", "nivel", "Rt", "pop",
    "tempmin", "tempmed", "tempmax", "umidmin", "umidmed", "umidmax", "receptivo",
    "transmissao",
]


def list_municipalities() -> pd.DataFrame:
    """Lê código IBGE e nome dos municípios a partir da malha geográfica."""
    with open(GEOJSON, encoding="utf-8") as f:
        geo = json.load(f)
    rows = [(f["properties"]["id"], f["properties"]["name"]) for f in geo["features"]]
    return pd.DataFrame(rows, columns=["geocode", "municipio"])


def fetch_city(geocode: str, retries: int = 3) -> pd.DataFrame:
    """Baixa a série semanal completa de um município."""
    params = {
        "geocode": geocode, "disease": "dengue", "format": "csv",
        "ew_start": 1, "ew_end": 53, "ey_start": YEAR_START, "ey_end": YEAR_END,
    }
    for attempt in range(retries):
        try:
            r = requests.get(INFODENGUE_URL, params=params, timeout=90)
            r.raise_for_status()
            if r.text.strip():
                df = pd.read_csv(io.StringIO(r.text))
                if len(df):
                    df = df[[c for c in KEEP_COLS if c in df.columns]].copy()
                    df["geocode"] = geocode
                    return df
        except (requests.RequestException, pd.errors.ParserError):
            pass
        time.sleep(2 * (attempt + 1))
    return pd.DataFrame()


def collect(max_workers: int = 6) -> pd.DataFrame:
    munis = list_municipalities()
    frames, failed = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_city, g): g for g in munis["geocode"]}
        for i, fut in enumerate(as_completed(futures), 1):
            df = fut.result()
            (frames.append(df) if len(df) else failed.append(futures[fut]))
            if i % 50 == 0:
                print(f"{i}/{len(munis)} municípios baixados")
    if failed:  # segunda tentativa, sequencial
        for g in failed[:]:
            df = fetch_city(g, retries=5)
            if len(df):
                frames.append(df)
                failed.remove(g)
    print(f"Concluído. Falhas: {len(failed)}")

    data = pd.concat(frames, ignore_index=True)
    data = data.merge(munis, on="geocode", how="left")
    data["data_iniSE"] = pd.to_datetime(data["data_iniSE"])
    data = data.sort_values(["geocode", "data_iniSE"]).drop_duplicates(["geocode", "SE"])
    return data


def main() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    data = collect()
    validate_raw(data)  # não sobrescreve os dados bons com uma coleta defeituosa
    data.to_parquet(DATA_RAW / "infodengue_sp.parquet", index=False)
    print(f"{len(data):,} linhas | {data['geocode'].nunique()} municípios | "
          f"{data['data_iniSE'].min().date()} a {data['data_iniSE'].max().date()}")


if __name__ == "__main__":
    main()
