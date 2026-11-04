"""Validação dos dados antes de treinar ou prever.

Se a API mudar de formato, vier incompleta ou desatualizada, o pipeline para com uma mensagem
clara em vez de gerar previsões erradas sem que ninguém perceba.
"""

from __future__ import annotations

import pandas as pd

REQUIRED = ["geocode", "municipio", "data_iniSE", "SE", "casos", "casos_est", "pop", "Rt",
            "tempmin", "tempmed", "tempmax", "umidmin", "umidmed", "umidmax", "receptivo"]
MIN_MUNICIPIOS = 640
MAX_DIAS_DESATUALIZADO = 21


class DataValidationError(ValueError):
    """Os dados não passaram nas checagens mínimas de qualidade."""


def validate_raw(df: pd.DataFrame, hoje: pd.Timestamp | None = None) -> list[str]:
    """Levanta DataValidationError se houver problema grave. Devolve avisos não fatais."""
    erros, avisos = [], []

    faltando = [c for c in REQUIRED if c not in df.columns]
    if faltando:
        raise DataValidationError(f"Colunas ausentes: {faltando}")

    n_mun = df["geocode"].nunique()
    if n_mun < MIN_MUNICIPIOS:
        erros.append(f"Só {n_mun} municípios (mínimo {MIN_MUNICIPIOS})")
    if (df["casos"] < 0).any():
        erros.append("Há contagens de casos negativas")
    if (df["pop"] <= 0).any():
        erros.append("Há população zero ou negativa")
    dup = df.duplicated(["geocode", "SE"]).sum()
    if dup:
        erros.append(f"{dup} semanas duplicadas")

    hoje = hoje or pd.Timestamp.now().normalize()
    atraso = (hoje - pd.Timestamp(df["data_iniSE"].max())).days
    if atraso > MAX_DIAS_DESATUALIZADO:
        avisos.append(f"Dados desatualizados: última semana tem {atraso} dias")

    falt_clima = df.loc[df["data_iniSE"] == df["data_iniSE"].max(), "tempmed"].isna().mean()
    if falt_clima > 0.5:
        avisos.append(f"{falt_clima:.0%} dos municípios sem clima na última semana")

    if erros:
        raise DataValidationError("; ".join(erros))
    for a in avisos:
        print(f"AVISO: {a}")
    return avisos
