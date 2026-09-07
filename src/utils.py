"""Utilidades compartidas del proyecto MAP.

Contiene únicamente lo que se reutiliza entre notebooks: las rutas del
proyecto, el estilo común de los gráficos y dos ayudas de lectura y escritura
en `data/processed`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RAIZ = Path(__file__).resolve().parents[1]
DATA_RAW = RAIZ / "data" / "raw"
DATA_PROCESSED = RAIZ / "data" / "processed"

PALETA = "deep"


def set_estilo() -> None:
    """Aplica un estilo homogéneo a todos los gráficos del proyecto."""
    sns.set_theme(style="whitegrid", palette=PALETA)
    plt.rcParams.update(
        {
            "figure.figsize": (9, 4.5),
            "figure.dpi": 110,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def guardar(df: pd.DataFrame, nombre: str) -> Path:
    """Guarda un DataFrame en `data/processed/<nombre>.parquet`."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    ruta = DATA_PROCESSED / f"{nombre}.parquet"
    df.to_parquet(ruta, index=False)
    return ruta


def cargar(nombre: str) -> pd.DataFrame:
    """Lee `data/processed/<nombre>.parquet`."""
    return pd.read_parquet(DATA_PROCESSED / f"{nombre}.parquet")
