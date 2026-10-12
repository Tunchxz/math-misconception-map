"""Estilo común de los gráficos y utilidades de guardado de figuras."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FIGURES, PALETA_CUALITATIVA


def set_estilo() -> None:
    """Aplica el estilo y la paleta del proyecto a matplotlib y seaborn."""
    sns.set_theme(style="whitegrid")
    sns.set_palette(PALETA_CUALITATIVA)
    plt.rcParams.update(
        {
            "figure.figsize": (9, 4.5),
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.prop_cycle": plt.cycler(color=PALETA_CUALITATIVA),
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def guardar_figura(fig: plt.Figure, nombre: str) -> Path:
    """Guarda `fig` en `reports/figures/<nombre>.png` y devuelve la ruta."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    ruta = FIGURES / f"{nombre}.png"
    fig.savefig(ruta)
    return ruta
