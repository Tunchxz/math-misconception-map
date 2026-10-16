"""Paleta y ayudas de presentación de la aplicación.

La paleta cualitativa es la de Okabe-Ito, legible bajo los tres tipos de
daltonismo. Los colores semánticos se reservan para acierto, error y neutro.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

from src.config import (
    COLOR_ACIERTO,
    COLOR_ERROR,
    COLOR_NEUTRO,
    PALETA_CUALITATIVA,
    PALETA_SECUENCIAL,
)

TIPOGRAFIA = "system-ui, -apple-system, Segoe UI, Roboto, sans-serif"

COLOR_FONDO = "#FFFFFF"
COLOR_PANEL = "#F4F6F8"
COLOR_TEXTO = "#1A1A1A"
COLOR_TEXTO_SUAVE = "#53606B"
COLOR_BORDE = "#D6DBE0"

# Color fijo por categoría, para que su significado no cambie entre gráficas.
COLOR_CATEGORIA = {
    "True_Correct": "#009E73",
    "True_Neither": "#56B4E9",
    "True_Misconception": "#E69F00",
    "False_Correct": "#0072B2",
    "False_Neither": "#CC79A7",
    "False_Misconception": "#D55E00",
}

COLOR_ESQUEMA = {"aleatorio": "#56B4E9", "agrupado": "#D55E00"}


def registrar_tema() -> None:
    """Registra y activa el tema de Plotly del proyecto."""
    tema = go.layout.Template()
    tema.layout = go.Layout(
        colorway=PALETA_CUALITATIVA,
        font={"family": TIPOGRAFIA, "size": 13, "color": COLOR_TEXTO},
        paper_bgcolor=COLOR_FONDO,
        plot_bgcolor=COLOR_FONDO,
        colorscale={"sequential": PALETA_SECUENCIAL},
        xaxis={"gridcolor": COLOR_BORDE, "zerolinecolor": COLOR_BORDE},
        yaxis={"gridcolor": COLOR_BORDE, "zerolinecolor": COLOR_BORDE},
        margin={"l": 60, "r": 30, "t": 60, "b": 50},
        hoverlabel={"font_family": TIPOGRAFIA},
    )
    pio.templates["map"] = tema
    pio.templates.default = "plotly_white+map"


CSS = f"""
<style>
    .stApp {{ background: {COLOR_FONDO}; }}
    .bloque-nota {{
        background: {COLOR_PANEL};
        border-left: 4px solid {COLOR_NEUTRO};
        border-radius: 4px;
        padding: 0.9rem 1.1rem;
        margin: 0.6rem 0 1.1rem 0;
        color: {COLOR_TEXTO};
        font-size: 0.92rem;
        line-height: 1.55;
    }}
    .bloque-alerta {{
        background: #FDF3EC;
        border-left: 4px solid {COLOR_ERROR};
        border-radius: 4px;
        padding: 0.9rem 1.1rem;
        margin: 0.6rem 0 1.1rem 0;
        font-size: 0.92rem;
        line-height: 1.55;
    }}
    .bloque-exito {{
        background: #EAF6F1;
        border-left: 4px solid {COLOR_ACIERTO};
        border-radius: 4px;
        padding: 0.9rem 1.1rem;
        margin: 0.6rem 0 1.1rem 0;
        font-size: 0.92rem;
        line-height: 1.55;
    }}
    .etiqueta-chip {{
        display: inline-block;
        padding: 0.18rem 0.6rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #FFFFFF;
    }}
    div[data-testid="stMetricValue"] {{ font-size: 1.55rem; }}
    section[data-testid="stSidebar"] {{ background: {COLOR_PANEL}; }}
</style>
"""


def nota(texto: str) -> str:
    """Devuelve el HTML de un bloque informativo."""
    return f'<div class="bloque-nota">{texto}</div>'


def alerta(texto: str) -> str:
    """Devuelve el HTML de un bloque de advertencia."""
    return f'<div class="bloque-alerta">{texto}</div>'


def exito(texto: str) -> str:
    """Devuelve el HTML de un bloque de resultado favorable."""
    return f'<div class="bloque-exito">{texto}</div>'


def chip(texto: str, color: str) -> str:
    """Devuelve el HTML de una etiqueta coloreada."""
    return f'<span class="etiqueta-chip" style="background:{color}">{texto}</span>'
