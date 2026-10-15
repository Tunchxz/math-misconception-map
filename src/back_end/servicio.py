"""Capa de servicio de la aplicación.

Centraliza el acceso a los datos, a los modelos entrenados y a las métricas,
de modo que la interfaz no manipule rutas ni artefactos directamente.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DATA_TRANSFORMED, MODELS
from src.data.make_dataset import cargar, cargar_transformado
from src.features import build_features as bf
from src.models import predict_model as pm


def hay_modelos() -> bool:
    """Indica si existe al menos un artefacto entrenado en `models/`."""
    return bool(pm.modelos_disponibles())


def listar_modelos() -> dict[str, Path]:
    """Devuelve los modelos entrenados disponibles, indexados por nombre."""
    return pm.modelos_disponibles()


def cargar_datos() -> pd.DataFrame:
    """Carga el conjunto limpio con las columnas objetivo ya unificadas."""
    return bf.preparar_objetivos(cargar("train_limpio"))


def cargar_metricas() -> pd.DataFrame:
    """Carga la tabla de métricas producida por el cuaderno de modelado.

    Devuelve un DataFrame vacío si el cuaderno aún no se ha ejecutado.
    """
    ruta = DATA_TRANSFORMED / "metricas_clasicos.parquet"
    return cargar_transformado("metricas_clasicos") if ruta.exists() else pd.DataFrame()


def cargar_comparacion() -> pd.DataFrame:
    """Carga la comparación entre modelos con texto y la línea base por contexto."""
    ruta = DATA_TRANSFORMED / "comparacion_texto_contexto.parquet"
    return cargar_transformado("comparacion_texto_contexto") if ruta.exists() else pd.DataFrame()


def catalogo_preguntas(datos: pd.DataFrame) -> pd.DataFrame:
    """Devuelve las preguntas con su enunciado y sus cuatro opciones de respuesta."""
    return (
        datos.groupby("QuestionId")
        .agg(
            enunciado=("QuestionText", "first"),
            opciones=("MC_Answer_limpia", lambda s: sorted(set(s))),
            respuestas=("row_id", "size"),
        )
        .reset_index()
    )


def clasificar(
    explicacion: str,
    nombres: list[str],
    question_id: int = -1,
    mc_answer: str = "",
    question_text: str = "",
    k: int = 3,
) -> list[dict]:
    """Clasifica una explicación con los modelos indicados por nombre."""
    disponibles = listar_modelos()
    seleccion = {n: disponibles[n] for n in nombres if n in disponibles}
    return pm.predecir_con_varios(
        explicacion, seleccion, question_id, mc_answer, question_text, k
    )
