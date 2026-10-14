"""Carga de modelos entrenados y predicción en dos etapas.

Expone la composición de las etapas `Category` y `Misconception` en la etiqueta
`Category:Misconception` que evalúa la competencia, junto con el
preprocesamiento necesario para aceptar texto escrito directamente por una
persona usuaria.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS, SIN_MISCONCEPTION
from src.features import build_features as bf
from src.models.train_model import puntajes

ARCHIVO_CONFIG = "configuracion.joblib"


def modelos_disponibles() -> dict[str, Path]:
    """Devuelve los artefactos encontrados en `models/`, indexados por nombre."""
    if not MODELS.exists():
        return {}
    rutas = sorted(p for p in MODELS.glob("*.joblib") if p.name != ARCHIVO_CONFIG)
    return {p.stem.replace("_", " ").capitalize(): p for p in rutas}


@lru_cache(maxsize=8)
def cargar_modelo(ruta: str) -> dict:
    """Carga un artefacto desde disco. El resultado queda en caché."""
    return joblib.load(ruta)


@lru_cache(maxsize=1)
def cargar_configuracion() -> dict:
    """Carga las columnas de entrada y los hiperparámetros usados al entrenar."""
    return joblib.load(MODELS / ARCHIVO_CONFIG)


def preparar_entrada(
    explicacion: str,
    question_id: int = -1,
    mc_answer: str = "",
    question_text: str = "",
) -> pd.DataFrame:
    """Convierte una respuesta escrita a mano en la fila que esperan los modelos.

    Aplica la misma normalización y derivación de variables usadas durante el
    entrenamiento, de modo que el preprocesamiento es transparente para quien
    usa la aplicación.
    """
    limpia = bf.normalizar_explicacion(explicacion)
    fila = pd.DataFrame(
        [
            {
                "QuestionId": question_id,
                "QuestionText": question_text,
                "MC_Answer": mc_answer,
                "MC_Answer_limpia": bf.limpiar_latex(mc_answer),
                "StudentExplanation": explicacion,
                "explicacion_limpia": limpia,
                "explicacion_min": limpia.lower(),
            }
        ]
    )
    return bf.derivar_variables_texto(fila)


def _normalizar(matriz: np.ndarray) -> np.ndarray:
    """Lleva una matriz de puntajes a filas que suman uno y son no negativas."""
    matriz = np.asarray(matriz, dtype=float)
    if matriz.min() < 0:
        matriz = matriz - matriz.min(axis=1, keepdims=True)
    matriz = np.clip(matriz, 1e-9, None)
    return matriz / matriz.sum(axis=1, keepdims=True)


def componer_probabilidades(
    proba_categoria: np.ndarray,
    clases_categoria: np.ndarray,
    proba_misconception: np.ndarray,
    clases_misconception: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Combina ambas etapas en el espacio de etiquetas `Category:Misconception`.

    Para las categorías que contienen un error conceptual se multiplica la
    probabilidad de la categoría por la de cada error. Las demás reciben el
    sufijo ``NA`` y conservan la probabilidad de la categoría.
    """
    etiquetas, columnas = [], []
    for i, categoria in enumerate(clases_categoria):
        if "Misconception" in categoria:
            for j, error in enumerate(clases_misconception):
                etiquetas.append(f"{categoria}:{error}")
                columnas.append(proba_categoria[:, i] * proba_misconception[:, j])
        else:
            etiquetas.append(f"{categoria}:{SIN_MISCONCEPTION}")
            columnas.append(proba_categoria[:, i])
    return np.column_stack(columnas), np.array(etiquetas)


def predecir(
    explicacion: str,
    modelo: str | Path | dict,
    question_id: int = -1,
    mc_answer: str = "",
    question_text: str = "",
    k: int = 3,
) -> dict:
    """Clasifica una explicación y devuelve las `k` etiquetas más probables.

    `modelo` puede ser la ruta de un artefacto o el diccionario ya cargado.
    El resultado incluye el detalle de cada etapa además del objetivo compuesto.
    """
    artefacto = modelo if isinstance(modelo, dict) else cargar_modelo(str(modelo))
    fila = preparar_entrada(explicacion, question_id, mc_answer, question_text)

    etapa1, etapa2 = artefacto["etapa1"], artefacto["etapa2"]
    p_cat = _normalizar(puntajes(etapa1, fila))
    p_mis = _normalizar(puntajes(etapa2, fila))
    clases_cat = np.asarray(etapa1[-1].classes_)
    clases_mis = np.asarray(etapa2[-1].classes_)

    proba, etiquetas = componer_probabilidades(p_cat, clases_cat, p_mis, clases_mis)
    orden = np.argsort(-proba[0])[:k]

    return {
        "modelo": artefacto.get("nombre", "desconocido"),
        "candidatas": [
            {"etiqueta": etiquetas[i], "probabilidad": float(proba[0, i])} for i in orden
        ],
        "envio": " ".join(etiquetas[orden]),
        "categoria": {
            clase: float(p) for clase, p in zip(clases_cat, p_cat[0])
        },
        "misconception": {
            clase: float(p) for clase, p in zip(clases_mis, p_mis[0])
        },
        "variables_derivadas": fila[bf.NUMERICAS].iloc[0].to_dict(),
    }


def predecir_con_varios(
    explicacion: str,
    rutas: dict[str, Path],
    question_id: int = -1,
    mc_answer: str = "",
    question_text: str = "",
    k: int = 3,
) -> list[dict]:
    """Aplica varios modelos a la misma entrada y devuelve una lista de resultados."""
    return [
        predecir(explicacion, ruta, question_id, mc_answer, question_text, k)
        for ruta in rutas.values()
    ]
