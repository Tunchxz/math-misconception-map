"""Construcción de variables y representaciones para el modelado.

Reúne la derivación de variables a partir del texto, la unificación de las
etiquetas de error conceptual y los transformadores de scikit-learn que
convierten un DataFrame en la matriz de entrada de cada modelo.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import (
    COLUMNA_TEXTO,
    NUMERICAS,
    OBJETIVO_COMPUESTO,
    OBJETIVO_ETAPA2,
    SEMILLA,
    SIN_MISCONCEPTION,
)

# Etiquetas que designan el mismo error conceptual con grafías distintas.
# La clave es la forma canónica usada durante el entrenamiento y el valor es la
# grafía original que debe restituirse por pregunta al generar predicciones.
ALIAS_ETIQUETAS = {"Wrong_fraction": ("Wrong_fraction", "Wrong_Fraction")}

_RE_ESPACIOS = re.compile(r"\s+")
_RE_FRACCION = re.compile(r"\d\s*/\s*\d")
_RE_SIMBOLO = re.compile(r"[+\-*=<>%×÷]")
_RE_PORQUE = re.compile(r"\b(?:because|cause|cos|coz|as)\b")
_RE_LATEX_FRAC = re.compile(r"\\frac\{([^{}]*)\}\{([^{}]*)\}")
_RE_LATEX_CMD = re.compile(r"\\[a-zA-Z]+")


def limpiar_latex(texto: str) -> str:
    """Convierte notación LaTeX de una respuesta en texto legible.

    Las fracciones pasan a la forma ``a/b`` y se retiran delimitadores y comandos.
    """
    t = _RE_LATEX_FRAC.sub(r"\1/\2", str(texto))
    t = _RE_LATEX_CMD.sub(" ", t)
    t = t.replace("\\(", " ").replace("\\)", " ")
    t = re.sub(r"[{}$]", " ", t)
    return _RE_ESPACIOS.sub(" ", t).strip()


def normalizar_explicacion(texto: str) -> str:
    """Colapsa espacios repetidos y recorta los extremos. No corrige ortografía."""
    return _RE_ESPACIOS.sub(" ", str(texto)).strip()


def derivar_variables_texto(df: pd.DataFrame, columna: str = "explicacion_limpia") -> pd.DataFrame:
    """Devuelve `df` con las nueve variables derivadas de `columna` añadidas.

    Las variables son las documentadas en ``references/codebook.md``: longitud en
    caracteres y palabras, longitud media de palabra, conteo de dígitos y cinco
    indicadores lógicos.
    """
    df = df.copy()
    texto = df[columna].astype(str)
    minusculas = texto.str.lower()

    df["n_caracteres"] = texto.str.len()
    df["n_palabras"] = texto.str.split().str.len().clip(lower=1)
    df["long_media_palabra"] = (df["n_caracteres"] / df["n_palabras"]).round(2)
    df["n_digitos"] = texto.str.count(r"\d")
    df["tiene_digitos"] = df["n_digitos"] > 0
    df["tiene_fraccion"] = texto.str.contains(_RE_FRACCION, regex=True)
    df["tiene_simbolo_mat"] = texto.str.contains(_RE_SIMBOLO, regex=True)
    df["tiene_interrogacion"] = texto.str.contains(r"\?", regex=True)
    df["tiene_porque"] = minusculas.str.contains(_RE_PORQUE, regex=True)
    return df


def unificar_etiquetas(serie: pd.Series) -> pd.Series:
    """Mapea las grafías alternativas de una etiqueta a su forma canónica."""
    mapa = {
        variante: canonica
        for canonica, variantes in ALIAS_ETIQUETAS.items()
        for variante in variantes
    }
    return serie.map(lambda v: mapa.get(v, v))


def restituir_etiqueta(etiqueta: str, question_id: int | None, tabla: pd.DataFrame) -> str:
    """Devuelve la grafía original de `etiqueta` para `question_id`.

    `tabla` debe tener las columnas ``QuestionId``, ``canonica`` y ``original``.
    Si no hay correspondencia registrada, se devuelve `etiqueta` sin cambios.
    """
    if etiqueta not in ALIAS_ETIQUETAS or question_id is None:
        return etiqueta
    fila = tabla[(tabla["canonica"] == etiqueta) & (tabla["QuestionId"] == question_id)]
    return fila["original"].iat[0] if len(fila) else etiqueta


def tabla_alias(df: pd.DataFrame) -> pd.DataFrame:
    """Construye la correspondencia entre forma canónica y grafía original por pregunta."""
    con_alias = df[df["misconception"].isin({v for vs in ALIAS_ETIQUETAS.values() for v in vs})]
    tabla = (
        con_alias.assign(canonica=lambda d: unificar_etiquetas(d["misconception"]))
        .rename(columns={"misconception": "original"})[["QuestionId", "canonica", "original"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return tabla


def preparar_objetivos(df: pd.DataFrame) -> pd.DataFrame:
    """Añade las columnas objetivo unificadas usadas por las dos etapas del modelo."""
    df = df.copy()
    df[OBJETIVO_ETAPA2] = unificar_etiquetas(df["misconception"])
    df[OBJETIVO_COMPUESTO] = df["Category"] + ":" + df[OBJETIVO_ETAPA2]
    return df


def _a_denso(x):
    """Convierte booleanos y enteros a float para los estimadores que lo exigen."""
    return np.asarray(x, dtype=float)


def vectorizador_texto(min_df: int = 3, svd_componentes: int | None = None) -> Pipeline:
    """Devuelve el transformador de texto: TF-IDF de palabras y de caracteres.

    Los n-gramas de caracteres aportan robustez ante los errores de ortografía
    frecuentes en el texto escrito por estudiantes. Si `svd_componentes` no es
    ``None`` se añade una reducción por SVD truncada.
    """
    pasos = [
        (
            "tfidf",
            ColumnTransformer(
                [
                    (
                        "palabras",
                        TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, sublinear_tf=True),
                        COLUMNA_TEXTO,
                    ),
                    (
                        "caracteres",
                        TfidfVectorizer(
                            analyzer="char_wb",
                            ngram_range=(3, 5),
                            min_df=min_df * 2,
                            sublinear_tf=True,
                        ),
                        COLUMNA_TEXTO,
                    ),
                ]
            ),
        )
    ]
    if svd_componentes:
        pasos.append(("svd", TruncatedSVD(n_components=svd_componentes, random_state=SEMILLA)))
    return Pipeline(pasos)


def transformador_texto_mas_numericas(svd_componentes: int = 200) -> Pipeline:
    """Transformador denso: SVD sobre TF-IDF concatenado con las variables derivadas.

    Pensado para estimadores basados en árboles, que no operan bien sobre
    matrices dispersas de alta dimensión.
    """
    return Pipeline(
        [
            (
                "union",
                ColumnTransformer(
                    [
                        ("texto", vectorizador_texto(svd_componentes=svd_componentes), [COLUMNA_TEXTO]),
                        (
                            "numericas",
                            Pipeline(
                                [
                                    ("a_float", FunctionTransformer(_a_denso)),
                                    ("escala", StandardScaler()),
                                ]
                            ),
                            NUMERICAS,
                        ),
                    ]
                ),
            )
        ]
    )


def transformador_contexto() -> ColumnTransformer:
    """Transformador de la línea base: solo la pregunta y la opción seleccionada.

    No incluye ninguna señal del texto escrito por el estudiante.
    """
    return ColumnTransformer(
        [
            (
                "contexto",
                OneHotEncoder(handle_unknown="ignore", min_frequency=1),
                ["QuestionId", "MC_Answer"],
            )
        ]
    )


def filas_con_misconception(df: pd.DataFrame) -> pd.DataFrame:
    """Subconjunto de `df` en el que la etapa 2 es aplicable."""
    return df[df[OBJETIVO_ETAPA2] != SIN_MISCONCEPTION].reset_index(drop=True)
