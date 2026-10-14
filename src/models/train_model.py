"""Definición, entrenamiento y validación de los modelos de clasificación.

Expone la métrica MAP@3 de la competencia, el catálogo de modelos con sus
rejillas de hiperparámetros y la rutina de validación cruzada con los dos
esquemas de partición empleados en el proyecto.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import N_FOLDS, SEMILLA
from src.features import build_features as bf

ESQUEMAS = ("aleatorio", "agrupado")


def puntajes(modelo: BaseEstimator, X) -> np.ndarray:
    """Devuelve una matriz de puntajes por clase, usable para ordenar candidatas.

    Emplea ``predict_proba`` cuando está disponible y ``decision_function`` en
    caso contrario. Para problemas binarios la salida se expande a dos columnas.
    """
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)
    margenes = modelo.decision_function(X)
    if margenes.ndim == 1:
        margenes = np.column_stack([-margenes, margenes])
    return margenes


def map_at_k(y_true: np.ndarray, puntuaciones: np.ndarray, clases: np.ndarray, k: int = 3) -> float:
    """Precisión promedio media hasta la posición `k`.

    Cada observación aporta ``1/p`` si su etiqueta aparece en la posición ``p``
    de las `k` candidatas mejor puntuadas, y 0 si no aparece.
    """
    orden = np.argsort(-puntuaciones, axis=1)[:, :k]
    predichas = clases[orden]
    aciertos = predichas == np.asarray(y_true).reshape(-1, 1)
    pesos = 1.0 / np.arange(1, k + 1)
    return float((aciertos * pesos).sum(axis=1).mean())


def formato_envio(puntuaciones: np.ndarray, clases: np.ndarray, k: int = 3) -> list[str]:
    """Construye la cadena de `k` candidatas separadas por espacio que espera Kaggle."""
    orden = np.argsort(-puntuaciones, axis=1)[:, :k]
    return [" ".join(clases[fila]) for fila in orden]


@dataclass
class Modelo:
    """Modelo candidato con su transformador, estimador y rejilla de búsqueda."""

    nombre: str
    construir: Any
    rejilla: dict[str, list] = field(default_factory=dict)
    usa_texto: bool = True
    admite_disperso: bool = True


def _pipeline(transformador, estimador) -> Pipeline:
    return Pipeline([("features", transformador), ("clf", estimador)])


def catalogo_modelos() -> list[Modelo]:
    """Devuelve los modelos a comparar, en el orden en que se reportan."""
    return [
        Modelo(
            nombre="Linea base (mayoritaria)",
            construir=lambda: _pipeline(
                bf.transformador_contexto(), DummyClassifier(strategy="most_frequent")
            ),
            usa_texto=False,
        ),
        Modelo(
            nombre="Linea base por contexto",
            construir=lambda: _pipeline(
                bf.transformador_contexto(),
                LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced"),
            ),
            usa_texto=False,
        ),
        Modelo(
            nombre="Regresion logistica",
            construir=lambda: _pipeline(
                bf.vectorizador_texto(),
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
            rejilla={"clf__C": [0.5, 1.0, 2.0, 4.0]},
        ),
        Modelo(
            nombre="LinearSVC",
            construir=lambda: _pipeline(
                bf.vectorizador_texto(),
                LinearSVC(class_weight="balanced", dual="auto", max_iter=3000),
            ),
            rejilla={"clf__C": [0.1, 0.5, 1.0, 2.0]},
        ),
        Modelo(
            nombre="Complement NB",
            construir=lambda: _pipeline(bf.vectorizador_texto(), ComplementNB()),
            rejilla={"clf__alpha": [0.1, 0.3, 1.0, 3.0]},
        ),
        Modelo(
            nombre="Gradient boosting",
            construir=lambda: _pipeline(
                bf.transformador_texto_mas_numericas(svd_componentes=150),
                HistGradientBoostingClassifier(
                    random_state=SEMILLA, max_iter=200, early_stopping=True
                ),
            ),
            rejilla={"clf__learning_rate": [0.1, 0.2], "clf__max_leaf_nodes": [31, 63]},
            admite_disperso=False,
        ),
    ]


def particiones(X: pd.DataFrame, y: np.ndarray, grupos: np.ndarray, esquema: str):
    """Genera los índices de entrenamiento y prueba del esquema indicado.

    ``aleatorio`` usa StratifiedKFold y ``agrupado`` usa GroupKFold sobre
    ``QuestionId``, de modo que ninguna pregunta aparece en ambos lados.
    """
    if esquema == "aleatorio":
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMILLA)
        return cv.split(X, y)
    if esquema == "agrupado":
        n = min(N_FOLDS, len(np.unique(grupos)))
        return GroupKFold(n_splits=n).split(X, y, grupos)
    raise ValueError(f"Esquema desconocido: {esquema}")


def evaluar(
    modelo: Modelo,
    X: pd.DataFrame,
    y: np.ndarray,
    grupos: np.ndarray,
    esquema: str,
    k: int = 3,
) -> dict:
    """Valida `modelo` con el esquema indicado y devuelve sus métricas agregadas.

    Las clases presentes en la partición de prueba pero ausentes del
    entrenamiento se cuentan como error, sin excluirse.
    """
    f1_macro, f1_pond, maps, no_vistas = [], [], [], []
    t0 = time.perf_counter()

    for idx_tr, idx_te in particiones(X, y, grupos, esquema):
        est = clone(modelo.construir())
        est.fit(X.iloc[idx_tr], y[idx_tr])

        y_pred = est.predict(X.iloc[idx_te])
        y_real = y[idx_te]

        f1_macro.append(f1_score(y_real, y_pred, average="macro", zero_division=0))
        f1_pond.append(f1_score(y_real, y_pred, average="weighted", zero_division=0))

        clases = np.asarray(est.classes_ if hasattr(est, "classes_") else est[-1].classes_)
        maps.append(map_at_k(y_real, puntajes(est, X.iloc[idx_te]), clases, k=k))
        no_vistas.append(len(set(y_real) - set(y[idx_tr])))

    return {
        "modelo": modelo.nombre,
        "esquema": esquema,
        "f1_macro": float(np.mean(f1_macro)),
        "f1_macro_std": float(np.std(f1_macro)),
        "f1_ponderado": float(np.mean(f1_pond)),
        f"map@{k}": float(np.mean(maps)),
        "clases_no_vistas": float(np.mean(no_vistas)),
        "segundos": time.perf_counter() - t0,
    }


def tunear(
    modelo: Modelo,
    X: pd.DataFrame,
    y: np.ndarray,
    grupos: np.ndarray,
    criterio: str = "f1_macro",
) -> tuple[dict, pd.DataFrame]:
    """Busca los hiperparámetros de `modelo` sobre una única partición de retención.

    Se usa una sola partición en lugar de validación cruzada completa para
    acotar el costo. Devuelve los mejores parámetros y la tabla de resultados.
    """
    if not modelo.rejilla:
        return {}, pd.DataFrame()

    idx_tr, idx_te = next(iter(particiones(X, y, grupos, "aleatorio")))
    X_tr, X_te, y_tr, y_te = X.iloc[idx_tr], X.iloc[idx_te], y[idx_tr], y[idx_te]

    combinaciones = _producto(modelo.rejilla)
    filas = []
    for i, combinacion in enumerate(combinaciones):
        est = clone(modelo.construir()).set_params(**combinacion)
        t0 = time.perf_counter()
        est.fit(X_tr, y_tr)
        y_pred = est.predict(X_te)
        filas.append(
            {
                "_combinacion": i,
                **combinacion,
                "f1_macro": f1_score(y_te, y_pred, average="macro", zero_division=0),
                "f1_ponderado": f1_score(y_te, y_pred, average="weighted", zero_division=0),
                "segundos": time.perf_counter() - t0,
            }
        )

    tabla = pd.DataFrame(filas).sort_values(criterio, ascending=False).reset_index(drop=True)
    # Se devuelve la combinación original y no la fila del DataFrame, que convertiría
    # los enteros en float y sería rechazada por los estimadores.
    mejores = combinaciones[int(tabla.loc[0, "_combinacion"])]
    return mejores, tabla.drop(columns="_combinacion")


def _producto(rejilla: dict[str, list]) -> list[dict]:
    """Expande una rejilla de hiperparámetros en la lista de combinaciones."""
    from itertools import product

    nombres = list(rejilla)
    return [dict(zip(nombres, valores)) for valores in product(*(rejilla[n] for n in nombres))]


def entrenar_final(modelo: Modelo, X: pd.DataFrame, y: np.ndarray, params: dict) -> Pipeline:
    """Ajusta `modelo` sobre todos los datos con los hiperparámetros indicados."""
    est = clone(modelo.construir())
    if params:
        est.set_params(**params)
    est.fit(X, y)
    return est



def _probabilidades(estimador, X) -> np.ndarray:
    """Devuelve puntajes no negativos normalizados por fila."""
    p = np.asarray(puntajes(estimador, X), dtype=float)
    if p.min() < 0:
        p = p - p.min(axis=1, keepdims=True)
    p = np.clip(p, 1e-9, None)
    return p / p.sum(axis=1, keepdims=True)


def componer(proba_cat, clases_cat, proba_mis, clases_mis):
    """Combina ambas etapas en el espacio de etiquetas `Category:Misconception`."""
    etiquetas, columnas = [], []
    for i, categoria in enumerate(clases_cat):
        if "Misconception" in categoria:
            for j, error in enumerate(clases_mis):
                etiquetas.append(f"{categoria}:{error}")
                columnas.append(proba_cat[:, i] * proba_mis[:, j])
        else:
            etiquetas.append(f"{categoria}:NA")
            columnas.append(proba_cat[:, i])
    return np.column_stack(columnas), np.array(etiquetas)


def evaluar_completo(
    modelo: Modelo,
    datos: pd.DataFrame,
    columnas: list[str],
    esquema: str,
    params_e1: dict | None = None,
    params_e2: dict | None = None,
    k: int = 3,
) -> list[dict]:
    """Valida las dos etapas y su composición en un único recorrido de particiones.

    Entrena una sola vez por partición y reutiliza los ajustes para las tres
    mediciones, en lugar de repetir el entrenamiento por cada etapa.
    """
    from src.config import OBJETIVO_COMPUESTO, OBJETIVO_ETAPA2, SIN_MISCONCEPTION

    X = datos[columnas]
    y1 = datos["Category"].values
    y2 = datos[OBJETIVO_ETAPA2].values
    yc = datos[OBJETIVO_COMPUESTO].values
    grupos = datos["QuestionId"].values

    acum = {
        "1. Category": {"f1_macro": [], "f1_ponderado": [], "map": [], "no_vistas": []},
        "2. Misconception": {"f1_macro": [], "f1_ponderado": [], "map": [], "no_vistas": []},
        "3. Compuesto": {"f1_macro": [], "f1_ponderado": [], "map": [], "no_vistas": []},
    }
    t0 = time.perf_counter()

    for idx_tr, idx_te in particiones(X, y1, grupos, esquema):
        X_tr, X_te = X.iloc[idx_tr], X.iloc[idx_te]
        tiene_mis_tr = y2[idx_tr] != SIN_MISCONCEPTION
        tiene_mis_te = y2[idx_te] != SIN_MISCONCEPTION

        e1 = entrenar_final(modelo, X_tr, y1[idx_tr], params_e1 or {})
        e2 = entrenar_final(
            modelo, X_tr[tiene_mis_tr], y2[idx_tr][tiene_mis_tr], params_e2 or {}
        )

        clases1 = np.asarray(e1[-1].classes_)
        clases2 = np.asarray(e2[-1].classes_)
        p1 = _probabilidades(e1, X_te)
        p2 = _probabilidades(e2, X_te)

        # Etapa 1
        pred1 = clases1[p1.argmax(axis=1)]
        acum["1. Category"]["f1_macro"].append(
            f1_score(y1[idx_te], pred1, average="macro", zero_division=0)
        )
        acum["1. Category"]["f1_ponderado"].append(
            f1_score(y1[idx_te], pred1, average="weighted", zero_division=0)
        )
        acum["1. Category"]["map"].append(map_at_k(y1[idx_te], p1, clases1, k=k))
        acum["1. Category"]["no_vistas"].append(len(set(y1[idx_te]) - set(y1[idx_tr])))

        # Etapa 2, solo donde hay error conceptual
        if tiene_mis_te.any():
            y2_te = y2[idx_te][tiene_mis_te]
            p2_te = p2[tiene_mis_te]
            pred2 = clases2[p2_te.argmax(axis=1)]
            acum["2. Misconception"]["f1_macro"].append(
                f1_score(y2_te, pred2, average="macro", zero_division=0)
            )
            acum["2. Misconception"]["f1_ponderado"].append(
                f1_score(y2_te, pred2, average="weighted", zero_division=0)
            )
            acum["2. Misconception"]["map"].append(map_at_k(y2_te, p2_te, clases2, k=k))
            acum["2. Misconception"]["no_vistas"].append(
                len(set(y2_te) - set(y2[idx_tr][tiene_mis_tr]))
            )

        # Objetivo compuesto
        proba_c, clases_c = componer(p1, clases1, p2, clases2)
        pred_c = clases_c[proba_c.argmax(axis=1)]
        acum["3. Compuesto"]["f1_macro"].append(
            f1_score(yc[idx_te], pred_c, average="macro", zero_division=0)
        )
        acum["3. Compuesto"]["f1_ponderado"].append(
            f1_score(yc[idx_te], pred_c, average="weighted", zero_division=0)
        )
        acum["3. Compuesto"]["map"].append(map_at_k(yc[idx_te], proba_c, clases_c, k=k))
        acum["3. Compuesto"]["no_vistas"].append(len(set(yc[idx_te]) - set(yc[idx_tr])))

    segundos = time.perf_counter() - t0
    return [
        {
            "modelo": modelo.nombre,
            "esquema": esquema,
            "etapa": etapa,
            "f1_macro": float(np.mean(v["f1_macro"])) if v["f1_macro"] else np.nan,
            "f1_ponderado": float(np.mean(v["f1_ponderado"])) if v["f1_ponderado"] else np.nan,
            f"map@{k}": float(np.mean(v["map"])) if v["map"] else np.nan,
            "clases_no_vistas": float(np.mean(v["no_vistas"])) if v["no_vistas"] else np.nan,
            "segundos": segundos,
        }
        for etapa, v in acum.items()
    ]
