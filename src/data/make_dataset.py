"""Descarga los archivos de la competencia MAP desde la API de Kaggle.

Este script permite reconstruir `data/raw/` en cualquier máquina. Si
los tres archivos ya existen, no se vuelve a descargar nada.

El token de acceso se lee de `~/.kaggle/access_token` o de la variable de
entorno `KAGGLE_ACCESS_TOKEN`.

Uso:
    python -m src.data.make_dataset
"""

from __future__ import annotations

import io
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

from src.config import ARCHIVOS_CRUDOS as ARCHIVOS
from src.config import COMPETENCIA, DATA_PROCESSED, DATA_RAW, DATA_TRANSFORMED

URL = f"https://www.kaggle.com/api/v1/competitions/data/download-all/{COMPETENCIA}"


def leer_token() -> str:
    """Devuelve el token de acceso de Kaggle."""
    token = os.environ.get("KAGGLE_ACCESS_TOKEN", "").strip()
    if token:
        return token

    ruta = Path.home() / ".kaggle" / "access_token"
    if ruta.exists():
        return ruta.read_text().strip()

    raise SystemExit(
        "No se encontró el token de Kaggle. Se esperaba ~/.kaggle/access_token "
        "o la variable de entorno KAGGLE_ACCESS_TOKEN."
    )


def descargar() -> None:
    """Descarga y extrae los archivos de la competencia en `data/raw/`."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    faltantes = [n for n in ARCHIVOS if not (DATA_RAW / n).exists()]
    if not faltantes:
        print(f"Los datos ya están en {DATA_RAW}. No se descarga nada.")
        return

    print(f"Faltan {faltantes}. Descargando desde Kaggle...")
    respuesta = requests.get(
        URL,
        headers={"Authorization": f"Bearer {leer_token()}"},
        timeout=300,
    )
    respuesta.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(respuesta.content)) as z:
        z.extractall(DATA_RAW)

    for nombre in ARCHIVOS:
        ruta = DATA_RAW / nombre
        estado = f"{ruta.stat().st_size:,} bytes" if ruta.exists() else "NO ENCONTRADO"
        print(f"  {nombre}: {estado}")


def guardar(df: pd.DataFrame, nombre: str, carpeta: Path = DATA_PROCESSED) -> Path:
    """Guarda `df` como `<carpeta>/<nombre>.parquet` y devuelve la ruta."""
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"{nombre}.parquet"
    df.to_parquet(ruta, index=False)
    return ruta


def cargar(nombre: str, carpeta: Path = DATA_PROCESSED) -> pd.DataFrame:
    """Lee `<carpeta>/<nombre>.parquet`."""
    return pd.read_parquet(carpeta / f"{nombre}.parquet")


def guardar_transformado(df: pd.DataFrame, nombre: str) -> Path:
    """Guarda `df` en `data/transformed/`."""
    return guardar(df, nombre, DATA_TRANSFORMED)


def cargar_transformado(nombre: str) -> pd.DataFrame:
    """Lee `df` desde `data/transformed/`."""
    return cargar(nombre, DATA_TRANSFORMED)


if __name__ == "__main__":
    sys.exit(descargar())


COLUMNAS_LIMPIAS = [
    "row_id", "QuestionId", "QuestionText",
    "MC_Answer", "MC_Answer_limpia",
    "StudentExplanation", "explicacion_limpia", "explicacion_min",
    "Category", "respuesta_correcta", "tipo_explicacion",
    "Misconception", "misconception", "objetivo",
    "n_caracteres", "n_palabras", "long_media_palabra", "n_digitos",
    "tiene_digitos", "tiene_fraccion", "tiene_simbolo_mat",
    "tiene_interrogacion", "tiene_porque",
    "respuesta_correcta_pregunta",
]


def construir_limpio(crudo: pd.DataFrame) -> pd.DataFrame:
    """Aplica a `train.csv` la limpieza documentada en el cuaderno 01.

    Normaliza el texto, descompone `Category`, recodifica los vacíos de
    `Misconception` y deriva las nueve variables cuantitativas.
    """
    from src.features.build_features import (
        derivar_variables_texto,
        limpiar_latex,
        normalizar_explicacion,
    )

    df = crudo.copy()
    df["MC_Answer_limpia"] = df["MC_Answer"].map(limpiar_latex)
    df["explicacion_limpia"] = df["StudentExplanation"].map(normalizar_explicacion)
    df["explicacion_min"] = df["explicacion_limpia"].str.lower()

    partes = df["Category"].str.split("_", n=1, expand=True)
    df["respuesta_correcta"] = partes[0].map({"True": True, "False": False})
    df["tipo_explicacion"] = partes[1]
    df["misconception"] = df["Misconception"].fillna("NA")
    df["objetivo"] = df["Category"] + ":" + df["misconception"]

    df = derivar_variables_texto(df, "explicacion_limpia")

    clave = (
        df[df["respuesta_correcta"]]
        .groupby("QuestionId")["MC_Answer"]
        .agg(lambda s: s.mode().iat[0])
        .rename("respuesta_correcta_pregunta")
    )
    df = df.merge(clave, on="QuestionId", how="left")
    return df[COLUMNAS_LIMPIAS]


def asegurar_datos(verbose: bool = True) -> pd.DataFrame:
    """Garantiza que exista `data/processed/train_limpio.parquet` y lo devuelve.

    Descarga los archivos crudos si faltan y reconstruye el conjunto limpio si
    no está presente. Permite ejecutar cualquier cuaderno en una sesión nueva.
    """
    destino = DATA_PROCESSED / "train_limpio.parquet"
    if destino.exists():
        if verbose:
            print(f"Conjunto limpio encontrado en {destino}")
        return pd.read_parquet(destino)

    if not (DATA_RAW / "train.csv").exists():
        if verbose:
            print("Faltan los datos crudos, descargando desde Kaggle...")
        descargar()

    if verbose:
        print("Reconstruyendo el conjunto limpio...")
    limpio = construir_limpio(pd.read_csv(DATA_RAW / "train.csv"))
    guardar(limpio, "train_limpio")
    if verbose:
        print(f"Guardado en {destino} con forma {limpio.shape}")
    return limpio
