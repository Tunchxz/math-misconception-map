"""Descarga los archivos de la competencia MAP desde la API de Kaggle.

Este script permite reconstruir `data/raw/` en cualquier máquina. Si
los tres archivos ya existen, no se vuelve a descargar nada.

El token de acceso se lee de `~/.kaggle/access_token` o de la variable de
entorno `KAGGLE_ACCESS_TOKEN`.

Uso:
    .venv/bin/python src/descargar_datos.py
"""

from __future__ import annotations

import io
import os
import sys
import zipfile
from pathlib import Path

import requests

COMPETENCIA = "map-charting-student-math-misunderstandings"
URL = f"https://www.kaggle.com/api/v1/competitions/data/download-all/{COMPETENCIA}"
ARCHIVOS = ("train.csv", "test.csv", "sample_submission.csv")

RAIZ = Path(__file__).resolve().parents[1]
DATA_RAW = RAIZ / "data" / "raw"


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


if __name__ == "__main__":
    sys.exit(descargar())
