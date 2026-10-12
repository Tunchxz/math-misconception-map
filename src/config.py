"""Rutas, constantes y paleta de color del proyecto."""

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

DATA_RAW = RAIZ / "data" / "raw"
DATA_TRANSFORMED = RAIZ / "data" / "transformed"
DATA_PROCESSED = RAIZ / "data" / "processed"
DATA_EXTERNAL = RAIZ / "data" / "external"
MODELS = RAIZ / "models"
REPORTS = RAIZ / "reports"
FIGURES = REPORTS / "figures"

SEMILLA = 42
N_FOLDS = 5

COMPETENCIA = "map-charting-student-math-misunderstandings"
ARCHIVOS_CRUDOS = ("train.csv", "test.csv", "sample_submission.csv")

COLUMNA_TEXTO = "explicacion_min"
COLUMNA_GRUPO = "QuestionId"
OBJETIVO_ETAPA1 = "Category"
OBJETIVO_ETAPA2 = "misconception_unificada"
OBJETIVO_COMPUESTO = "objetivo_unificado"

NUMERICAS = [
    "n_caracteres",
    "n_palabras",
    "long_media_palabra",
    "n_digitos",
    "tiene_digitos",
    "tiene_fraccion",
    "tiene_simbolo_mat",
    "tiene_interrogacion",
    "tiene_porque",
]

SIN_MISCONCEPTION = "NA"

# Paleta cualitativa de Okabe-Ito, distinguible con los tres tipos de daltonismo.
PALETA_CUALITATIVA = [
    "#0072B2",  # azul
    "#D55E00",  # bermellón
    "#009E73",  # verde azulado
    "#CC79A7",  # rosa
    "#E69F00",  # naranja
    "#56B4E9",  # celeste
    "#F0E442",  # amarillo
    "#000000",  # negro
]

# Secuencial para magnitudes y divergente para diferencias con cero significativo.
PALETA_SECUENCIAL = "viridis"
PALETA_DIVERGENTE = "RdBu_r"

COLOR_ACIERTO = "#009E73"
COLOR_ERROR = "#D55E00"
COLOR_NEUTRO = "#0072B2"
