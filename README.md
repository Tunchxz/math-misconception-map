# MAP - Charting Student Math Misunderstandings

Análisis exploratorio del conjunto de datos de la competencia de Kaggle [*MAP - Charting Student Math Misunderstandings*](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings), organizada por la Universidad de Vanderbilt y The Learning Agency con datos de Eedi.

## El Problema

Cuando un estudiante resuelve mal un ejercicio de matemática, detrás de la respuesta equivocada suele haber una forma de razonar coherente pero incorrecta, que aplica de manera consistente. Estas estructuras se llaman errores conceptuales y son el verdadero objetivo de la enseñanza correctiva.

Para descubrirlas hace falta leer la explicación escrita del estudiante, no solo mirar la opción que marcó. El problema es de escala: etiquetar cada explicación a mano es una tarea que no se sostiene cuando hay millones de respuestas.

La tarea consiste en predecir, a partir de la explicación escrita, una etiqueta compuesta `Category:Misconception` que indica si la respuesta fue correcta, si la explicación revela un error conceptual y cuál es ese error.

## Estructura

```
math-misconception-map/
├── docs/
│   └── informe.pdf                 # planteamiento del problema, objetivos y resultados del EDA
├── notebooks/
│   ├── 01_carga_y_limpieza.ipynb   # estructura, calidad, limpieza y variables derivadas
│   └── 02_eda.ipynb                # análisis exploratorio, hallazgos y conclusiones
├── src/
│   ├── descargar_datos.py          # descarga los datos desde la API de Kaggle
│   └── utils.py                    # rutas, estilo de gráficos y lectura/escritura
├── data/raw/                       # datos crudos, no se modifican ni se versionan
├── data/processed/                 # datos generados, reproducibles, no se versionan
├── codebook.md                     # diccionario de datos, variables crudas y derivadas
└── requirements.txt
```

## Cómo Reproducirlo

### Requisitos

- **Python 3.12**, gestionado con [`uv`](https://docs.astral.sh/uv/).
- Una cuenta de Kaggle que haya **aceptado las reglas de la competencia**. Sin ese paso la API responde `403` y no permite descargar nada.
- Un token de acceso de Kaggle en `~/.kaggle/access_token`, o bien la variable de entorno `KAGGLE_ACCESS_TOKEN`.

### 1. Crear el entorno

```bash
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install -r requirements.txt
```

### 2. Descargar los datos

```bash
.venv/bin/python src/descargar_datos.py
```

Deja `train.csv`, `test.csv` y `sample_submission.csv` en `data/raw/`. El script es idempotente: si los archivos ya existen no vuelve a descargarlos.

### 3. Registrar el kernel de Jupyter

```bash
.venv/bin/python -m ipykernel install --user \
  --name map-misconceptions --display-name "MAP (py3.12)"
```

### 4. Ejecutar los notebooks en orden

El orden es obligatorio, porque el segundo consume el archivo que produce el primero.

```bash
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
  notebooks/01_carga_y_limpieza.ipynb
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
  notebooks/02_eda.ipynb
```

## Nota Sobre los Datos

Los datos pertenecen a la competencia y están sujetos a sus reglas. Ni `data/raw/` ni `data/processed/` se versionan en este repositorio.
