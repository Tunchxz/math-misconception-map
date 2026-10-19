# MAP - Charting Student Math Misunderstandings

Detección automática de errores conceptuales de matemática a partir de explicaciones escritas por estudiantes, con los datos de la competencia de Kaggle [*MAP - Charting Student Math Misunderstandings*](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings), organizada por la Universidad de Vanderbilt y The Learning Agency con datos de Eedi.

## El Problema

Cuando un estudiante resuelve mal un ejercicio de matemática, detrás de la respuesta equivocada suele haber una forma de razonar coherente pero incorrecta, que aplica de manera consistente. Estas estructuras se llaman errores conceptuales y son el verdadero objetivo de la enseñanza correctiva.

Para descubrirlas hace falta leer la explicación escrita del estudiante, no solo mirar la opción que marcó. El problema es de escala: etiquetar cada explicación a mano es una tarea que no se sostiene cuando hay millones de respuestas.

La tarea consiste en predecir, a partir de la explicación escrita, una etiqueta compuesta `Category:Misconception` que indica si la respuesta fue correcta, si la explicación revela un error conceptual y cuál es ese error.

## Estructura

```
math-misconception-map/
├── notebooks/
│   ├── 01_carga_y_limpieza.ipynb           # estructura, calidad, limpieza y variables derivadas
│   ├── 02_eda.ipynb                        # análisis exploratorio, hallazgos y conclusiones
│   ├── 03_modelado_clasico.ipynb           # entrenamiento y comparación de los modelos clásicos
│   ├── 04_modelado_transformer.ipynb       # afinado de un transformer destilado, requiere GPU
│   └── 05_comparacion_resultados.ipynb     # métricas reunidas y figuras del informe
├── src/
│   ├── config.py                           # rutas, semilla y paleta de color
│   ├── data/make_dataset.py                # descarga desde Kaggle y construcción del conjunto limpio
│   ├── features/build_features.py          # variables derivadas y representaciones del texto
│   ├── models/train_model.py               # catálogo de modelos, métrica MAP@3 y validación
│   ├── models/predict_model.py             # predicción en dos etapas
│   ├── visualization/visualize.py          # estilo común de los gráficos
│   ├── back_end/servicio.py                # capa de servicio de la aplicación
│   └── front_end/app.py                    # interfaz de Streamlit
├── .streamlit/config.toml                  # tema de la aplicación
├── references/                             # investigación, planteamiento y diccionario de datos
├── reports/
│   └── figures/                            # figuras para informes y capturas de la aplicación
├── models/                                 # artefactos entrenados, no se versionan
├── data/raw/                               # datos crudos, no se modifican ni se versionan
├── data/processed/                         # datos limpios, reproducibles, no se versionan
└── requirements.txt
```

## Documentación

| Archivo                                | Contenido                                                                     |
| -------------------------------------- | ----------------------------------------------------------------------------- |
| `references/01_investigacion.md`       | Marco conceptual del problema y técnicas de procesamiento de lenguaje natural |
| `references/02_planteamiento.md`       | Situación problemática, problema científico y objetivos                       |
| `references/03_revision_algoritmos.md` | Revisión bibliográfica de los algoritmos                                      |
| `references/codebook.md`               | Diccionario de datos, variables crudas y derivadas                            |

## Cómo Reproducirlo

Los cuadernos funcionan igual en Google Colab y en una instalación local. Cada uno comienza con una celda de arranque que detecta el entorno, clona el repositorio e instala las dependencias cuando hace falta.

### Opción A: Google Colab

Es la vía recomendada cuando la máquina local tiene poca memoria, y la única posible para el cuaderno 04, que necesita GPU.

**1. Registrar el token de Kaggle como secreto.** En Colab, panel izquierdo, icono de la llave. Agregar un secreto llamado `KAGGLE_ACCESS_TOKEN` con el contenido de `~/.kaggle/access_token` (obtener legacy api credentials [aquí](https://www.kaggle.com/settings/api)) y habilitar el acceso desde el cuaderno. La cuenta debe haber aceptado antes las reglas de la competencia.

**2. Abrir cada cuaderno** desde `Archivo`, `Abrir cuaderno`, pestaña `GitHub`, buscando `Tunchxz/math-misconception-map`.

**3. Ejecutarlos en orden.** El cuaderno 04 requiere activar la GPU en `Entorno de ejecución`, `Cambiar tipo de entorno de ejecución`, opción `T4 GPU`.

| Cuaderno                    | Qué hace                                    | GPU | Duración aproximada |
| --------------------------- | ------------------------------------------- | --- | ------------------- |
| `01_carga_y_limpieza`       | Estructura, calidad y limpieza de los datos | no  | 2 min               |
| `02_eda`                    | Análisis exploratorio                       | no  | 3 min               |
| `03_modelado_clasico`       | Entrena y compara los modelos clásicos      | no  | 15 a 25 min         |
| `04_modelado_transformer`   | Afina un transformer destilado              | sí  | 30 a 40 min         |
| `05_comparacion_resultados` | Reúne las métricas y genera las figuras     | no  | 2 min               |

**4. Recuperar los artefactos.** La última celda del cuaderno 03 descarga `artefactos_modelado.zip`. Para usarlo con la aplicación local:

```bash
unzip -o artefactos_modelado.zip -d /tmp/artefactos
cp -r /tmp/artefactos/models/. models/
cp -r /tmp/artefactos/transformed/. data/transformed/
```

### Opción B: Instalación Local

Requiere **Python 3.12** y [`uv`](https://docs.astral.sh/uv/), más un token de Kaggle en `~/.kaggle/access_token` (obtener legacy api credentials [aquí](https://www.kaggle.com/settings/api)) de una cuenta que haya aceptado las reglas de la competencia. Los cuadernos 01, 02 y 05 corren sin problema; el 03 es lento con poca memoria y el 04 no es viable sin GPU.

```bash
make env        # crea el entorno, instala las dependencias y registra el kernel
make data       # descarga los datos desde la API de Kaggle
make notebooks  # ejecuta los cuadernos en orden
```

### La Aplicación

```bash
make app        # equivale a: streamlit run src/front_end/app.py
```

Permite explorar las variables del conjunto limpio, clasificar explicaciones nuevas con uno o varios modelos a la vez, y comparar el rendimiento de los algoritmos en gráficas interactivas que pueden ocultarse. Necesita `data/processed/train_limpio.parquet` y, para clasificar, los artefactos en `models/`.

## Nota Sobre los Datos

Los datos pertenecen a la competencia y están sujetos a sus reglas. Ni `data/`, ni `models/` se versionan en este repositorio.
