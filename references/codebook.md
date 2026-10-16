# Codebook

Diccionario de datos del proyecto **MAP - Charting Student Math Misunderstandings**.

## Convenciones

**Nivel de medición**

| Nivel             | Qué significa                                              | Operaciones válidas                     |
| ----------------- | ---------------------------------------------------------- | --------------------------------------- |
| **Identificador** | Distingue una entidad. No es una categoría ni admite orden | Comparar igualdad, unir tablas          |
| **Nominal**       | Categorías sin orden                                       | Contar, agrupar, calcular moda          |
| **Lógico**        | Dos valores, verdadero o falso                             | Contar, calcular proporciones           |
| **De razón**      | Numérica con cero absoluto                                 | Todas, incluidas razones y proporciones |
| **Texto libre**   | Contenido no estructurado                                  | Análisis de texto                       |

**Archivos**

| Archivo                               | Filas × columnas | Sección                                     |
| ------------------------------------- | ---------------- | ------------------------------------------- |
| `data/raw/train.csv`                  | 36,696 × 7       | [1.1](#11-datarawtraincsv)                  |
| `data/raw/test.csv`                   | 3 × 5            | [1.2](#12-datarawtestcsv)                   |
| `data/raw/sample_submission.csv`      | 3 × 2            | [1.3](#13-datarawsample_submissioncsv)      |
| `data/processed/train_limpio.parquet` | 36,696 × 24      | [2.1](#21-dataprocessedtrain_limpioparquet) |


## 1. Datos crudos

Los tres archivos provienen de la API de Kaggle y se reconstruyen con `src/descargar_datos.py`. No se modifican nunca.

### 1.1 `data/raw/train.csv`

Único archivo con etiquetas. Cada fila es una respuesta de un estudiante a una pregunta diagnóstica, acompañada de su explicación escrita.

| #   | Variable             | Tipo   | Nivel         | Descripción                                                                                                                                                                                                                        |
| --- | -------------------- | ------ | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `row_id`             | entero | Identificador | Número de fila. Único en todo el archivo. Rango 0 a 36,695. No aparece documentado en la descripción oficial de la competencia                                                                                                     |
| 2   | `QuestionId`         | entero | Identificador | Identifica la pregunta diagnóstica. **15 valores distintos**                                                                                                                                                                       |
| 3   | `QuestionText`       | texto  | Texto libre   | Enunciado de la pregunta. Transcrito desde imagen mediante un proceso de reconocimiento óptico con revisión humana. Conserva notación LaTeX en 12 de las 15 preguntas y una descripción entre corchetes de la figura en 3 de ellas |
| 4   | `MC_Answer`          | texto  | Nominal       | Opción de respuesta seleccionada por el estudiante. **49 valores distintos**, 4 por pregunta. El 96% usa notación LaTeX                                                                                                            |
| 5   | `StudentExplanation` | texto  | Texto libre   | Justificación escrita por el estudiante. **35,726 textos distintos**. Contiene errores de ortografía sin corregir                                                                                                                  |
| 6   | `Category`           | texto  | Nominal       | Relación entre la opción elegida y la explicación. **6 valores**. Ver tabla de valores abajo                                                                                                                                       |
| 7   | `Misconception`      | texto  | Nominal       | Error conceptual identificado. **35 valores distintos**. Vacío en el 73.13% de las filas, siempre y solo cuando `Category` no contiene la palabra `Misconception`                                                                  |

**Valores de `Category`**

| Valor                 | Frecuencia | %     | Significado                                                       |
| --------------------- | ---------- | ----- | ----------------------------------------------------------------- |
| `True_Correct`        | 14,802     | 40.34 | Acertó y explicó bien                                             |
| `False_Misconception` | 9,457      | 25.77 | Falló y la explicación revela un error conceptual                 |
| `False_Neither`       | 6,542      | 17.83 | Falló y la explicación no permite identificar un error conceptual |
| `True_Neither`        | 5,265      | 14.35 | Acertó pero la explicación no muestra comprensión                 |
| `True_Misconception`  | 403        | 1.10  | Acertó pero la explicación revela un error conceptual             |
| `False_Correct`       | 227        | 0.62  | Falló pese a que la explicación es correcta                       |

**Valores de `Misconception`**

Las 35 etiquetas, ordenadas por frecuencia entre las 9,860 filas que la tienen:

| Etiqueta                  | n     | Etiqueta                                 | n   |
| ------------------------- | ----- | ---------------------------------------- | --- |
| `Incomplete`              | 1,454 | `Wrong_Fraction`                         | 273 |
| `Additive`                | 929   | `SwapDividend`                           | 206 |
| `Duplication`             | 704   | `Scale`                                  | 179 |
| `Subtraction`             | 620   | `Not_variable`                           | 154 |
| `Positive`                | 566   | `Firstterm`                              | 107 |
| `Wrong_term`              | 558   | `Adding_terms`                           | 97  |
| `Irrelevant`              | 497   | `Multiplying_by_4`                       | 96  |
| `Wrong_fraction`          | 418   | `FlipChange`                             | 78  |
| `Inversion`               | 414   | `Division`                               | 63  |
| `Mult`                    | 353   | `Definition`                             | 54  |
| `Denominator-only_change` | 336   | `Interior`                               | 50  |
| `Whole_numbers_larger`    | 329   | `Longer_is_bigger`                       | 24  |
| `Adding_across`           | 307   | `Base_rate`                              | 23  |
| `WNB`                     | 299   | `Ignores_zeroes`                         | 23  |
| `Tacking`                 | 290   | `Shorter_is_bigger`                      | 23  |
| `Unknowable`              | 282   | `Inverse_operation`                      | 21  |
|                           |       | `Certainty`                              | 18  |
|                           |       | `Incorrect_equivalent_fraction_addition` | 9   |
|                           |       | `Wrong_Operation`                        | 6   |

**Inconsistencia detectada**: `Wrong_fraction` y `Wrong_Fraction` son la misma etiqueta escrita de dos formas. La primera se usa solo en la pregunta 33471 y la segunda solo en la 31777. Se documenta pero no se corrige en los datos, porque la competencia evalúa contra las etiquetas literales. Unificadas suman 691 observaciones en 2 preguntas y el total baja a 34 etiquetas.

### 1.2 `data/raw/test.csv`

Mismas 5 primeras columnas de `train.csv`, sin `Category` ni `Misconception`. Las 3 filas que trae son un marcador de posición; la competencia lo sustituye por un archivo de aproximadamente 16,000 filas durante la evaluación.

### 1.3 `data/raw/sample_submission.csv`

| Variable                 | Tipo   | Nivel         | Descripción                                                                                                                                                                                                                  |
| ------------------------ | ------ | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `row_id`                 | entero | Identificador | Corresponde al `row_id` de `test.csv`                                                                                                                                                                                        |
| `Category:Misconception` | texto  | Nominal       | Predicción. Concatena `Category` y `Misconception` con dos puntos. Se admiten **hasta tres predicciones separadas por espacio**, ordenadas por confianza. Cuando no hay error conceptual se escribe `NA` tras los dos puntos |

## 2. Datos generados

No están versionados. Se reproducen ejecutando los notebooks en orden.

### 2.1 `data/processed/train_limpio.parquet`

Producido por `notebooks/01_carga_y_limpieza.ipynb`. Conserva las 36,696 observaciones y pasa de 7 a 24 columnas.

**Columnas heredadas de `train.csv` sin cambios**

`row_id`, `QuestionId`, `QuestionText`, `MC_Answer`, `StudentExplanation`, `Category`, `Misconception`. Se documentan en la sección 1.1.

**Columnas nuevas**

| #   | Variable                      | Tipo    | Nivel       | Origen y descripción                                                                                                                                                                            |
| --- | ----------------------------- | ------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `MC_Answer_limpia`            | texto   | Nominal     | `MC_Answer` sin notación LaTeX. Las fracciones pasan a la forma `a/b` y se retiran delimitadores y comandos. Ejemplo: `\( \frac{1}{3} \)` queda como `1/3`                                      |
| 2   | `explicacion_limpia`          | texto   | Texto libre | `StudentExplanation` con espacios normalizados y recortados. Afecta a 2,410 filas. No se corrige ortografía                                                                                     |
| 3   | `explicacion_min`             | texto   | Texto libre | `explicacion_limpia` en minúsculas. Base de todos los análisis léxicos                                                                                                                          |
| 4   | `respuesta_correcta`          | lógico  | Lógico      | Prefijo de `Category`. Verdadero si la opción elegida era la correcta. 20,470 verdaderos, 16,226 falsos                                                                                         |
| 5   | `tipo_explicacion`            | texto   | Nominal     | Sufijo de `Category`. Toma los valores `Correct` (15,029), `Misconception` (9,860) y `Neither` (11,807)                                                                                         |
| 6   | `misconception`               | texto   | Nominal     | `Misconception` con los vacíos sustituidos por la cadena `"NA"`. Sin valores faltantes. 36 valores distintos contando `"NA"`                                                                    |
| 7   | `objetivo`                    | texto   | Nominal     | `Category` y `misconception` unidos por dos puntos. Es la variable que la competencia evalúa. **65 valores distintos**, de los cuales 24 tienen menos de 10 observaciones                       |
| 8   | `respuesta_correcta_pregunta` | texto   | Nominal     | Opción correcta de cada pregunta, reconstruida a partir del prefijo de `Category`. Coincide con la etiqueta en el 99.95 % de las filas; las 18 discrepancias se concentran en la pregunta 31778 |
| 9   | `n_caracteres`                | entero  | De razón    | Longitud de `explicacion_limpia`. Media 69.78, mediana 60, rango 1 a 586                                                                                                                        |
| 10  | `n_palabras`                  | entero  | De razón    | Palabras de `explicacion_limpia`, separadas por espacios. Media 15.36, mediana 13, rango 1 a 177. Correlaciona 0.97 con `n_caracteres`                                                          |
| 11  | `long_media_palabra`          | decimal | De razón    | `n_caracteres` dividido entre `n_palabras`. Media 4.56, rango 1 a 15.67. Correlaciona negativamente con `n_digitos`                                                                             |
| 12  | `n_digitos`                   | entero  | De razón    | Cifras numéricas en `explicacion_limpia`. Media 4.72, mediana 4, rango 0 a 101                                                                                                                  |
| 13  | `tiene_digitos`               | lógico  | Lógico      | Verdadero si `n_digitos` es mayor que cero. 78.3 % de las filas                                                                                                                                 |
| 14  | `tiene_fraccion`              | lógico  | Lógico      | Verdadero si aparece el patrón de una fracción, un dígito seguido de barra y otro dígito. 26.9 %                                                                                                |
| 15  | `tiene_simbolo_mat`           | lógico  | Lógico      | Verdadero si aparece alguno de los símbolos `+ - * = < > % × ÷`. 22.6 %                                                                                                                         |
| 16  | `tiene_interrogacion`         | lógico  | Lógico      | Verdadero si aparece un signo de interrogación. 0.4 %                                                                                                                                           |
| 17  | `tiene_porque`                | lógico  | Lógico      | Verdadero si aparece alguna de las palabras `because`, `cause`, `cos`, `coz`, `as`. 54.6 %                                                                                                      |

**Notas de uso**

- `n_caracteres` y `n_palabras` son casi intercambiables. Para modelar conviene conservar solo una de las dos.
- `Misconception` sigue teniendo vacíos de forma deliberada, para conservar el original. `misconception` es la versión lista para usar.
- `respuesta_correcta_pregunta` permite localizar las 18 filas con anotación inconsistente.
