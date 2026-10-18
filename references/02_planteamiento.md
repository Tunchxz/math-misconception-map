# Planteamiento de la Investigación

## 1. Situación problemática

Cuando un estudiante resuelve mal un ejercicio de matemática, el error rara vez es un accidente. Detrás de muchas respuestas equivocadas hay una forma de razonar coherente pero incorrecta, que el estudiante aplica de manera consistente. Estas estructuras se conocen como errores conceptuales, y son el verdadero objetivo de la enseñanza correctiva. Corregir la respuesta sin corregir el razonamiento deja el problema intacto, y el mismo error reaparece en el siguiente ejercicio.

Para descubrir ese razonamiento no basta con saber qué opción marcó el estudiante. Hace falta leer su explicación. Plataformas como Eedi recogen precisamente eso: preguntas de opción múltiple cuyos distractores fueron diseñados para capturar errores conceptuales específicos, acompañadas de una justificación escrita por el estudiante.

El problema es de escala. Leer y etiquetar cada explicación es una tarea que hoy hace un docente o un anotador entrenado, de forma manual, y que consume un tiempo del que no se dispone. El conjunto de datos de esta competencia requirió quince anotadores capacitados. Una plataforma con millones de respuestas no puede sostener ese costo. El resultado es que la información diagnóstica más valiosa, la explicación del estudiante, se acumula sin ser aprovechada.

A la escala se suman tres dificultades propias del contenido. Primero, los errores conceptuales son sutiles y varían en nivel de detalle, de modo que la frontera entre un error conceptual y una explicación simplemente pobre no siempre es clara ni siquiera entre anotadores humanos. Segundo, unos pocos errores concentran la mayoría de los casos mientras que muchos otros aparecen rara vez, lo que dificulta aprenderlos. Tercero, y quizá lo más relevante, los organizadores reportan que los intentos iniciales con modelos de lenguaje preentrenados no dieron buen resultado, atribuyéndolo a la complejidad del contenido matemático. Estos modelos fueron entrenados sobre texto corriente, no sobre razonamiento matemático escrito por adolescentes.

Existe además un caso especialmente difícil y especialmente valioso: el estudiante que marca la respuesta correcta por un razonamiento equivocado. Para cualquier sistema que evalúe únicamente el resultado, ese estudiante pasa inadvertido y su error queda sin corregir hasta que el problema cambia lo suficiente como para que la regla equivocada falle.

## 2. Problema científico

De la situación anterior se desprende la siguiente pregunta:

> **¿Qué señales presentes en la explicación escrita de un estudiante, en la pregunta diagnóstica que la origina y en la opción de respuesta que seleccionó, permiten identificar de forma automática y generalizable el error conceptual que subyace a su razonamiento?**

El énfasis está en la palabra *generalizable*. No se busca un sistema que reconozca errores en las preguntas que ya vio, sino uno que reconozca el mismo error conceptual cuando este aparece en una pregunta nueva. Esto convierte al problema en algo más que una clasificación de texto: obliga a distinguir entre lo que caracteriza al error conceptual y lo que simplemente caracteriza a la pregunta.

Antes de construir cualquier modelo es necesario responder una pregunta previa, que es la que aborda esta primera fase del trabajo:

> **¿Cómo están estructurados los datos disponibles, qué tan separables son las categorías a partir del texto y qué obstáculos concretos, como desbalance, ambigüedad de etiquetas o dependencia de la pregunta, condicionarán el modelado posterior?**

## 3. Objetivos

### Objetivo general

Caracterizar el conjunto de datos de la competencia *MAP - Charting Student Math Misunderstandings* mediante un análisis exploratorio, con el fin de identificar las señales textuales y estructurales asociadas a cada categoría de error conceptual y de establecer las restricciones que deberá respetar el modelado posterior.

### Objetivos específicos

1. **Describir la estructura y calidad de los datos.** Determinar el número de observaciones y variables, el tipo y nivel de medición de cada una, la cantidad de valores faltantes por columna y la cantidad de registros duplicados, y ejecutar las operaciones de limpieza que de allí se deriven, dejando cada decisión documentada en el codebook.

2. **Cuantificar la distribución de las clases objetivo.** Construir las tablas de frecuencia y proporción de la variable `Category` y de la variable `Misconception`, medir el grado de desbalance mediante la curva de frecuencia acumulada y determinar qué proporción de los casos corresponde al escenario difícil de respuesta correcta con razonamiento equivocado.

3. **Medir la dependencia entre las etiquetas y el contexto de la pregunta.** Calcular la asociación entre `Misconception` y las variables `QuestionId` y `MC_Answer` mediante la V de Cramér, y contar en cuántas preguntas distintas aparece cada error conceptual, para estimar el riesgo de que un modelo aprenda la pregunta en lugar del error.

4. **Identificar las señales textuales discriminantes.** Derivar variables cuantitativas a partir de las explicaciones, estudiar su distribución y su relación con la categoría mediante estadística descriptiva y gráficos, y extraer los términos y secuencias de palabras con mayor poder de separación entre categorías mediante TF-IDF y la prueba chi cuadrado.

5. **Formular conclusiones operativas.** Redactar un conjunto de hallazgos que indiquen de forma explícita qué estrategia de representación del texto, qué esquema de validación y qué tratamiento del desbalance conviene adoptar en la fase de modelado.
