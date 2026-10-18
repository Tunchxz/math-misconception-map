# Investigación Preliminar: Errores Conceptuales en Matemática y su Detección Automática

Este documento reúne el marco conceptual necesario para abordar la competencia *MAP - Charting Student Math Misunderstandings*. Su propósito es: (1) entender qué fenómeno educativo se está midiendo y (2) conocer qué técnicas de procesamiento de lenguaje natural se usan para reconocer patrones en texto escrito. Ambas cosas orientan qué buscar durante el análisis exploratorio.

## 1. ¿Qué es un error conceptual y por qué no es un simple descuido?

En la literatura de educación matemática se distingue con cuidado entre tres cosas que a simple vista producen el mismo resultado, una respuesta equivocada.

| Fenómeno                               | Naturaleza                                           | Comportamiento esperado                                          |
| -------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------------------- |
| **Descuido o desliz**                  | Error de ejecución puntual, sin causa conceptual     | Aleatorio, no se repite de forma predecible                      |
| **Error procedimental**                | Aplicación incorrecta de un algoritmo aprendido      | Se repite mientras el procedimiento no se corrija                |
| **Error conceptual** (*misconception*) | Estructura de razonamiento coherente pero incorrecta | Sistemático, estable en el tiempo y transferible entre problemas |

El error conceptual es el objeto de interés porque es **sistemático**. Refleja un modelo mental que el estudiante construyó y que aplica de forma consistente. El ejemplo canónico de la competencia lo ilustra bien: quien afirma que 0.355 es mayor que 0.8 porque 355 es mayor que 8 no está distraído, está aplicando a los decimales una regla que sí funciona con los números enteros. Este mecanismo se conoce como sobregeneralización del conocimiento previo, y es una de las fuentes más documentadas de errores conceptuales.

Una consecuencia importante para el modelado: si el error es sistemático, entonces deja huellas regulares en la forma de escribir del estudiante. Esas huellas son justamente lo que un modelo de lenguaje puede aprender a reconocer.

Una segunda consecuencia, menos evidente: un error conceptual puede convivir con una respuesta correcta. El estudiante llega a la opción buena por un camino equivocado. La taxonomía de la competencia recoge exactamente ese caso en la categoría `True_Misconception`, y la literatura reciente lo llama el problema de la respuesta correcta engañosa. Es el caso más difícil de detectar, porque la señal no está en el resultado sino únicamente en la explicación.

## 2. Las preguntas diagnósticas y el papel de los distractores

Los datos provienen de Eedi, una plataforma que usa **Diagnostic Questions**. Una pregunta diagnóstica es una pregunta de opción múltiple con una respuesta correcta y tres distractores, y su rasgo distintivo es que **cada distractor fue diseñado a propósito para capturar un error conceptual concreto**. No son opciones de relleno.

Esto significa que la variable `MC_Answer` no es una categoría cualquiera. Cada opción incorrecta es, por diseño, una hipótesis sobre qué está pensando mal el estudiante. La investigación sobre distractores en matemática reporta que alrededor de dos tercios de los distractores confirman efectivamente la hipótesis de error conceptual para la que fueron construidos.

De aquí se desprende una expectativa concreta y verificable en el análisis exploratorio: **la combinación de pregunta y opción seleccionada debería estar fuertemente asociada con la etiqueta de error conceptual**. Si esa asociación resulta muy fuerte, el modelo podría aprender a predecir mirando solo la opción elegida, sin leer la explicación, lo que no generalizaría a preguntas nuevas. Medir esa asociación es una tarea prioritaria.

## 3. Origen del conjunto de datos y su taxonomía

El conjunto fue construido por la Universidad de Vanderbilt junto con The Learning Agency y Eedi. La taxonomía de categorías fue diseñada por especialistas en cognición matemática y se organiza en tres decisiones encadenadas:

1. ¿La opción seleccionada es correcta? Da el prefijo `True_` o `False_`.
2. ¿La explicación revela un error conceptual? Da el sufijo `Correct`, `Misconception` o `Neither`.
3. Si hay error conceptual, ¿cuál es? Da la etiqueta específica en `Misconception`.

La anotación fue realizada por un equipo de quince anotadores con experiencia en tutoría matemática. Los reportes del proceso señalan dos dificultades que conviene tener presentes al mirar los datos: (1) **etiquetas ruidosas o ambiguas**, porque la frontera entre un error conceptual y una explicación simplemente pobre no siempre es nítida, y (2) un **desbalance fuerte** entre tipos de error conceptual.

La evaluación de la competencia usa MAP@3, una métrica que permite hasta tres predicciones ordenadas por confianza. Esto reconoce de forma explícita que la tarea admite ambigüedad legítima.

## 4. ¿Qué patrones debe reconocer el algoritmo?

De lo anterior se derivan las señales que el análisis exploratorio debe buscar.

- **Señales léxicas de sobregeneralización.** Expresiones como "es más grande porque tiene más números", "al multiplicar siempre crece" o "el de arriba es mayor" delatan reglas aprendidas y mal transferidas. Se espera que ciertos términos y secuencias de palabras se concentren en categorías específicas.

- **Señales de ausencia de razonamiento.** Explicaciones muy cortas del tipo "porque sí" o "lo calculé" probablemente se agrupen en la categoría `Neither`, que recoge lo que no es ni correcto ni un error conceptual identificable.

- **Relación entre la explicación y la opción elegida.** Un error conceptual suele ser consistente con el distractor seleccionado. La coherencia o incoherencia entre ambos es informativa.

- **Transferencia entre preguntas.** El objetivo declarado de la competencia es identificar errores conceptuales que **generalicen entre problemas distintos**. Por tanto interesa medir en cuántas preguntas distintas aparece cada etiqueta. Una etiqueta que solo aparece en una pregunta no es generalizable y su aprendizaje se confunde con memorizar la pregunta.

## 5. Técnicas de procesamiento de lenguaje natural para detectar patrones en texto

Las técnicas usadas para reconocer patrones en lenguaje escrito se pueden ordenar por el tipo de información que capturan.

### 5.1 Representaciones basadas en frecuencia

La **bolsa de palabras** representa cada texto por el conteo de los términos que contiene, e ignora el orden. **TF-IDF** refina ese conteo: pondera cada término por su frecuencia en el documento y lo penaliza si aparece en muchos documentos, de modo que quedan realzadas las palabras que distinguen. Los **n-gramas**, secuencias de dos o tres palabras contiguas, recuperan parte del orden perdido y capturan expresiones como "más grande que" o "no se puede".

Existe también la variante de **n-gramas de caracteres**, que toma subcadenas de longitud fija. Es más robusta ante faltas de ortografía, abreviaturas y formas derivadas de una palabra, algo relevante cuando el texto lo escriben estudiantes de nueve a dieciséis años.

Estas representaciones tienen dos virtudes para un análisis exploratorio: son **interpretables**, porque cada dimensión es una palabra concreta, y son **baratas**, porque se calculan en segundos sin hardware especializado. Su limitación conocida es que producen vectores muy dispersos, y que en textos cortos esa dispersión los vuelve ruidosos.

### 5.2 Selección de términos discriminantes

Una vez construida la matriz de términos, la prueba **chi cuadrado** permite evaluar si la aparición de un término y la pertenencia a una clase son independientes. Los términos con mayor estadístico son los que más separan las clases. Es la herramienta natural para responder, de forma exploratoria y legible, la pregunta de qué palabras caracterizan a cada categoría de error conceptual. La literatura reporta que esta selección reduce la redundancia de la representación y mejora el desempeño de la clasificación.

### 5.3 Representaciones semánticas

Los **embeddings** representan cada palabra o cada texto como un vector denso en el que la cercanía refleja similitud de significado. Los modelos basados en **transformadores**, como BERT, producen representaciones sensibles al contexto y manejan mejor los textos cortos. En detección de errores conceptuales, BERT ha superado a los enfoques basados en reglas en precisión, por ejemplo en un curso de circuitos eléctricos.

Aun así, los organizadores de la competencia advierten que los intentos iniciales con modelos preentrenados **no resultaron exitosos**, probablemente por la complejidad del contenido matemático de las preguntas. Los modelos de lenguaje generales se entrenan sobre prosa, no sobre expresiones matemáticas transcritas por reconocimiento óptico, y el razonamiento que hay que juzgar no es lingüístico sino matemático.

### 5.4 Tratamiento del desbalance

Cuando unas clases son mucho más frecuentes que otras, un clasificador tiende a ignorar las raras. Las respuestas habituales son el uso de métricas ponderadas o macro en lugar de la exactitud simple, la ponderación de clases en la función de pérdida, el remuestreo y la agrupación de las clases de cola larga. Verificar la magnitud del desbalance es por tanto un objetivo directo del análisis exploratorio.

### 5.5 Criterio adoptado en este trabajo

Para la fase exploratoria se privilegian las representaciones basadas en frecuencia. La razón es que el objetivo de esta etapa no es maximizar una métrica sino **entender los datos**, y para eso la interpretabilidad de TF-IDF con chi cuadrado es más valiosa que la capacidad predictiva de un modelo denso. Un modelo que solo indica que dos textos son parecidos no explica qué los hace parecidos; una lista de términos discriminantes sí.

## 6. Implicaciones directas para el análisis exploratorio

La investigación anterior se traduce en preguntas concretas que el análisis debe responder:

1. ¿Cuántas categorías y cuántas etiquetas de error conceptual hay, y qué tan desbalanceadas están?
2. ¿Qué proporción de los casos corresponde a respuestas correctas acompañadas de explicaciones con error conceptual, es decir al caso difícil?
3. ¿Qué tan asociada está la etiqueta con la opción seleccionada y con la pregunta? ¿Hay riesgo de que el modelo aprenda la pregunta en lugar del error conceptual?
4. ¿En cuántas preguntas distintas aparece cada etiqueta de error conceptual?
5. ¿Qué tan largas son las explicaciones y cómo se relaciona la longitud con la categoría?
6. ¿Qué términos y secuencias de palabras distinguen a cada categoría?
7. ¿Hay explicaciones repetidas, y reciben siempre la misma etiqueta?

## Referencias

- The Learning Agency. (s.f.). *Case Study: Math Misconceptions Competition*. https://the-learning-agency.com/the-cutting-ed/article/case-study-math-misconceptions-competition/

- The Learning Agency. (s.f.). *AI in Math: Improving Error Identification and Feedback*. https://the-learning-agency.com/news-press/ai-in-math-improving-error-identification-and-feedback/

- Wang, Z., Lamb, A., Saveliev, E., Cameron, P., Zaykov, Y., Hernández-Lobato, J. M., et al. (2021). *Results and Insights from Diagnostic Questions: The NeurIPS 2020 Education Challenge*. Proceedings of Machine Learning Research, 133. https://arxiv.org/abs/2104.04034

- Uçar, M. B., & Demir, C. (2018). *Rethinking Mathematics Misconceptions*. International Journal of Research in Education and Science. https://files.eric.ed.gov/fulltext/EJ1198004.pdf

- *Analysis of distractors in mathematics questions and their potential to lead misconceptions*. ScienceDirect. https://www.sciencedirect.com/science/article/abs/pii/S1871187124002712

- *Developing Distractors for Mathematics Multiple Choice Items: A Literature Review*. ResearchGate. https://www.researchgate.net/publication/385718064

- Gorgun, G., & Botelho, A. F. (2023). *Enhancing the Automatic Identification of Common Math Misconceptions Using Natural Language Processing*. AIED 2023, CCIS 1831, 302-307. Springer. https://doi.org/10.1007/978-3-031-36336-8_47

- *Artificial intelligence applications in education: Natural language processing in detecting misconceptions* (2024). Education and Information Technologies, Springer. https://doi.org/10.1007/s10639-024-12919-1

- *WIP: Detection of Student Misconceptions of Electrical Circuit Concepts in a Short Answer Question Using NLP*. ResearchGate. https://www.researchgate.net/publication/378349342

- *MiRAGE: Misconception Detection with Retrieval-Guided Multi-Stage Reasoning and Ensemble Fusion*. https://arxiv.org/pdf/2511.01182

- *The Correct Answer Trap: Pedagogically-Grounded Detection and Feedback for Hidden Misconceptions*. https://arxiv.org/pdf/2606.23205

- *Review on Application of Chi-square Statistic in Text Classification in Recent Five Years*. ResearchGate. https://www.researchgate.net/publication/386139212

- *Research on N-grams feature selection methods for text classification*. ResearchGate. https://www.researchgate.net/publication/349300946

- *Embedding generation for text classification of Brazilian Portuguese user reviews: from bag-of-words to transformers*. https://arxiv.org/pdf/2212.00587

- Kaggle. (s.f.). *MAP - Charting Student Math Misunderstandings*. https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings
