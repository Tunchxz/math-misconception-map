# Revisión Bibliográfica: Algoritmos para la Detección Automática de Errores Conceptuales

Este documento reúne la evidencia publicada sobre qué algoritmos funcionan en problemas parecidos al de la competencia, y concluye con la selección razonada de los modelos que se construyen en este proyecto.

## 1. Alcance y método de búsqueda

Se buscaron trabajos en Google Scholar, IEEE Xplore, ACL Anthology, Springer, ScienceDirect y arXiv, con combinaciones de los términos *short answer grading*, *misconception detection*, *text classification*, *class imbalance*, *TF-IDF*, *transformer* y *knowledge distillation*. Se priorizaron publicaciones revisadas por pares y, dentro de ellas, las que reportan comparaciones directas entre familias de algoritmos.

El criterio de selección fue la cercanía al problema que se enfrenta: clasificación multiclase, muy desbalanceada, sobre textos cortos escritos por estudiantes, con la exigencia adicional de generalizar a preguntas no vistas durante el entrenamiento.

## 2. Familias de algoritmos

### 2.1 Clasificadores lineales sobre representaciones dispersas

Los modelos lineales sobre bolsas de palabras ponderadas son el punto de partida clásico de la clasificación de texto. Joachims (1998) argumentó que las máquinas de vectores de soporte son especialmente adecuadas para esta tarea, porque toleran espacios de alta dimensión, trabajan bien con vectores dispersos y resultan robustas cuando casi todos los términos aportan algo de información.

Lejos de haber quedado obsoletos, Lin et al. (2023) sostienen que los clasificadores lineales siguen siendo una línea base que conviene ejecutar siempre junto a los métodos avanzados. Su argumento es doble: los métodos lineales suelen ser competitivos, eficientes y robustos, y además permiten verificar si el resultado de un modelo complejo es razonable o si refleja un problema de aplicación. Esta recomendación resulta especialmente pertinente aquí, donde los organizadores de la competencia reportan que los modelos preentrenados no dieron buen resultado.

La evidencia empírica reciente apunta en la misma dirección. En una comparación sobre clasificación de sensibilidad documental, la máquina de vectores de soporte obtuvo el mejor desempeño con F1 de 86.95% usando únicamente recursos de CPU, seguida de cerca por la regresión logística con 86.83%.

### 2.2 Naive Bayes y su corrección para clases desbalanceadas

Naive Bayes multinomial es rápido y sencillo, pero Rennie et al. (2003) identificaron un problema serio: cuando una clase tiene muchos más ejemplos de entrenamiento que otra, el algoritmo asigna pesos deficientes a la frontera de decisión, porque un sesgo poco estudiado encoge los pesos de las clases con pocos ejemplos.

Para corregirlo propusieron la formulación por clase complementaria, conocida como Complement Naive Bayes. En lugar de estimar los parámetros de una clase a partir de sus propios datos, los estima a partir de todos los datos *excepto* los de esa clase. Los autores reportan que las correcciones producen un algoritmo rápido y competitivo con el estado del arte de la época, incluidas las máquinas de vectores de soporte.

Dado que el conjunto de esta competencia presenta una razón de 242 a 1 entre la clase más y la menos frecuente, esta variante resulta directamente pertinente.

### 2.3 Ensambles de árboles sobre representaciones reducidas

Los árboles con gradiente potenciado no operan bien sobre matrices dispersas de decenas de miles de columnas, de modo que se suele reducir antes la dimensión. La descomposición en valores singulares truncada aplicada a una matriz TF-IDF es lo que Deerwester et al. (1990) llamaron análisis semántico latente, y produce una representación densa de pocas dimensiones.

Conviene registrar una advertencia. Liu et al. (2004) reportan que el análisis semántico latente global puede no ayudar a la clasificación e incluso reducir la exactitud, porque los temas poco frecuentes, representados por términos poco frecuentes, tienden a ser proyectados fuera como si fueran ruido. Esta observación es preocupante en el presente problema, donde las clases raras son justamente las de mayor interés, y motiva comparar esta vía contra las representaciones dispersas en lugar de asumir su superioridad.

Entre las implementaciones de árboles potenciados, LightGBM (Ke et al., 2017) destaca por su eficiencia en memoria y velocidad, lo que importa dadas las restricciones de cómputo de este proyecto.

### 2.4 Transformers y variantes destiladas

La arquitectura transformer (Vaswani et al., 2017) y su aplicación al preentrenamiento bidireccional (Devlin et al., 2019) redefinieron el campo. Su ventaja frente a las representaciones de frecuencia es que producen representaciones sensibles al contexto, de modo que la misma palabra recibe vectores distintos según la frase que la rodea.

El costo computacional llevó al desarrollo de versiones destiladas. DistilBERT (Sanh et al., 2019) reduce el tamaño del modelo original en un 40%, conserva alrededor del 97% de su capacidad de comprensión del lenguaje y resulta un 60% más rápido. MiniLM (Wang et al., 2020) sigue una estrategia distinta, al destilar las relaciones de autoatención en lugar de los estados ocultos, y alcanza unos 33 millones de parámetros frente a los 66 millones de DistilBERT.

Para este proyecto la diferencia es práctica: un modelo más pequeño se entrena más rápido en los recursos gratuitos de Google Colab y, sobre todo, se carga más fácilmente dentro de una aplicación interactiva.

## 3. Evidencia del dominio educativo

La calificación automática de respuestas abiertas, conocida como *Automatic Short Answer Grading*, es el campo más cercano al problema de esta competencia, y ofrece dos hallazgos que conviene tomar en cuenta.

**Primero, las representaciones aprendidas no bastan por sí solas.** Haller et al. (2022) concluyen en su revisión que el aprendizaje profundo afectó a este campo de forma distinta a otras áreas del procesamiento de lenguaje natural, porque las representaciones aprendidas por sí mismas no alcanzan los mejores resultados. El mejor desempeño corresponde a métodos que combinan características diseñadas a mano con las descripciones semánticas de los modelos recientes. Esto justifica incorporar al modelo las variables derivadas del texto que se construyeron durante la limpieza, en lugar de confiar únicamente en la representación automática.

**Segundo, y más importante para este trabajo, generalizar a preguntas nuevas es un problema abierto.** Condor et al. (2021) estudiaron exactamente esta cuestión, comparando tres esquemas de evaluación: retención aleatoria, retención por pregunta y retención por grupo de preguntas. Su resultado es inequívoco. En todos los casos observaron una exactitud mucho menor en los experimentos con retención por pregunta y por grupo que en el de retención aleatoria, siendo el de grupo el más bajo. Los autores reconocen que sus resultados no son prometedores respecto a la capacidad de generalización de estos modelos a preguntas no vistas.

Este precedente valida directamente el diseño adoptado en el presente proyecto, que reporta los dos esquemas de validación en paralelo. La brecha entre ambos no es un defecto del experimento, es el fenómeno que se desea medir.

En el terreno específico de los errores conceptuales en matemática, Gorgun y Botelho (2023) exploraron métodos de procesamiento de lenguaje natural y agrupamiento, y concluyeron que resultan prometedores para distinguir un error conceptual de su ausencia, pero que requieren más desarrollo para mejorar la interpretabilidad del error específico. Esa distinción entre detectar *si* hay un error e identificar *cuál* es coincide con lo que el análisis exploratorio de este proyecto encontró en los datos.

## 4. Tratamiento del desbalance

Henning et al. (2023) organizan los métodos disponibles en cinco grupos: muestreo, aumento de datos, modificación de la función de pérdida, entrenamiento por etapas y diseño del modelo. Su revisión advierte que elegir un enfoque adecuado para una tarea y un escenario de desbalance concretos es difícil, y que no existe una receta universal.

Entre las técnicas de muestreo, la más citada es el sobremuestreo sintético de la clase minoritaria propuesto por Chawla et al. (2002). Su aplicación al texto es discutible, porque interpola entre vectores y los puntos resultantes no corresponden a ningún texto real.

He y Garcia (2009), en una revisión de referencia sobre aprendizaje con datos desbalanceados, insisten en un punto que resulta decisivo aquí: la exactitud global es una métrica engañosa bajo desbalance severo, ya que un clasificador que siempre predice la clase mayoritaria puede alcanzar valores altos sin aportar nada.

Para este proyecto se adopta la vía más simple y mejor soportada: la ponderación de clases en la función de pérdida, disponible de forma directa en los estimadores empleados, combinada con métricas que no se dejen engañar por el desbalance.

## 5. Métricas de evaluación

De lo anterior se desprenden tres decisiones sobre medición.

- **F1 macro** promedia el F1 de cada clase sin ponderar por su frecuencia, de modo que una clase con seis observaciones pesa lo mismo que una con catorce mil. Es la métrica que revela si el modelo aprendió las clases raras.
- **F1 ponderado** promedia según la frecuencia de cada clase y describe el comportamiento esperado sobre una muestra real.
- **MAP@3**, la precisión promedio media hasta la tercera posición, es la métrica oficial de la competencia. Pertenece a la familia de métricas de recuperación de información descritas por Manning et al. (2008) y premia que la respuesta correcta aparezca entre las tres primeras candidatas, reconociendo así la ambigüedad legítima de la tarea.

Reportar las tres en conjunto permite distinguir un modelo que acierta mucho de uno que acierta en todas las clases.

## 6. Algoritmos seleccionados

A partir de la evidencia revisada se seleccionan cinco modelos.

| Modelo                      | Representación                           | Justificación bibliográfica                                                                                                                                                                                                                        |
| --------------------------- | ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Línea base por contexto** | `QuestionId` y `MC_Answer`, sin texto    | No proviene de la literatura sino del análisis exploratorio propio. Mide cuánto se logra sin leer la explicación, y por tanto cuánta de la señal de los demás modelos es memorización de la pregunta                                               |
| **Regresión logística**     | TF-IDF de palabras y caracteres          | Lin et al. (2023) la recomiendan como línea base obligatoria. Produce probabilidades calibradas, necesarias para construir las tres candidatas de MAP@3, y es interpretable                                                                        |
| **LinearSVC**               | TF-IDF de palabras y caracteres          | Joachims (1998) y la evidencia comparativa reciente la señalan como fuerte en texto disperso de alta dimensión                                                                                                                                     |
| **Complement Naive Bayes**  | TF-IDF de palabras                       | Rennie et al. (2003) la diseñaron específicamente para el escenario de desbalance severo que presentan estos datos                                                                                                                                 |
| **LightGBM**                | SVD sobre TF-IDF más variables derivadas | Aporta capacidad no lineal y permite incorporar las características diseñadas a mano que Haller et al. (2022) señalan como decisivas. La advertencia de Liu et al. (2004) sobre la pérdida de clases raras se somete a prueba en lugar de asumirse |
| **Transformer destilado**   | Tokens contextuales                      | Devlin et al. (2019) establecen el estándar. Se usa una variante destilada por restricciones de cómputo, siguiendo a Sanh et al. (2019) y Wang et al. (2020)                                                                                       |

Se incluye además el uso de n-gramas de caracteres junto a los de palabras, decisión que responde a una característica concreta de estos datos: las explicaciones las escriben estudiantes de entre nueve y dieciséis años y contienen errores de ortografía frecuentes, ante los cuales las subcadenas de caracteres son más robustas que las palabras completas.

## Referencias

- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research, 16*, 321-357. https://doi.org/10.1613/jair.953

- Condor, A., Litster, M., & Pardos, Z. (2021). Automatic short answer grading with SBERT on out-of-sample questions. En *Proceedings of the 14th International Conference on Educational Data Mining* (pp. 345-352). International Educational Data Mining Society. https://educationaldatamining.org/EDM2021/virtual/static/pdf/EDM21_paper_149.pdf

- Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K., & Harshman, R. (1990). Indexing by latent semantic analysis. *Journal of the American Society for Information Science, 41*(6), 391-407. https://doi.org/10.1002/(SICI)1097-4571(199009)41:6<391::AID-ASI1>3.0.CO;2-9

- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. En *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics* (pp. 4171-4186). https://doi.org/10.18653/v1/N19-1423

- Gorgun, G., & Botelho, A. F. (2023). Enhancing the automatic identification of common math misconceptions using natural language processing. En *Artificial Intelligence in Education. Posters and Late Breaking Results* (Vol. 1831, pp. 302-307). Springer. https://doi.org/10.1007/978-3-031-36336-8_47

- Haller, S., Aldea, A., Seifert, C., & Strisciuglio, N. (2022). *Survey on automated short answer grading with deep learning: From word embeddings to transformers* (arXiv:2204.03503). arXiv. https://arxiv.org/abs/2204.03503

- He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering, 21*(9), 1263-1284. https://doi.org/10.1109/TKDE.2008.239

- Henning, S., Beluch, W., Fraser, A., & Friedrich, A. (2023). A survey of methods for addressing class imbalance in deep-learning based natural language processing. En *Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics* (pp. 523-540). https://doi.org/10.18653/v1/2023.eacl-main.38

- Joachims, T. (1998). Text categorization with support vector machines: Learning with many relevant features. En *Machine Learning: ECML-98* (pp. 137-142). Springer. https://doi.org/10.1007/BFb0026683

- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. En *Advances in Neural Information Processing Systems* (Vol. 30, pp. 3146-3154).

- Lin, Y.-C., Chen, S.-A., Liu, J.-J., & Lin, C.-J. (2023). Linear classifier: An often-forgotten baseline for text classification. En *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)* (pp. 1876-1888). https://doi.org/10.18653/v1/2023.acl-short.160

- Liu, T., Chen, Z., Zhang, B., Ma, W.-Y., & Wu, G. (2004). Improving text classification using local latent semantic indexing. En *Proceedings of the 4th IEEE International Conference on Data Mining* (pp. 162-169). IEEE. https://doi.org/10.1109/ICDM.2004.10096

- Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to information retrieval*. Cambridge University Press. https://doi.org/10.1017/CBO9780511809071

- Rennie, J. D. M., Shih, L., Teevan, J., & Karger, D. R. (2003). Tackling the poor assumptions of naive Bayes text classifiers. En *Proceedings of the 20th International Conference on Machine Learning* (pp. 616-623). AAAI Press.

- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter* (arXiv:1910.01108). arXiv. https://arxiv.org/abs/1910.01108

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. En *Advances in Neural Information Processing Systems* (Vol. 30, pp. 5998-6008).

- Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M. (2020). MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers. En *Advances in Neural Information Processing Systems* (Vol. 33, pp. 5776-5788).
