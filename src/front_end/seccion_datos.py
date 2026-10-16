"""Sección de exploración del conjunto de datos."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import NUMERICAS
from src.front_end import estilo

DESCRIPCION_VARIABLES = {
    "QuestionId": ("Identificador", "Identifica la pregunta diagnóstica. Hay 15 distintas"),
    "QuestionText": ("Texto libre", "Enunciado, transcrito desde imagen"),
    "MC_Answer": ("Nominal", "Opción seleccionada. Cuatro por pregunta"),
    "StudentExplanation": ("Texto libre", "Justificación escrita por el estudiante"),
    "Category": ("Nominal", "Relación entre la opción elegida y la explicación"),
    "Misconception": ("Nominal", "Error conceptual específico, cuando aplica"),
    "n_caracteres": ("De razón", "Longitud de la explicación en caracteres"),
    "n_palabras": ("De razón", "Número de palabras de la explicación"),
    "long_media_palabra": ("De razón", "Caracteres por palabra en promedio"),
    "n_digitos": ("De razón", "Cantidad de cifras numéricas"),
    "tiene_digitos": ("Lógico", "La explicación contiene alguna cifra"),
    "tiene_fraccion": ("Lógico", "Aparece notación de fracción"),
    "tiene_simbolo_mat": ("Lógico", "Aparece un operador matemático"),
    "tiene_interrogacion": ("Lógico", "Aparece un signo de interrogación"),
    "tiene_porque": ("Lógico", "La explicación enuncia una causa"),
}

USADAS_EN_MODELOS = set(NUMERICAS) | {"StudentExplanation", "QuestionId", "MC_Answer"}


def _tabla_variables(datos: pd.DataFrame) -> pd.DataFrame:
    """Construye el diccionario de variables con su papel en los modelos."""
    filas = []
    for nombre, (nivel, descripcion) in DESCRIPCION_VARIABLES.items():
        if nombre not in datos.columns:
            continue
        filas.append(
            {
                "Variable": nombre,
                "Nivel de medición": nivel,
                "Valores distintos": int(datos[nombre].nunique()),
                "Usada en los modelos": "sí" if nombre in USADAS_EN_MODELOS else "no",
                "Descripción": descripcion,
            }
        )
    return pd.DataFrame(filas)


def render(datos: pd.DataFrame) -> None:
    """Dibuja la sección de exploración."""
    st.header("Explorar el conjunto de datos")
    st.markdown(
        estilo.nota(
            "Los datos provienen de la plataforma Eedi. Cada fila es la respuesta de un "
            "estudiante a una pregunta de opción múltiple, acompañada de la explicación que "
            "escribió para justificarla. Las variables numéricas no venían en el archivo "
            "original, fueron derivadas del texto durante la limpieza."
        ),
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("Filtros")
        preguntas = sorted(datos["QuestionId"].unique())
        sel_preguntas = st.multiselect(
            "Preguntas", preguntas, default=preguntas, format_func=lambda q: f"Pregunta {q}"
        )
        categorias = sorted(datos["Category"].unique())
        sel_categorias = st.multiselect("Categorías", categorias, default=categorias)

    filtrado = datos[
        datos["QuestionId"].isin(sel_preguntas) & datos["Category"].isin(sel_categorias)
    ]

    if filtrado.empty:
        st.markdown(
            estilo.alerta("Ningún registro cumple los filtros. Amplíe la selección."),
            unsafe_allow_html=True,
        )
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observaciones", f"{len(filtrado):,}")
    c2.metric("Preguntas", filtrado["QuestionId"].nunique())
    c3.metric("Errores conceptuales", filtrado["misconception_unificada"].nunique() - 1)
    c4.metric("Palabras por explicación", f"{filtrado['n_palabras'].median():.0f}")

    st.divider()

    if st.checkbox("Mostrar el diccionario de variables", value=True):
        st.subheader("Variables disponibles")
        st.dataframe(_tabla_variables(filtrado), use_container_width=True, hide_index=True)

    if st.checkbox("Mostrar la distribución de las clases", value=True):
        st.subheader("Qué revela cada explicación")
        izq, der = st.columns([3, 2])

        conteo = (
            filtrado["Category"].value_counts().rename_axis("Category").reset_index(name="n")
        )
        conteo["porcentaje"] = (conteo["n"] / len(filtrado) * 100).round(2)
        fig = px.bar(
            conteo,
            x="n",
            y="Category",
            orientation="h",
            color="Category",
            color_discrete_map=estilo.COLOR_CATEGORIA,
            text="porcentaje",
            labels={"n": "Observaciones", "Category": ""},
        )
        fig.update_traces(texttemplate="%{text:.1f} %", textposition="outside")
        fig.update_layout(showlegend=False, height=330, title="Frecuencia de cada categoría")
        izq.plotly_chart(fig, use_container_width=True)

        der.markdown(
            estilo.nota(
                "<b>True</b> y <b>False</b> indican si la opción marcada era la correcta. "
                "<b>Correct</b>, <b>Misconception</b> y <b>Neither</b> indican qué revela la "
                "explicación.<br><br>La categoría <b>True_Misconception</b>, acertar razonando "
                "mal, es la más valiosa para un docente y la más escasa en los datos."
            ),
            unsafe_allow_html=True,
        )

    if st.checkbox("Mostrar el desbalance de los errores conceptuales", value=True):
        st.subheader("Cuán desiguales son los errores conceptuales")
        con_error = filtrado[filtrado["misconception_unificada"] != "NA"]
        if con_error.empty:
            st.markdown(
                estilo.alerta("La selección actual no contiene errores conceptuales."),
                unsafe_allow_html=True,
            )
        else:
            frec = (
                con_error["misconception_unificada"]
                .value_counts()
                .rename_axis("error")
                .reset_index(name="n")
            )
            frec["acumulado"] = (frec["n"].cumsum() / frec["n"].sum() * 100).round(1)
            fig = px.bar(
                frec.head(20),
                x="n",
                y="error",
                orientation="h",
                labels={"n": "Observaciones", "error": ""},
                color_discrete_sequence=[estilo.COLOR_NEUTRO],
            )
            fig.update_layout(
                height=460,
                yaxis={"categoryorder": "total ascending"},
                title="Los 20 errores conceptuales más frecuentes",
            )
            st.plotly_chart(fig, use_container_width=True)
            n50 = int((frec["acumulado"] < 50).sum()) + 1
            st.markdown(
                estilo.nota(
                    f"Bastan <b>{n50} de {len(frec)}</b> etiquetas para cubrir la mitad de los "
                    "casos. Las demás son tan escasas que resultan muy difíciles de aprender."
                ),
                unsafe_allow_html=True,
            )

    if st.checkbox("Mostrar las variables derivadas del texto", value=False):
        st.subheader("Cómo se relaciona la forma de escribir con la categoría")
        variable = st.selectbox(
            "Variable a examinar",
            ["n_palabras", "n_caracteres", "n_digitos", "long_media_palabra"],
            format_func=lambda v: f"{v} · {DESCRIPCION_VARIABLES[v][1]}",
        )
        fig = px.box(
            filtrado,
            x=variable,
            y="Category",
            color="Category",
            color_discrete_map=estilo.COLOR_CATEGORIA,
            points=False,
            labels={"Category": ""},
        )
        fig.update_layout(showlegend=False, height=380, title=f"{variable} según la categoría")
        st.plotly_chart(fig, use_container_width=True)

        medianas = (
            filtrado.groupby("Category")[variable].median().sort_values().round(2)
        )
        st.markdown(
            estilo.nota(
                f"La mediana va de <b>{medianas.iloc[0]:.0f}</b> en "
                f"<i>{medianas.index[0]}</i> a <b>{medianas.iloc[-1]:.0f}</b> en "
                f"<i>{medianas.index[-1]}</i>. Quien comprende tiende a escribir más y a "
                "apoyarse más en el cálculo."
            ),
            unsafe_allow_html=True,
        )

    if st.checkbox("Mostrar ejemplos reales", value=False):
        st.subheader("Explicaciones del conjunto")
        n = st.slider("Cuántos ejemplos", 5, 50, 10, step=5)
        muestra = filtrado.sample(min(n, len(filtrado)), random_state=7)
        st.dataframe(
            muestra[
                ["QuestionId", "MC_Answer_limpia", "StudentExplanation", "Category", "Misconception"]
            ].rename(
                columns={
                    "MC_Answer_limpia": "Respuesta elegida",
                    "StudentExplanation": "Explicación",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
