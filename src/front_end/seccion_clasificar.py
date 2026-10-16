"""Sección de clasificación de explicaciones nuevas."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.back_end import servicio
from src.front_end import estilo

EJEMPLOS = {
    "Razonamiento correcto": "I divided 24 by 2 which gives 12, so the answer is 12.",
    "Regla mal generalizada": "because two negatives always make a positive",
    "Sin explicación real": "i think this because i worked it out",
    "Comparación de decimales": "0.355 is bigger than 0.8 because 355 is bigger than 8",
}


def _grafico_candidatas(candidatas: list[dict], titulo: str):
    """Construye el gráfico de barras de las candidatas de un modelo."""
    df = pd.DataFrame(candidatas)
    df["probabilidad"] = (df["probabilidad"] * 100).round(2)
    df["categoria"] = df["etiqueta"].str.split(":").str[0]
    fig = px.bar(
        df,
        x="probabilidad",
        y="etiqueta",
        orientation="h",
        color="categoria",
        color_discrete_map=estilo.COLOR_CATEGORIA,
        text="probabilidad",
        labels={"probabilidad": "Confianza (%)", "etiqueta": ""},
    )
    fig.update_traces(texttemplate="%{text:.1f} %", textposition="outside")
    fig.update_layout(
        height=250,
        title=titulo,
        showlegend=False,
        yaxis={"categoryorder": "total ascending"},
        xaxis={"range": [0, max(df["probabilidad"].max() * 1.25, 10)]},
    )
    return fig


def render(datos: pd.DataFrame) -> None:
    """Dibuja la sección de clasificación."""
    st.header("Clasificar una explicación")

    if not servicio.hay_modelos():
        st.markdown(
            estilo.alerta(
                "No se encontró ningún modelo entrenado en <code>models/</code>. "
                "Ejecute antes el cuaderno <code>03_modelado_clasico.ipynb</code>."
            ),
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        estilo.nota(
            "Escriba la justificación de un estudiante y el sistema estimará qué revela sobre "
            "su razonamiento. La limpieza del texto y la construcción de las variables ocurren "
            "de forma automática."
        ),
        unsafe_allow_html=True,
    )

    catalogo = servicio.catalogo_preguntas(datos)
    disponibles = list(servicio.listar_modelos())

    izq, der = st.columns([3, 2])

    with izq:
        ejemplo = st.selectbox(
            "Partir de un ejemplo", ["Escribir desde cero", *EJEMPLOS], index=0
        )
        texto_inicial = EJEMPLOS.get(ejemplo, "")
        explicacion = st.text_area(
            "Explicación del estudiante",
            value=texto_inicial,
            height=120,
            placeholder="because the number with the most digits must be the biggest",
        )

    with der:
        usar_contexto = st.checkbox("Indicar la pregunta y la respuesta elegida", value=True)
        if usar_contexto:
            qid = st.selectbox(
                "Pregunta",
                catalogo["QuestionId"].tolist(),
                format_func=lambda q: f"Pregunta {q}",
            )
            fila = catalogo[catalogo["QuestionId"] == qid].iloc[0]
            respuesta = st.selectbox("Respuesta seleccionada", list(fila["opciones"]))
            enunciado = fila["enunciado"]
        else:
            qid, respuesta, enunciado = -1, "", ""

        seleccion = st.multiselect(
            "Modelos a aplicar", disponibles, default=disponibles,
        )

    if usar_contexto:
        with st.expander("Ver el enunciado de la pregunta"):
            st.write(enunciado)

    if not st.button("Clasificar", type="primary", use_container_width=True):
        return

    if not explicacion.strip():
        st.markdown(
            estilo.alerta("Escriba una explicación antes de clasificar."),
            unsafe_allow_html=True,
        )
        return
    if not seleccion:
        st.markdown(
            estilo.alerta("Seleccione al menos un modelo."), unsafe_allow_html=True
        )
        return

    with st.spinner("Clasificando..."):
        resultados = servicio.clasificar(
            explicacion, seleccion, qid, respuesta, enunciado, k=3
        )

    st.divider()
    st.subheader("Resultado")

    etiquetas_top = [r["candidatas"][0]["etiqueta"] for r in resultados]
    if len(set(etiquetas_top)) == 1:
        st.markdown(
            estilo.exito(
                f"Los {len(resultados)} modelos coinciden en la primera candidata: "
                f"<b>{etiquetas_top[0]}</b>."
            ),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            estilo.alerta(
                "Los modelos <b>no coinciden</b> en la primera candidata. El desacuerdo suele "
                "indicar una explicación ambigua, que es precisamente el caso difícil de esta "
                "tarea."
            ),
            unsafe_allow_html=True,
        )

    if st.checkbox("Ver el detalle por modelo", value=True):
        for r in resultados:
            with st.container(border=True):
                principal = r["candidatas"][0]
                categoria = principal["etiqueta"].split(":")[0]
                st.markdown(
                    f"**{r['modelo']}** &nbsp; "
                    + estilo.chip(categoria, estilo.COLOR_CATEGORIA.get(categoria, "#53606B")),
                    unsafe_allow_html=True,
                )
                a, b = st.columns([3, 2])
                a.plotly_chart(
                    _grafico_candidatas(r["candidatas"], "Las tres candidatas"),
                    use_container_width=True,
                )
                b.metric("Confianza de la primera", f"{principal['probabilidad'] * 100:.1f} %")
                b.caption("Formato de envío a la competencia")
                b.code(r["envio"], language=None)

    if st.checkbox("Ver las variables que el sistema extrajo del texto", value=False):
        st.subheader("Preprocesamiento aplicado")
        st.markdown(
            estilo.nota(
                "Estas variables se calcularon automáticamente a partir del texto y alimentan "
                "a los modelos junto con la representación de las palabras."
            ),
            unsafe_allow_html=True,
        )
        derivadas = pd.DataFrame(
            [
                {"Variable": k, "Valor": v}
                for k, v in resultados[0]["variables_derivadas"].items()
            ]
        )
        st.dataframe(derivadas, use_container_width=True, hide_index=True)

    if st.checkbox("Ver la probabilidad de cada etapa", value=False):
        st.subheader("Las dos etapas del modelo")
        st.markdown(
            estilo.nota(
                "El sistema decide en dos pasos. Primero estima la categoría general y después, "
                "solo si corresponde, el error conceptual específico."
            ),
            unsafe_allow_html=True,
        )
        for r in resultados:
            st.markdown(f"**{r['modelo']}**")
            a, b = st.columns(2)
            cat = (
                pd.Series(r["categoria"]).sort_values(ascending=False).head(6).reset_index()
            )
            cat.columns = ["Categoría", "Probabilidad"]
            fig = px.bar(
                cat, x="Probabilidad", y="Categoría", orientation="h",
                color="Categoría", color_discrete_map=estilo.COLOR_CATEGORIA,
            )
            fig.update_layout(height=260, showlegend=False, title="Etapa 1: categoría")
            a.plotly_chart(fig, use_container_width=True)

            mis = (
                pd.Series(r["misconception"]).sort_values(ascending=False).head(6).reset_index()
            )
            mis.columns = ["Error conceptual", "Probabilidad"]
            fig = px.bar(
                mis, x="Probabilidad", y="Error conceptual", orientation="h",
                color_discrete_sequence=[estilo.COLOR_NEUTRO],
            )
            fig.update_layout(height=260, showlegend=False, title="Etapa 2: error conceptual")
            b.plotly_chart(fig, use_container_width=True)
