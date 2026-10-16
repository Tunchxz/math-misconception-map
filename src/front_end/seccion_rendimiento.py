"""Sección de rendimiento comparado de los modelos."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.back_end import servicio
from src.front_end import estilo

NOMBRE_ESQUEMA = {
    "aleatorio": "Aleatorio (preguntas ya vistas)",
    "agrupado": "Agrupado por pregunta (preguntas nuevas)",
}


def _preparar(metricas: pd.DataFrame) -> pd.DataFrame:
    """Añade la etiqueta legible del esquema de validación."""
    df = metricas.copy()
    df["validacion"] = df["esquema"].map(NOMBRE_ESQUEMA)
    return df


def render() -> None:
    """Dibuja la sección de rendimiento."""
    st.header("Rendimiento de los modelos")

    metricas = servicio.cargar_metricas()
    if metricas.empty:
        st.markdown(
            estilo.alerta(
                "Todavía no hay métricas. Ejecute el cuaderno "
                "<code>03_modelado_clasico.ipynb</code> para generarlas."
            ),
            unsafe_allow_html=True,
        )
        return

    metricas = _preparar(metricas)

    st.markdown(
        estilo.nota(
            "Cada modelo se evaluó de dos formas. En la validación <b>aleatoria</b> las mismas "
            "preguntas aparecen al entrenar y al evaluar. En la <b>agrupada por pregunta</b> se "
            "evalúa sobre preguntas que el modelo nunca vio, que es la situación real de la "
            "competencia. <b>La diferencia entre ambas mide cuánta memorización hay.</b>"
        ),
        unsafe_allow_html=True,
    )

    etapas = sorted(metricas["etapa"].unique())
    etapa = st.selectbox("Etapa del problema", etapas)
    sub = metricas[metricas["etapa"] == etapa]

    if st.checkbox("Mostrar la comparación general", value=True):
        st.subheader("Comparación de los modelos")
        metrica = st.radio(
            "Métrica",
            [m for m in ["map@3", "f1_macro", "f1_ponderado"] if sub[m].notna().any()],
            horizontal=True,
            format_func=lambda m: {
                "map@3": "MAP@3 (métrica oficial)",
                "f1_macro": "F1 macro (trata igual a todas las clases)",
                "f1_ponderado": "F1 ponderado (según frecuencia)",
            }[m],
        )
        datos = sub.dropna(subset=[metrica])
        fig = px.bar(
            datos,
            x=metrica,
            y="modelo",
            color="validacion",
            barmode="group",
            orientation="h",
            color_discrete_map={
                NOMBRE_ESQUEMA["aleatorio"]: estilo.COLOR_ESQUEMA["aleatorio"],
                NOMBRE_ESQUEMA["agrupado"]: estilo.COLOR_ESQUEMA["agrupado"],
            },
            text=metrica,
            labels={metrica: metrica, "modelo": "", "validacion": "Validación"},
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(
            height=470,
            legend={"orientation": "h", "y": -0.15},
            title=f"{metrica} por modelo · {etapa}",
        )
        st.plotly_chart(fig, use_container_width=True)

    if st.checkbox("Mostrar la brecha entre las dos validaciones", value=True):
        st.subheader("Cuánta memorización de la pregunta hay")
        pivote = sub.pivot_table(index="modelo", columns="esquema", values="map@3")
        if {"aleatorio", "agrupado"}.issubset(pivote.columns):
            pivote = pivote.assign(
                brecha=lambda d: (d["aleatorio"] - d["agrupado"]).round(4)
            ).reset_index()
            fig = px.bar(
                pivote.sort_values("brecha"),
                x="brecha",
                y="modelo",
                orientation="h",
                color="brecha",
                color_continuous_scale="OrRd",
                text="brecha",
                labels={"brecha": "Caída de MAP@3 ante preguntas nuevas", "modelo": ""},
            )
            fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig.update_layout(height=400, coloraxis_showscale=False, title="Brecha por modelo")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                estilo.nota(
                    "Una barra larga significa que el modelo depende mucho de haber visto antes "
                    "la pregunta. Una barra corta indica que lo aprendido se transfiere a "
                    "problemas nuevos."
                ),
                unsafe_allow_html=True,
            )

    comparacion = servicio.cargar_comparacion()
    if not comparacion.empty and st.checkbox(
        "Mostrar cuánto aporta leer la explicación", value=True
    ):
        st.subheader("Leer el texto frente a mirar solo la pregunta")
        st.markdown(
            estilo.nota(
                "Una de las líneas base <b>no lee la explicación</b>, solo conoce la pregunta y "
                "la opción marcada. Comparar contra ella revela cuánto aporta realmente el "
                "texto escrito por el estudiante."
            ),
            unsafe_allow_html=True,
        )
        largo = comparacion.melt(
            id_vars=["etapa", "esquema"],
            value_vars=["map3_con_texto", "map3_solo_contexto"],
            var_name="fuente",
            value_name="map3",
        )
        largo["fuente"] = largo["fuente"].map(
            {"map3_con_texto": "Lee la explicación", "map3_solo_contexto": "Solo el contexto"}
        )
        largo["validacion"] = largo["esquema"].map(NOMBRE_ESQUEMA)
        fig = px.bar(
            largo,
            x="etapa",
            y="map3",
            color="fuente",
            barmode="group",
            facet_col="validacion",
            color_discrete_sequence=[estilo.COLOR_ACIERTO, estilo.COLOR_TEXTO_SUAVE],
            text="map3",
            labels={"map3": "MAP@3", "etapa": "", "fuente": ""},
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.update_layout(height=430, legend={"orientation": "h", "y": -0.2})
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(comparacion, use_container_width=True, hide_index=True)

    if st.checkbox("Mostrar el costo de cómputo", value=False):
        st.subheader("Rendimiento frente a tiempo de entrenamiento")
        datos = sub.dropna(subset=["segundos", "map@3"])
        if datos.empty:
            st.markdown(
                estilo.nota("No hay tiempos registrados para esta etapa."),
                unsafe_allow_html=True,
            )
        else:
            fig = px.scatter(
                datos,
                x="segundos",
                y="map@3",
                color="modelo",
                symbol="validacion",
                size_max=18,
                labels={"segundos": "Segundos de validación cruzada", "map@3": "MAP@3"},
            )
            fig.update_traces(marker={"size": 13})
            fig.update_layout(height=430, title="Un modelo más lento no siempre acierta más")
            st.plotly_chart(fig, use_container_width=True)

    if st.checkbox("Mostrar la tabla completa de métricas", value=False):
        st.subheader("Todas las métricas")
        st.dataframe(
            metricas.drop(columns=["esquema"]).round(4),
            use_container_width=True,
            hide_index=True,
        )
