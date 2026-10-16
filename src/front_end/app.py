"""Aplicación de exploración y clasificación de errores conceptuales.

Uso:
    streamlit run src/front_end/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.back_end import servicio  # noqa: E402
from src.front_end import estilo, seccion_clasificar, seccion_datos, seccion_rendimiento  # noqa: E402

SECCIONES = {
    "Explorar los datos": "Qué contiene el conjunto y cómo se distribuyen sus variables",
    "Clasificar una explicación": "Probar los modelos con texto nuevo",
    "Rendimiento de los modelos": "Comparar los algoritmos y su capacidad de generalizar",
}


@st.cache_data(show_spinner="Cargando los datos...")
def _datos():
    """Carga el conjunto limpio una sola vez por sesión."""
    return servicio.cargar_datos()


def main() -> None:
    """Punto de entrada de la aplicación."""
    st.set_page_config(
        page_title="MAP · Errores conceptuales en matemática",
        page_icon="📐",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    estilo.registrar_tema()
    st.markdown(estilo.CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.title("MAP")
        st.caption("Detección de errores conceptuales en matemática")
        seccion = st.radio("Sección", list(SECCIONES), label_visibility="collapsed")
        st.caption(SECCIONES[seccion])
        st.divider()

    st.title("Errores conceptuales en explicaciones de matemática")
    st.caption(
        "Competencia MAP · Charting Student Math Misunderstandings · "
        "Vanderbilt University, The Learning Agency y Eedi"
    )

    try:
        datos = _datos()
    except FileNotFoundError:
        st.markdown(
            estilo.alerta(
                "No se encontró <code>data/processed/train_limpio.parquet</code>. "
                "Ejecute antes <code>notebooks/01_carga_y_limpieza.ipynb</code>."
            ),
            unsafe_allow_html=True,
        )
        return

    if seccion == "Explorar los datos":
        seccion_datos.render(datos)
    elif seccion == "Clasificar una explicación":
        seccion_clasificar.render(datos)
    else:
        seccion_rendimiento.render()

    with st.sidebar:
        st.caption(
            f"{len(datos):,} observaciones · {datos['QuestionId'].nunique()} preguntas · "
            f"{len(servicio.listar_modelos())} modelos entrenados"
        )


if __name__ == "__main__":
    main()
