import streamlit as st
from components.ui import heading
from components.experience import _set_light

st.set_page_config(page_title="QR Numerical Lab", page_icon="∑", layout="wide",
                   initial_sidebar_state="collapsed")
st.session_state.setdefault("lab_light_value", False)
with st.sidebar:
    st.toggle("Tema claro", value=st.session_state.lab_light_value, key="home_light_widget",
              on_change=_set_light, args=("home_light_widget",))
heading("Métodos Numéricos II · Laboratorio interactivo", "QR Numerical Lab",
        "Explora la factorización QR, compara Gram-Schmidt modificado y Householder, y aplica sus resultados a tres problemas numéricos.")

st.markdown("""## Una factorización, tres aplicaciones

**1. Factorizamos A = QR.** Q contiene direcciones ortonormales y R es triangular superior.

**2. Comparamos dos construcciones.** Gram-Schmidt modificado resta proyecciones; Householder aplica reflectores para crear ceros bajo la diagonal.

**3. Aplicamos y comprobamos.** Resolvemos **Rβ = Qᵀy** y distinguimos precisión de QR, calidad del ajuste y sensibilidad del problema.
""")

st.info("En los tres ejemplos, **Resolución con QR** muestra la factorización paso a paso y permite comparar Gram-Schmidt modificado y Householder con los datos del ejercicio. Después sigue Qᵀy y la sustitución regresiva hasta obtener sus coeficientes.")

st.markdown("## Elige un laboratorio")
cols = st.columns(3)
items = [
    ("01", "Regresión inmobiliaria", "¿Cómo estimar un precio con observaciones imperfectas?", "pages/1_regresion_qr.py"),
    ("02", "Estabilidad numérica", "¿Por qué un ruido pequeño puede cambiar mucho los coeficientes?", "pages/2_estabilidad_numerica.py"),
    ("03", "Projectile Mission Control", "¿Cómo reconstruir un vuelo a partir de posiciones medidas?", "pages/3_trayectoria_proyectil.py"),
]
for col, (number, title, description, page) in zip(cols, items):
    with col:
        st.markdown(f"### {number} · {title}")
        st.write(description)
        st.page_link(page, label="Abrir laboratorio →")

st.divider()
st.caption("Métodos QR implementados en Python · Gráficos basados en cálculos reales · SciPy como referencia de validación")
