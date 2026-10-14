import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sympy import Float, Symbol, latex
from components.ui import COLORS, METHODS, plot_style
from components.experience import animated_plot, fingerprint, insight, story_header
from components.qr_lab import qr_resolution
from experiments.housing import design_matrix, simulated_housing
from numerical_methods import least_squares
from numerical_methods.metrics import fit_metrics

st.set_page_config(page_title="Regresión | QR Numerical Lab", page_icon="⌂", layout="wide",
                   initial_sidebar_state="collapsed")
OBSERVATIONS = (
    'Cada vivienda aporta una fila de A; sus columnas representan los términos del modelo de precio.',
    'La información procede de superficies distintas. Repetir una superficie no crea una nueva dirección independiente.',
    'Cambiar el grado añade una columna a A; cambiar el ruido modifica y. Son efectos distintos sobre el mismo ajuste.',
    'Q aporta direcciones ortonormales; R indica cómo combinarlas para recuperar las columnas de los datos.',
    'Un buen ajuste a estas muestras no implica una buena predicción fuera del intervalo observado.',
)
scene, reduced = story_header("housing", "Laboratorio 01 · Mínimos cuadrados", "Regresión inmobiliaria",
                                     "De puntos medidos a un precio estimado: sigue cómo QR construye el modelo.", OBSERVATIONS)
st.session_state.setdefault("housing_count_value", 16)
st.session_state.setdefault("housing_noise_value", 25000)
st.session_state.setdefault("housing_degree_value", 1)
st.session_state.setdefault("housing_method_value", METHODS[0])
st.session_state.setdefault("housing_target_value", 120.0)
st.session_state.setdefault("housing_seed", 7)
if scene == 1:
    st.session_state.housing_count_value = st.slider("¿Cuántas viviendas observamos?", 5, 40,
                                                      st.session_state.housing_count_value, key="housing_present_count")
    if st.button("Generar otro conjunto simulado"):
        st.session_state.housing_seed += 1
elif scene == 2:
    controls = st.columns(3)
    st.session_state.housing_noise_value = controls[0].slider("Ruido de precios (Bs)", 0, 120000,
                                                               st.session_state.housing_noise_value, 5000,
                                                               key="housing_present_noise")
    st.session_state.housing_degree_value = controls[1].radio("Modelo", [1, 2],
        format_func=lambda v: "Lineal" if v == 1 else "Cuadrático", horizontal=True,
        index=st.session_state.housing_degree_value-1, key="housing_present_degree")
    st.session_state.housing_target_value = controls[2].number_input("Superficie a estimar (m²)", min_value=1.0,
        value=st.session_state.housing_target_value, step=5.0, key="housing_present_target")
elif scene == 3:
    st.session_state.housing_method_value = st.selectbox("Método QR para observar", METHODS,
        index=METHODS.index(st.session_state.housing_method_value), key="housing_present_method")
count = st.session_state.housing_count_value
noise = st.session_state.housing_noise_value
degree = st.session_state.housing_degree_value
method = st.session_state.housing_method_value
target = st.session_state.housing_target_value
upload = None
source = simulated_housing(count, noise, st.session_state.housing_seed)
edited = source
try:
    clean = edited[["superficie_m2", "precio_bob"]].apply(pd.to_numeric, errors="raise").dropna()
    if len(clean) < degree + 1 or (clean.superficie_m2 <= 0).any():
        raise ValueError("Añade al menos grado + 1 observaciones con superficies positivas.")
    x, y = clean.superficie_m2.to_numpy(float), clean.precio_bob.to_numpy(float)
    a = design_matrix(x, degree)
    selected = least_squares(a, y, method)
    comparison = {name: least_squares(a, y, name) for name in METHODS}
except (ValueError, np.linalg.LinAlgError) as exc:
    st.warning(f"No se puede ajustar el modelo: {exc} Revisa superficies repetidas o columnas dependientes.")
    st.stop()

scores = fit_metrics(y, selected.fitted)
prediction_value = (design_matrix(np.array([target]), degree) @ selected.coefficients).item()
plot = go.Figure()
plot.add_scatter(x=x, y=y, mode="markers", name="Viviendas observadas",
                 marker=dict(color=COLORS["data"], size=9))
if scene == 0:
    st.write("Tenemos viviendas de distintas superficies y precios. Queremos una línea o curva que resuma la tendencia para estimar una vivienda nueva.")
    insight("Pregunta del experimento", "¿Cómo encontrar una tendencia cuando las mediciones no caen sobre una línea perfecta?")
    plot_style(plot, "Superficie (m²)", "Precio (Bs)")
    plot.update_layout(height=380)
    st.plotly_chart(plot, width="stretch", theme=None, alt="Precio observado en función de la superficie para viviendas simuladas")
elif scene == 1:
    insight("Qué representan los puntos", f"{len(x)} viviendas simuladas · superficie en m² · precio en bolivianos")
    plot_style(plot, "Superficie (m²)", "Precio (Bs)")
    plot.update_layout(height=380)
    st.plotly_chart(plot, width="stretch", theme=None, alt="Nube de observaciones simuladas")
    st.caption(f"Semilla reproducible: {st.session_state.housing_seed}. Cada cambio vuelve a calcular la solución mediante QR.")
elif scene == 2:
    frames = []
    first = min(len(x), max(degree + 1, 3))
    for k in range(first, len(x) + 1):
        partial = least_squares(a[:k], y[:k], method)
        local_grid = np.linspace(x[:k].min(), x[:k].max(), 120)
        curve = design_matrix(local_grid, degree) @ partial.coefficients
        frames.append(go.Frame(name=str(k), data=[go.Scatter(x=x[:k], y=y[:k], mode="markers"),
                                                 go.Scatter(x=local_grid, y=curve, mode="lines")]))
    plot = go.Figure(data=[go.Scatter(x=x[:first], y=y[:first], mode="markers", name="Datos incluidos",
                               marker=dict(color=COLORS["data"], size=9)),
                           go.Scatter(x=np.linspace(x[:first].min(), x[:first].max(), 120),
                                      y=design_matrix(np.linspace(x[:first].min(), x[:first].max(), 120), degree)
                                        @ least_squares(a[:first], y[:first], method).coefficients,
                                      mode="lines", name="Ajuste recalculado", line=dict(color=COLORS["fit"], width=3))])
    plot_style(plot, "Superficie (m²)", "Precio (Bs)")
    plot.update_layout(height=380)
    animated_plot(plot, frames, [f"{k} viviendas" for k in range(first, len(x)+1)], reduced,
                  "El ajuste se recalcula al incorporar cada vivienda")
    st.plotly_chart(plot, width="stretch", theme=None, alt="Animación del ajuste con observaciones incorporadas progresivamente")
    insight("Resultado con todos los datos", f"Precio estimado: Bs {prediction_value:,.0f} · error típico RMSE: Bs {scores['RMSE']:,.0f}")
    st.caption("Reproduce o recorre la barra manualmente. Cada fotograma recalcula el ajuste por QR con las observaciones mostradas.")
elif scene == 3:
    qr_resolution(comparison, method, "housing_qr", fingerprint(a, y, extra=method), reduced, y=y,
                 context="Regresión: cada fila es una vivienda; A contiene [1, superficie]"
                         + (" y superficie² para el modelo cuadrático." if degree == 2 else "."),
                 coefficient_labels=["a (Bs)", "b (Bs/m²)"] if degree == 1 else ["c₀", "c₁", "c₂"],
                 model_expression="p(x)=" + latex(sum(Float(float(value), 6)*Symbol("x")**power
                                                        for power, value in enumerate(selected.coefficients))))
else:
    insight("Qué aprendimos", f"El modelo explica la tendencia con R² = {scores['R²']:.3f}; no equivale a una tasación real.")
    st.write("Al aumentar el ruido, los puntos se dispersan y el error del ajuste suele crecer. El modelo cuadrático puede reducir el error de entrenamiento, pero también puede exagerar la curvatura.")
    rows = [{"Método": name, "RMSE (Bs)": fit_metrics(y, result.fitted)["RMSE"],
             "Ortogonalidad": result.qr.orthogonality_error} for name, result in comparison.items()]
    st.dataframe(pd.DataFrame(rows).style.format({"RMSE (Bs)": "{:.1f}", "Ortogonalidad": "{:.2e}"}), hide_index=True)
