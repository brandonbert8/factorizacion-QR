import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.linalg import lstsq as scipy_lstsq
from components.ui import COLORS, METHODS, plot_style
from components.experience import animated_plot, fingerprint, insight, story_header
from components.qr_lab import qr_resolution
from experiments.stability import experimental_matrix
from numerical_methods import least_squares
from numerical_methods.metrics import relative_error

st.set_page_config(page_title="Estabilidad | QR Numerical Lab", page_icon="≈", layout="wide",
                   initial_sidebar_state="collapsed")
OBSERVATIONS = (
    'Dos columnas casi iguales contienen poca información independiente, aunque haya más ecuaciones que incógnitas.',
    'El vector verdadero permite distinguir error de recuperación y residuo del sistema.',
    'La condición depende de A; el ruido afecta a y. Una perturbación pequeña puede producir un cambio grande en los coeficientes.',
    'Un error QR pequeño y una Q poco ortogonal pueden coexistir. Cada medida comprueba una propiedad distinta.',
    'La estabilidad pertenece al algoritmo; el condicionamiento pertenece al problema.',
)
scene, reduced = story_header("stability", "Laboratorio 02 · Condicionamiento", "Estabilidad numérica",
    "Dos columnas casi iguales pueden amplificar un ruido diminuto en la solución.", OBSERVATIONS)
for key, default in (("exponent", -6.0), ("noise_exp", -7.0), ("beta1", 2.0),
                     ("beta2", -1.0), ("seed", 17), ("method", METHODS[0])):
    st.session_state.setdefault(f"stability_{key}_value", default)
if scene == 1:
    col1, col2 = st.columns(2)
    st.session_state.stability_beta1_value = col1.number_input("β₁ verdadero", value=st.session_state.stability_beta1_value,
                                                                 key="stability_present_beta1")
    st.session_state.stability_beta2_value = col2.number_input("β₂ verdadero", value=st.session_state.stability_beta2_value,
                                                                 key="stability_present_beta2")
elif scene == 2:
    col1, col2 = st.columns(2)
    st.session_state.stability_exponent_value = col1.slider("log₁₀ ε: acerca las columnas", -12.0, -1.0,
                                                              st.session_state.stability_exponent_value, 0.5,
                                                              key="stability_present_epsilon")
    st.session_state.stability_noise_exp_value = col2.slider("log₁₀ ruido en y", -14.0, -1.0,
                                                               st.session_state.stability_noise_exp_value, 0.5,
                                                               key="stability_present_noise")
elif scene == 3:
    st.session_state.stability_method_value = st.selectbox("Método QR para observar", METHODS,
        index=METHODS.index(st.session_state.stability_method_value), key="stability_present_method")
exponent = st.session_state.stability_exponent_value
noise_exp = st.session_state.stability_noise_exp_value
beta1, beta2 = st.session_state.stability_beta1_value, st.session_state.stability_beta2_value
seed, method = st.session_state.stability_seed_value, st.session_state.stability_method_value
epsilon, noise = 10**exponent, 10**noise_exp

beta = np.array([beta1, beta2])
if np.linalg.norm(beta) == 0:
    st.warning("El vector β de referencia debe ser distinto de cero para calcular errores relativos.")
    st.stop()
a = experimental_matrix(epsilon)
rng = np.random.default_rng(seed)
direction = rng.normal(size=3)
direction /= np.linalg.norm(direction)
y_exact = a @ beta
y = y_exact + noise * direction
try:
    results = {name: least_squares(a, y, name) for name in METHODS}
    exact_results = {name: least_squares(a, y_exact, name) for name in METHODS}
except ValueError as exc:
    st.warning(f"Caso degenerado: {exc}")
    st.stop()
selected = results[method]
condition = float(np.linalg.cond(a))
reference = scipy_lstsq(a, y)[0]

solution_error = relative_error(selected.coefficients, beta)
if scene == 0:
    st.write("Un equipo mide un sistema cuyas dos columnas contienen casi la misma información. Queremos saber cuánto confiar en los coeficientes recuperados.")
    st.latex(r"A_\varepsilon=\begin{bmatrix}1&1\\1&1+\varepsilon\\1&1-\varepsilon\end{bmatrix}")
    insight("Pregunta del experimento", "¿Puede una QR precisa producir una solución sensible al ruido?")
elif scene == 1:
    values = go.Figure()
    for j, color in enumerate([COLORS["data"], COLORS["fit"]]):
        values.add_scatter(x=[1, 2, 3], y=a[:, j], mode="lines+markers", name=f"Columna {j+1}",
                           line=dict(color=color, width=3), marker=dict(size=10))
    plot_style(values, "Fila de la matriz", "Valor de la columna")
    values.update_layout(height=380)
    st.plotly_chart(values, width="stretch", theme=None, alt="Las dos columnas de A epsilon casi coinciden")
    insight("Punto de partida conocido", f"β verdadero = ({beta1:g}, {beta2:g}); así podemos medir el error de recuperación.")
elif scene == 2:
    eps_values = np.logspace(-1, -12, 28)
    conditions = np.array([np.linalg.cond(experimental_matrix(float(e))) for e in eps_values])
    moving = go.Figure(data=[go.Scatter(x=eps_values, y=conditions, mode="lines",
                                        name="Condición κ(A)", line=dict(color=COLORS["data"], width=3)),
                                   go.Scatter(x=[eps_values[0]], y=[conditions[0]], mode="markers",
                                              name="ε actual de la animación", marker=dict(color=COLORS["fit"], size=14))])
    sweep_frames = [go.Frame(name=str(i), data=[go.Scatter(x=eps_values, y=conditions),
                                                  go.Scatter(x=[eps_values[i]], y=[conditions[i]])])
                    for i in range(len(eps_values))]
    plot_style(moving, "ε (columnas más parecidas hacia la derecha)", "Número de condición κ(A)")
    moving.update_layout(height=390)
    moving.update_xaxes(type="log", autorange="reversed")
    moving.update_yaxes(type="log")
    animated_plot(moving, sweep_frames, [f"ε={value:.1e}" for value in eps_values], reduced,
                  "Al disminuir epsilon aumenta el número de condición")
    st.plotly_chart(moving, width="stretch", theme=None, alt="Barrido animado del número de condición frente a epsilon")
    insight("Tus parámetros seleccionados", f"κ(A) = {condition:.2e} · error relativo de β = {solution_error:.2e}")
    if condition > 1 / np.sqrt(np.finfo(float).eps):
        st.warning("Matriz muy mal condicionada: el ruido puede dominar el valor de los coeficientes.")
elif scene == 3:
    qr_resolution(results, method, "stability_qr", fingerprint(a, y, extra=method), reduced, y=y,
                  context=f"Estabilidad: factorizamos Aε con ε = {epsilon:.1e} y resolvemos las observaciones perturbadas.",
                  coefficient_labels=["β₁", "β₂"])
else:
    insight("Conclusión central", "Householder puede conservar mejor Q; ninguno elimina el mal condicionamiento de A.")
    rows = [{"Método": name, "Error de β": relative_error(result.coefficients, beta),
             "Error QR": result.qr.reconstruction_error,
             "Ortogonalidad": result.qr.orthogonality_error} for name, result in results.items()]
    st.dataframe(pd.DataFrame(rows).style.format({"Error de β": "{:.2e}", "Error QR": "{:.2e}",
                                                 "Ortogonalidad": "{:.2e}"}), hide_index=True)
    st.write("El error de β mide sensibilidad del problema. Los otros dos valores comprueban la factorización numérica.")
