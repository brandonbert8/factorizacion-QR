import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from components.ui import COLORS, METHODS, plot_style
from components.experience import animated_plot, fingerprint, insight, story_header
from components.qr_lab import qr_resolution
from experiments.projectile import physical_quantities, simulated_projectile
from numerical_methods import least_squares
from numerical_methods.teaching import number

st.set_page_config(page_title="Proyectil | QR Numerical Lab", page_icon="↗", layout="wide",
                   initial_sidebar_state="collapsed")
OBSERVATIONS = (
    'Las mediciones definen un sistema sobredeterminado; la trayectoria reconstruida depende de los coeficientes obtenidos por QR.',
    'Cada tiempo genera una fila de A. Tiempos distintos permiten separar posición, velocidad y curvatura.',
    'La trayectoria ideal se conoce por la simulación; la reconstruida se calcula a partir de las mediciones.',
    'Con g estimada se factoriza [1,t,t²]; con g conocida se factoriza [1,t] y se transforma el lado derecho.',
    'El término cuadrático representa −g/2 bajo gravedad constante y sin resistencia del aire.',
)
scene, reduced = story_header("projectile", "Laboratorio 03 · Física experimental", "Projectile Mission Control",
    "De mediciones dispersas a una trayectoria física reconstruida con QR.", OBSERVATIONS)
for key, default in (("v0", 35.0), ("angle", 42.0), ("y0", 3.0), ("gravity", 9.81),
                     ("count", 14), ("noise", 0.8), ("seed", 13),
                     ("mode", "Estimar g de los datos"), ("method", METHODS[0])):
    st.session_state.setdefault(f"projectile_{key}_value", default)
if scene == 1:
    col1, col2, col3 = st.columns(3)
    st.session_state.projectile_v0_value = col1.slider("Velocidad inicial (m/s)", 5.0, 100.0,
        st.session_state.projectile_v0_value, 1.0, key="projectile_present_v0")
    st.session_state.projectile_angle_value = col2.slider("Ángulo (°)", 5.0, 85.0,
        st.session_state.projectile_angle_value, 1.0, key="projectile_present_angle")
    st.session_state.projectile_y0_value = col3.slider("Altura inicial (m)", 0.0, 80.0,
        st.session_state.projectile_y0_value, 0.5, key="projectile_present_y0")
    st.session_state.projectile_count_value = st.slider("Mediciones", 4, 60,
        st.session_state.projectile_count_value, key="projectile_present_count")
    if st.button("Generar otro conjunto de mediciones"):
        st.session_state.projectile_seed_value += 1
elif scene == 2:
    col1, col2, col3 = st.columns(3)
    st.session_state.projectile_noise_value = col1.slider("Ruido de medición (m)", 0.0, 8.0,
        st.session_state.projectile_noise_value, 0.1, key="projectile_present_noise")
    st.session_state.projectile_gravity_value = col2.number_input("g configurada (m/s²)", min_value=0.1,
        value=st.session_state.projectile_gravity_value, step=0.01, key="projectile_present_gravity")
    st.session_state.projectile_mode_value = col3.radio("Tratamiento de g",
        ["Estimar g de los datos", "Usar g conocida"],
        index=0 if st.session_state.projectile_mode_value == "Estimar g de los datos" else 1,
        key="projectile_present_mode")
elif scene == 3:
    st.session_state.projectile_method_value = st.selectbox("Método QR para observar", METHODS,
        index=METHODS.index(st.session_state.projectile_method_value), key="projectile_present_method")
v0, angle, y0 = (st.session_state[f"projectile_{key}_value"] for key in ("v0", "angle", "y0"))
gravity, count, noise = (st.session_state[f"projectile_{key}_value"] for key in ("gravity", "count", "noise"))
mode, method = st.session_state.projectile_mode_value, st.session_state.projectile_method_value
upload = None
data = simulated_projectile(v0, angle, y0, gravity, count, noise,
                            st.session_state.projectile_seed_value)
simulated = True
try:
    data = data.apply(pd.to_numeric, errors="raise")
    if len(data) < 4 or not np.all(np.isfinite(data.to_numpy())) or data.t_s.nunique() < 3:
        raise ValueError("Se necesitan al menos cuatro filas finitas y tres tiempos distintos.")
    t, x, y = (data[column].to_numpy(float) for column in ["t_s", "x_m", "y_m"])
    full_a = np.column_stack([np.ones_like(t), t, t**2])
    full_fit = least_squares(full_a, y, method)
    if mode == "Usar g conocida":
        vertical_a = np.column_stack([np.ones_like(t), t])
        vertical_fit = least_squares(vertical_a, y + 0.5 * gravity * t**2, method)
        c0, c1 = vertical_fit.coefficients
        c2 = -0.5 * gravity
        fitted_y = vertical_a @ vertical_fit.coefficients - 0.5 * gravity * t**2
    else:
        vertical_a = full_a
        vertical_fit = full_fit
        c0, c1, c2 = full_fit.coefficients
        fitted_y = full_fit.fitted
    horizontal_a = np.column_stack([np.ones_like(t), t])
    horizontal_fit = least_squares(horizontal_a, x, method)
except (ValueError, np.linalg.LinAlgError) as exc:
    st.warning(f"No se puede reconstruir la trayectoria: {exc}")
    st.stop()

vx = float(horizontal_fit.coefficients[1])
g_est = float(-2 * c2)
fitted_x = horizontal_fit.fitted
trajectory_rmse = float(np.sqrt(np.mean((x - fitted_x)**2 + (y - fitted_y)**2)))
vertical_rmse = float(np.sqrt(np.mean((y - fitted_y)**2)))
true_vx = v0 * np.cos(np.deg2rad(angle))
true_vy = v0 * np.sin(np.deg2rad(angle))
x_grid = np.linspace(0, float(t.max()), 100)
ideal_x = true_vx * x_grid
ideal_y = y0 + true_vy * x_grid - 0.5 * gravity * x_grid**2
model_x = horizontal_fit.coefficients[0] + vx * x_grid
model_y = c0 + c1 * x_grid + c2 * x_grid**2
if scene == 0:
    st.write("Un sensor observa un proyectil, pero sus puntos no siguen una parábola perfecta. Queremos recuperar la trayectoria y sus parámetros físicos.")
    st.latex(r"x(t)=v_x t,\qquad y(t)=y_0+v_y t-\tfrac12gt^2")
    insight("Pregunta del experimento", "¿Qué movimiento explica mejor mediciones con error?")
elif scene == 1:
    points = go.Figure()
    points.add_scatter(x=ideal_x, y=ideal_y, name="Movimiento ideal simulado",
                       line=dict(color=COLORS["ideal"], dash="dash", width=3))
    points.add_scatter(x=x, y=y, mode="markers", name="Mediciones con ruido",
                       marker=dict(color=COLORS["data"], size=9))
    plot_style(points, "Distancia horizontal (m)", "Altura (m)")
    points.update_layout(height=380)
    st.plotly_chart(points, width="stretch", theme=None, alt="Mediciones ruidosas alrededor de la trayectoria ideal")
    insight("Datos conocidos en la simulación", f"v₀={v0:.1f} m/s · θ={angle:.0f}° · {len(t)} mediciones")
elif scene == 2:
    times = np.linspace(0, float(t.max()), 48)
    trace_ideal = go.Scatter(x=ideal_x, y=ideal_y, name="Ideal simulada",
                             line=dict(color=COLORS["ideal"], dash="dash", width=2))
    trace_model = go.Scatter(x=model_x, y=model_y, name="Reconstruida por QR",
                             line=dict(color=COLORS["fit"], width=3))
    trace_measured = go.Scatter(x=x[:1], y=y[:1], mode="markers", name="Puntos ya medidos",
                                marker=dict(color=COLORS["data"], size=8))
    trace_ball = go.Scatter(x=[model_x[0]], y=[model_y[0]], mode="markers", name="Posición reconstruida",
                            marker=dict(color=COLORS["error"], size=15, symbol="circle"))
    moving = go.Figure(data=[trace_ideal, trace_model, trace_measured, trace_ball])
    motion_frames = []
    for i, current in enumerate(times):
        sampled = t <= current
        motion_frames.append(go.Frame(name=str(i), data=[go.Scatter(x=ideal_x, y=ideal_y),
                                                           go.Scatter(x=model_x, y=model_y),
                                                           go.Scatter(x=x[sampled], y=y[sampled]),
                                                           go.Scatter(x=[horizontal_fit.coefficients[0] + vx*current],
                                                                      y=[c0+c1*current+c2*current**2])]))
    plot_style(moving, "Distancia horizontal (m)", "Altura (m)")
    moving.update_layout(height=390)
    animated_plot(moving, motion_frames, [f"t={moment:.2f} s" for moment in times], reduced,
                  "Punto reconstruido avanza y se revelan mediciones al transcurrir el tiempo")
    st.plotly_chart(moving, width="stretch", theme=None, alt="Animación temporal del proyectil y las mediciones")
    insight("Calidad de la reconstrucción", f"RMSE de la trayectoria = {trajectory_rmse:.3f} m · g usada/estimada = {g_est:.3f} m/s²")
    st.caption("Ideal = parámetros simulados; reconstruida = mínimos cuadrados sobre los puntos observados.")
elif scene == 3:
    system = st.radio("Ecuación de movimiento a resolver", ["Altura y(t)", "Posición horizontal x(t)"],
                      horizontal=True, key="projectile_equation")
    if system == "Altura y(t)":
        target_y = y + 0.5*gravity*t**2 if mode == "Usar g conocida" else y
        comparison = {name: vertical_fit if name == method else least_squares(vertical_a, target_y, name)
                      for name in METHODS}
        qr_resolution(comparison, method, "projectile_qr", fingerprint(vertical_a, target_y, extra=mode+method+system), reduced,
                     y=target_y,
                     context=("Proyectil: A = [1, t]; con g conocida ajustamos y + gt²/2."
                              if mode == "Usar g conocida" else "Proyectil: A = [1, t, t²]; el coeficiente cuadrático estima −g/2."),
                     coefficient_labels=["y₀ (m)", "vᵧ (m/s)"] if mode == "Usar g conocida" else ["y₀ (m)", "vᵧ (m/s)", "−g/2"],
                     model_expression=rf"y(t)=({number(c0)})+({number(c1)})t+({number(c2)})t^2")
    else:
        comparison = {name: horizontal_fit if name == method else least_squares(horizontal_a, x, name)
                      for name in METHODS}
        qr_resolution(comparison, method, "projectile_qr", fingerprint(horizontal_a, x, extra=method+system), reduced,
                     y=x, context="Proyectil: A = [1, t] para recuperar posición inicial y velocidad horizontal.",
                     coefficient_labels=["x₀ (m)", "vₓ (m/s)"],
                     model_expression=rf"x(t)=({number(horizontal_fit.coefficients[0])})+({number(vx)})t")
else:
    insight("Qué recuperamos de las mediciones", f"vₓ = {vx:.2f} m/s · g = {g_est:.2f} m/s² · RMSE = {trajectory_rmse:.2f} m")
    st.write("c₀ aproxima la altura inicial, c₁ la velocidad vertical y c₂ equivale a −g/2 en el modelo ideal. La precisión depende del ruido y de cuántos tiempos se midieron.")
    st.dataframe(pd.DataFrame([{"Magnitud": "Gravedad (m/s²)", "Verdadero": gravity, "Del modelo": g_est},
                               {"Magnitud": "Altura inicial (m)", "Verdadero": y0, "Del modelo": float(c0)},
                               {"Magnitud": "Velocidad horizontal (m/s)", "Verdadero": true_vx, "Del modelo": vx}])
                 .style.format({"Verdadero": "{:.3f}", "Del modelo": "{:.3f}"}), hide_index=True)
    if mode == "Usar g conocida":
        st.caption("La gravedad de esta tabla fue fijada de antemano; no se estimó de las mediciones.")
    st.caption("Supuestos: movimiento plano, gravedad constante y sin resistencia del aire.")
