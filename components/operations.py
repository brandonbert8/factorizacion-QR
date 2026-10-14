"""Visualización de operaciones vectoriales y aritmética de la solución."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from numerical_methods.teaching import TeachingStep, number, vector_latex
from .ui import COLORS, matrix, plot_style


def operation_figure(step: TeachingStep) -> tuple[go.Figure, list[go.Frame]]:
    """Los extremos son valores ejecutados; los estados interiores ilustran la operación."""
    if step.kind in ("producto", "sustituir"):
        x = np.arange(1, len(step.before)+1)
        frames = []
        if step.kind == "producto":
            fig = go.Figure([go.Bar(x=x, y=np.zeros(len(x)), name="Productos qᵢ · yᵢ", marker_color=COLORS["data"]),
                             go.Scatter(x=x, y=np.full(len(x), np.nan), name="Suma acumulada", yaxis="y2",
                                        line=dict(color=COLORS["fit"], width=3), mode="lines+markers")])
            for count in range(1, len(x)+1):
                products = np.where(np.arange(len(x)) < count, step.before, 0.)
                cumulative = np.where(np.arange(len(x)) < count, step.after, np.nan)
                frames.append(go.Frame(name=str(count), data=[go.Bar(y=products), go.Scatter(y=cumulative)]))
            plot_style(fig, "Observación i", "Aporte al producto escalar")
            fig.update_layout(yaxis2=dict(title="Suma acumulada", overlaying="y", side="right", showgrid=False))
        else:
            colors = [COLORS["fit"] if j == step.values["row"] else COLORS["data"] for j in range(len(x))]
            fig = go.Figure(go.Bar(x=[f"β{j}" for j in x], y=step.before, marker_color=colors, name="Coeficientes calculados"))
            for k, fraction in enumerate(np.linspace(0, 1, 24)):
                frames.append(go.Frame(name=str(k), data=[go.Bar(y=step.before + fraction*(step.after-step.before))]))
            plot_style(fig, "Coeficiente", "Valor")
        return fig, frames

    # Fijar el eje por su significado evita que la SVD gire arbitrariamente
    # la escena: q en MGS y e₁ en la columna objetivo de Householder.
    if step.kind == "proyectar":
        primary = step.values["q"]
        x_label = "Componente paralela a q"
    elif step.kind in ("reflector", "reflejar"):
        primary = step.auxiliary
        if step.kind == "reflector" or step.values["column"] == step.values["target_column"]:
            primary = np.eye(1, len(step.before)).ravel()
            x_label = "Eje e₁ de la subcolumna activa"
        else:
            x_label = "Dirección normal v"
    else:
        primary = step.before
        x_label = "Dirección del vector residual"
    first = primary / np.linalg.norm(primary)
    candidates = [step.before, step.after]
    if step.auxiliary is not None:
        candidates.append(step.auxiliary)
    residuals = [v - first * (first @ v) for v in candidates]
    second = max(residuals, key=np.linalg.norm)
    if np.linalg.norm(second) <= 1e-13 * max(np.linalg.norm(v) for v in candidates):
        fallback = np.eye(len(first))[np.argmin(np.abs(first))]
        second = fallback - first * (first @ fallback)
    second_norm = np.linalg.norm(second)
    second = second / second_norm if second_norm > 1e-14 else np.zeros_like(first)
    basis = np.column_stack([first, second])
    def coordinates(vector):
        value = np.asarray(vector) @ basis
        return np.pad(value, (0, max(0, 2-len(value))))[:2]
    start, end = coordinates(step.before), coordinates(step.after)
    extent = max(np.linalg.norm(start), np.linalg.norm(end), 1e-12) * 1.3
    fig = go.Figure()
    fig.add_scatter(x=[0, start[0]], y=[0, start[1]], name="Antes", mode="lines+markers",
                    line=dict(color=COLORS["data"], width=4, dash="dash"), marker=dict(size=[3, 11]))
    fig.add_scatter(x=[0, start[0]], y=[0, start[1]], name="Resultado en construcción", mode="lines+markers",
                    line=dict(color=COLORS["error"], width=4), marker=dict(size=[3, 13]))
    fig.add_scatter(x=[end[0]], y=[end[1]], name="Resultado final", mode="markers",
                    marker=dict(color=COLORS["error"], size=13, symbol="circle-open"))
    if step.auxiliary is not None:
        auxiliary = coordinates(step.auxiliary)
        if step.kind in ("reflector", "reflejar"):
            normal = auxiliary / max(np.linalg.norm(auxiliary), 1e-30)
            mirror = np.array([-normal[1], normal[0]]) * extent
            fig.add_scatter(x=[-mirror[0], mirror[0]], y=[-mirror[1], mirror[1]], name="Espejo H",
                            mode="lines", line=dict(color=COLORS["ideal"], width=3, dash="dot"))
            auxiliary = normal * extent * .55
            name = "Normal v del espejo (dirección)"
        else:
            name = "Proyección que se resta"
        fig.add_scatter(x=[0, auxiliary[0]], y=[0, auxiliary[1]], name=name, mode="lines+markers",
                        line=dict(color=COLORS["fit"], width=3), marker=dict(size=[3, 9]))
    if step.kind == "proyectar":
        projection = coordinates(step.auxiliary)
        fig.add_scatter(x=[start[0], end[0]], y=[start[1], end[1]], name="Restar la proyección",
                        mode="lines", line=dict(color=COLORS["fit"], width=2, dash="dot"))
        fig.add_scatter(x=[projection[0], start[0]], y=[projection[1], start[1]],
                        name="Descomposición: original = proyección + residuo", mode="lines",
                        line=dict(color=COLORS["ideal"], width=2, dash="dot"))
        size = extent * .06
        fig.add_scatter(x=[size, size, 0], y=[0, size, size], mode="lines", showlegend=False,
                        line=dict(color=COLORS["ideal"], width=2))
    elif step.kind in ("reflector", "reflejar"):
        fig.add_scatter(x=[start[0], end[0]], y=[start[1], end[1]], name="Segmento perpendicular al espejo",
                        mode="lines", line=dict(color=COLORS["fit"], width=2, dash="dot"))
    if step.kind == "proyectar" and 0 < np.linalg.norm(end) < .05*np.linalg.norm(start):
        magnification = .65*np.linalg.norm(start)/np.linalg.norm(end)
        enlarged = end*magnification
        fig.add_scatter(x=[0, enlarged[0]], y=[0, enlarged[1]], mode="lines+markers",
                        name=f"Residuo ampliado ×{magnification:.2g}",
                        line=dict(color=COLORS["ideal"], width=3, dash="dot"), marker=dict(size=[3, 8]))
    frames = []
    for i, fraction in enumerate(np.linspace(0, 1, 30)):
        point = start + fraction*(end-start)
        frames.append(go.Frame(name=str(i), traces=[1], data=[go.Scatter(x=[0, point[0]], y=[0, point[1]])]))
    plot_style(fig, x_label, "Componente perpendicular en el plano activo")
    fig.update_xaxes(range=[-extent, extent], zeroline=True)
    fig.update_yaxes(range=[-extent, extent], zeroline=True, scaleanchor="x", scaleratio=1)
    return fig, frames


def render_operation(step: TeachingStep, reduced: bool, animate, autoplay: bool = False):
    st.subheader(step.title)
    st.write(step.explanation)
    fig, frames = operation_figure(step)
    fig.update_layout(height=470, legend=dict(orientation="h", y=1.42, font=dict(size=11)))
    labels = ([str(i+1) for i in range(len(frames))] if step.kind == "producto"
              else [f"{int(100*i/(len(frames)-1))}%" for i in range(len(frames))])
    animate(fig, frames, labels, reduced, step.explanation)
    fig.update_layout(margin=dict(t=120, b=105, l=60, r=25))
    if fig.layout.updatemenus:
        fig.layout.updatemenus[0].buttons[0].label = "▶ Animar operación"
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 60
        fig.layout.updatemenus[0].buttons[0].args[1]["fromcurrent"] = False
    if reduced:
        # Presentar el resultado sin exigir reproducir movimiento.
        for trace, data in zip(frames[-1].traces or range(len(frames[-1].data)), frames[-1].data):
            fig.data[trace].update(data.to_plotly_json())
        fig.layout.sliders[0].active = len(frames)-1
    if autoplay and not reduced:
        st.iframe(fig.to_html(full_html=False, include_plotlyjs=True, auto_play=True,
                                    animation_opts={"frame": {"duration": 60, "redraw": True},
                                                    "transition": {"duration": 0}, "mode": "immediate"}),
                  height=500, alt=step.title + ". " + step.explanation)
    else:
        st.plotly_chart(fig, width="stretch", theme=None, alt=step.title + ". " + step.explanation)
    st.latex(step.formula)
    observations = {
        "normalizar": "La norma se conserva en rₖₖ: normalizar cambia la escala, pero Q y R juntas retienen la información original.",
        "proyectar": "La componente que se resta queda guardada en R. Esa es la conexión entre ortogonalizar y poder reconstruir A.",
        "reflector": "HᵀH = I y H² = I: el espejo conserva distancias y se deshace aplicándolo de nuevo.",
        "reflejar": "Se trabaja solo en el bloque pendiente para conservar los ceros creados por los reflectores anteriores.",
        "producto": "Qᵀy expresa la proyección de las observaciones en las direcciones de Q; el residuo perpendicular queda fuera de ese subespacio.",
        "sustituir": "La forma triangular de R permite empezar por la última incógnita y avanzar sin resolver otro sistema general.",
    }
    st.caption(observations[step.kind])
    if len(step.before) > 3 and step.kind not in ("producto", "sustituir"):
        st.caption("La fórmula abrevia los vectores después de tres componentes. La animación usa todas las componentes.")
    if step.kind == "sustituir":
        row = step.values["row"]
        r, rhs = step.values["R"], step.values["rhs"]
        terms = "+".join(rf"({number(r[row,j])})\beta_{{{j+1}}}" for j in range(row, len(rhs)))
        st.latex(terms + "=" + number(rhs[row]))
    if step.kind in ("normalizar", "proyectar", "reflector", "reflejar"):
        st.caption("Los vectores se dibujan en su propio plano mediante coordenadas ortonormales: se conservan longitudes y ángulos. La transición muestra gradualmente la operación; los extremos son los valores calculados.")
    if step.kind == "proyectar":
        st.caption(f"Normas: vector original {np.linalg.norm(step.before):.6g} · proyección {np.linalg.norm(step.auxiliary):.6g} · residuo {np.linalg.norm(step.after):.6g}.")
        st.success(f"Comprobación: qᵀ · residuo = {step.values['orthogonality']:.3e} ≈ 0.")
    elif step.kind == "normalizar":
        st.success(f"Comprobación: ||q||₂ = {np.linalg.norm(step.after):.6f}.")
    elif step.kind == "reflejar":
        st.success(f"Longitud antes: {step.values['norm_before']:.6g}; después: {step.values['norm_after']:.6g}. El reflector conserva la norma.")
    if step.kind in ("reflector", "reflejar"):
        st.caption("Una reflexión es una transformación discreta. El recorrido del punto ilustra la resta 2v(vᵀx); sus posiciones intermedias no son reflectores y no conservan la norma. El espejo conserva la longitud entre los dos extremos.")
    if step.kind == "producto":
        st.success(f"La suma termina en {step.values['rhs']:.6g}: esta es una entrada de Qᵀy.")
    elif step.kind == "sustituir":
        st.success(f"β{step.values['row']+1} = {step.after[step.values['row']]:.6g}")
    with st.expander("Consultar todos los valores de esta operación"):
        matrix("Antes", step.before)
        matrix("Después", step.after)
        if step.auxiliary is not None:
            matrix("Proyección" if step.kind == "proyectar" else "Vector v", step.auxiliary)
