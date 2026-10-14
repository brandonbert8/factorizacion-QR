"""Recorrido guiado y animaciones basadas en estados numéricos reales."""

from hashlib import blake2b
from time import monotonic
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from numerical_methods.qr import QRResult
from numerical_methods.teaching import factorization_steps, solution_steps
from .operations import render_operation
from .ui import COLORS, heading, matrix, plot_style, theme

SCENES = ("Problema", "Datos", "Experimento", "Resolución con QR", "Conclusión")


def _move_scene(state_key: str, delta: int):
    st.session_state[state_key] = min(max(st.session_state[state_key] + delta, 0), len(SCENES) - 1)


def _set_scene(state_key: str, value: int):
    st.session_state[state_key] = value


def _set_light(widget_key: str):
    st.session_state.lab_light_value = st.session_state[widget_key]


def fingerprint(*arrays: np.ndarray, extra: str = "") -> str:
    digest = blake2b(digest_size=12)
    digest.update(extra.encode())
    for array in arrays:
        value = np.ascontiguousarray(np.asarray(array))
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def story_header(key: str, kicker: str, title: str, summary: str,
                 observations: tuple[str, ...]) -> tuple[int, bool]:
    st.session_state.setdefault("lab_light_value", False)
    stage_key = f"{key}_scene"
    st.session_state.setdefault(stage_key, 0)
    if st.session_state[stage_key] == 3:
        theme()
        st.caption(kicker)
        st.subheader(title + " · Resolución QR")
    else:
        heading(kicker, title, summary)
    resolution, appearance = st.columns([3, 1])
    if st.session_state[stage_key] != 3:
        resolution.button("Resolver con QR →", key=f"{key}_resolve", type="primary",
                          on_click=_set_scene, args=(stage_key, 3))
    else:
        resolution.caption("Factorización → Qᵀy → sustitución regresiva → coeficientes")
    with appearance:
        with st.popover("Apariencia y movimiento"):
            light_key = f"{key}_light_widget"
            st.toggle("Tema claro", value=st.session_state.lab_light_value, key=light_key,
                      on_change=_set_light, args=(light_key,))
            reduced = st.toggle("Reducir movimiento", value=False, key=f"{key}_reduced")
    stage = min(max(int(st.session_state[stage_key]), 0), len(SCENES) - 1)
    if stage != 3:
        st.progress((stage + 1) / len(SCENES), text=f"Escena {stage+1} de {len(SCENES)} · {SCENES[stage]}")
    previous, middle, following = st.columns([1, 3, 1])
    previous.button("← Anterior", key=f"{key}_previous", disabled=stage == 0, width="stretch",
                    on_click=_move_scene, args=(stage_key, -1))
    with middle:
        st.markdown(f"<div class='scene-title'>{SCENES[stage]}</div>", unsafe_allow_html=True)
    following.button("Siguiente →", key=f"{key}_following", disabled=stage == len(SCENES) - 1,
                     width="stretch", on_click=_move_scene, args=(stage_key, 1))
    st.caption(observations[stage])
    return stage, reduced


def insight(question: str, answer: str):
    st.markdown(f"<div class='insight'><span>{question}</span><strong>{answer}</strong></div>",
                unsafe_allow_html=True)


def qr_animation(qr: QRResult, key: str, signature: str, reduced_motion: bool = False,
                 *, y: np.ndarray | None = None, coefficient_labels: list[str] | None = None,
                 model_expression: str | None = None):
    """Reproduce la factorización y toda la resolución de mínimos cuadrados."""
    steps = factorization_steps(qr) if y is None else solution_steps(qr, y)
    tick_key = f"{key}_last_tick"
    signature_key, index_key, play_key, scrub_key = (f"{key}_{suffix}" for suffix in
                                                    ("signature", "frame", "playing", "scrub"))
    if st.session_state.get(signature_key) != signature:
        st.session_state[signature_key] = signature
        st.session_state[index_key] = 0
        st.session_state[play_key] = False
        st.session_state[scrub_key] = 0
        st.session_state[tick_key] = monotonic()
    if reduced_motion:
        st.session_state[play_key] = False

    def scrubbed():
        st.session_state[index_key] = st.session_state[scrub_key]
        st.session_state[play_key] = False

    @st.fragment(run_every=None if reduced_motion else 6.0, key=f"{key}_fragment")
    def viewer():
        index = min(st.session_state[index_key], len(steps)-1)
        if st.session_state[play_key] and monotonic() - st.session_state[tick_key] >= 6.0:
            st.session_state[tick_key] = monotonic()
            if index < len(steps)-1:
                index += 1
            else:
                st.session_state[play_key] = False
        back, play, next_, reset = st.columns(4)
        if back.button("← Paso", key=f"{key}_back", disabled=index == 0, width="stretch"):
            index -= 1
            st.session_state[play_key] = False
        if not reduced_motion:
            if play.button("Pausar" if st.session_state[play_key] else "▶ Reproducir",
                           key=f"{key}_play", width="stretch"):
                st.session_state[play_key] = not st.session_state[play_key]
                st.session_state[tick_key] = monotonic()
        else:
            play.caption("Avance manual")
        if next_.button("Paso →", key=f"{key}_next", disabled=index == len(steps)-1, width="stretch"):
            index += 1
            st.session_state[play_key] = False
        if reset.button("Reiniciar", key=f"{key}_reset", width="stretch"):
            index = 0
            st.session_state[play_key] = False
        st.session_state[index_key] = index
        st.session_state[scrub_key] = index
        st.selectbox("Saltar a una operación", list(range(len(steps))), key=scrub_key,
                     format_func=lambda i: f"{i+1}. {steps[i].title}", on_change=scrubbed,
                     label_visibility="collapsed")
        step = steps[index]
        st.progress((index+1) / len(steps), text=f"{step.phase} · {index+1}/{len(steps)}")
        st.caption(f"{qr.method} · {step.phase} · operación {index+1} de {len(steps)}")
        render_operation(step, reduced_motion, animated_plot, st.session_state[play_key])
        if index == len(steps)-1 and y is None:
            insight("Factorización obtenida", "A = QR · Q aporta las direcciones ortonormales; R reconstruye cada columna de A.")
            st.caption(f"Reconstrucción relativa: {qr.reconstruction_error:.3e} · ortogonalidad: {qr.orthogonality_error:.3e}")
        if index == len(steps)-1 and y is not None:
            labels = coefficient_labels or [f"β{i+1}" for i in range(len(step.after))]
            insight("Solución obtenida", " · ".join(f"{label} = {value:.6g}" for label, value in zip(labels, step.after)))
            if model_expression:
                st.latex(model_expression)
            st.write("Estos coeficientes son los que construyen el ajuste del ejercicio. El procedimiento ha terminado en la solución, después de factorizar A y transformar y.")
        with st.expander("Comprobar Q y R" if y is None else "Comprobar Q, R y el sistema triangular"):
            matrix("Q", qr.Q)
            matrix("R", qr.R)
            if y is not None:
                matrix("Qᵀy", qr.Q.T @ y)
            st.caption(f"Error A−QR: {qr.reconstruction_error:.3e} · ortogonalidad: {qr.orthogonality_error:.3e}")
    viewer()


def animated_plot(fig: go.Figure, frames: list[go.Frame], labels: list[str],
                  reduced_motion: bool, alt: str) -> go.Figure:
    """Añade controles Plotly de reproducción y recorrido manual."""
    fig.frames = frames
    steps = [dict(label=label, method="animate",
                  args=[[frame.name], {"frame": {"duration": 0, "redraw": True},
                                       "transition": {"duration": 0}, "mode": "immediate"}])
             for frame, label in zip(frames, labels)]
    fig.update_layout(sliders=[dict(active=0, steps=steps, pad=dict(t=35), currentvalue=dict(prefix="Estado: "))],
                      margin=dict(l=35, r=22, t=45, b=95), meta=dict(alt=alt))
    if not reduced_motion:
        light = st.session_state.get("lab_light_value", False)
        fig.update_layout(updatemenus=[dict(type="buttons", direction="left", x=0, y=-.13,
                                            showactive=False, bgcolor="#dbe7f3" if light else "#253750",
                                            bordercolor="#9bb0c7" if light else "#405775",
                                            font=dict(color="#172338" if light else "#e5edf7", size=12),
                                            buttons=[dict(label="▶ Reproducir", method="animate",
                                                          args=[None, {"frame": {"duration": 650, "redraw": True},
                                                                       "transition": {"duration": 0}, "fromcurrent": True}]),
                                                     dict(label="Pausar", method="animate",
                                                          args=[[None], {"frame": {"duration": 0, "redraw": False},
                                                                         "mode": "immediate"}])])])
    return fig
