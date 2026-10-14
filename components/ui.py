"""Presentación común, visualización de pasos y exportación."""

from html import escape
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from numerical_methods.qr import QRResult

METHODS = ["Householder", "Gram-Schmidt modificado"]


class _Colors:
    palettes = {
        False: {"data": "#7dd3fc", "fit": "#fbbf24", "ideal": "#a78bfa", "error": "#fb7185"},
        True: {"data": "#0369a1", "fit": "#b45309", "ideal": "#6d28d9", "error": "#be123c"},
    }

    def __getitem__(self, name: str) -> str:
        return self.palettes[st.session_state.get("lab_light_value", False)][name]


COLORS = _Colors()


def theme():
    st.html("""<style>
    h1,h2,h3 {letter-spacing: -.035em}
    [data-testid="stMainBlockContainer"] {padding-top: 3.5rem}
    h1 {font-size: 2.5rem !important}
    .lead {opacity: .78; font-size: 1.12rem; max-width: 750px}
    .eyebrow {color: #7dd3fc; letter-spacing: .16em; text-transform: uppercase; font-size: .72rem; font-weight: 700}
    [data-testid="stMetric"] {background: color-mix(in srgb, currentColor 7%, transparent); border: 1px solid rgba(125,160,190,.25); border-radius: 9px; padding: 12px}
    [data-testid="stPlotlyChart"] {border: 1px solid #253750; border-radius: 9px; overflow: hidden}
    .scene-title {text-align: center; font-size: 1.22rem; font-weight: 700; padding: .4rem 0}
    .insight {border-left: 4px solid #38bdf8; background: color-mix(in srgb, currentColor 7%, transparent); padding: 1rem 1.2rem; margin: 1rem 0 1.5rem; display: grid; gap: .35rem; border-radius: 0 9px 9px 0}
    .insight span {opacity: .75; font-size: .88rem}
    .insight strong {font-size: 1.1rem; font-weight: 650}
    @media (max-width: 800px) {h1 {font-size: 2rem !important} .lead {font-size: 1rem} [data-testid="stMainBlockContainer"] {padding-top: 3.5rem}}
    </style>""")
    if st.session_state.get("lab_light_value", False):
        st.html("""<style>
        [data-testid="stApp"] {background: #f7f9fc !important; color: #172338 !important}
        [data-testid="stHeader"] {background: #f7f9fc !important}
        [data-testid="stSidebar"] {background: #edf2f8 !important}
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] a {color: #172338 !important}
        [data-testid="stHeader"] button,
        [data-testid="stHeader"] button span,
        [data-testid="stHeader"] button svg,
        [data-testid="stSidebar"] button,
        [data-testid="stSidebar"] button span,
        [data-testid="stSidebar"] button svg {color: #172338 !important; fill: #172338 !important}
        [data-testid="stMainBlockContainer"] h1,
        [data-testid="stMainBlockContainer"] h2,
        [data-testid="stMainBlockContainer"] h3,
        [data-testid="stMainBlockContainer"] p,
        [data-testid="stMainBlockContainer"] .scene-title {color: #172338}
        [data-testid="stApp"] button {background-color: #e5edf7 !important; color: #172338 !important; border-color: #b5c8da !important}
        [data-testid="stApp"] button p {color: #172338 !important}
        [data-testid="stApp"] button[data-selected="true"] {background-color: #cfe7fa !important}
        [data-testid="stApp"] input {background-color: #ffffff !important; color: #172338 !important}
        [data-testid="stPopoverBody"] {background: #edf2f8 !important; color: #172338 !important}
        [data-testid="stPopoverBody"] p {color: #172338 !important}
        [data-testid="stPopoverBody"] button,
        [data-testid="stPopoverBody"] button svg {color: #172338 !important; fill: #172338 !important}
        .eyebrow {color: #075985 !important}
        .insight {background: #eaf1f8 !important; border-left-color: #0369a1}
        [data-testid="stMetric"] {background: #eaf1f8 !important; color: #172338}
        [data-testid="stProgressBarTrack"] {background: #d5e2ef !important}
        [data-testid="stProgressBarTrack"] > div {background: #0284c7 !important}
        [data-testid="stFileUploaderDropzone"] {background: #eaf1f8 !important; border-color: #b5c8da !important}
        [data-testid="stFileUploaderDropzone"] * {color: #172338 !important}
        [data-testid="stPlotlyChart"] {border-color: #c5d3e3}
        </style>""")


def heading(kicker: str, title: str, description: str):
    theme()
    st.markdown(f'<div class="eyebrow">{escape(kicker)}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="lead">{escape(description)}</p>', unsafe_allow_html=True)


def plot_style(fig: go.Figure, x: str, y: str) -> go.Figure:
    light = st.session_state.get("lab_light_value", False)
    fig.update_layout(template="plotly_white" if light else "plotly_dark",
                      paper_bgcolor="#ffffff" if light else "#0f1929",
                      plot_bgcolor="#ffffff" if light else "#0f1929",
                      font=dict(color="#172338" if light else "#cfdded"),
                      margin=dict(l=35, r=22, t=35, b=35),
                      xaxis_title=x, yaxis_title=y, legend=dict(orientation="h", y=1.12))
    fig.update_xaxes(gridcolor="#dae3ee" if light else "#253750", zerolinecolor="#a9b9cd" if light else "#405775")
    fig.update_yaxes(gridcolor="#dae3ee" if light else "#253750", zerolinecolor="#a9b9cd" if light else "#405775")
    return fig


def matrix(label: str, values: np.ndarray):
    st.caption(f"{label} · dimensión {np.shape(values)}")
    st.dataframe(pd.DataFrame(np.asarray(values)).style.format("{:.6g}"), hide_index=True, width="stretch")


def qr_inspector(qr: QRResult, key: str):
    for message in qr.warnings:
        st.warning(message)
    with st.expander("Matrices y verificación QR"):
        left, right = st.columns(2)
        with left:
            matrix("Q", qr.Q)
        with right:
            matrix("R", qr.R)
        st.caption(f"A ≈ QR · error relativo {qr.reconstruction_error:.3e} · "
                   f"ortogonalidad {qr.orthogonality_error:.3e}")
    with st.expander("Procedimiento paso a paso"):
        state_key = f"step_{key}"
        st.session_state[state_key] = min(st.session_state.get(state_key, 0), len(qr.steps) - 1)
        back, next_, reset = st.columns(3)
        if back.button("← Anterior", key=f"back_{key}"):
            st.session_state[state_key] = max(0, st.session_state[state_key] - 1)
        if next_.button("Siguiente →", key=f"next_{key}"):
            st.session_state[state_key] = min(len(qr.steps) - 1, st.session_state[state_key] + 1)
        if reset.button("Reiniciar", key=f"reset_{key}"):
            st.session_state[state_key] = 0
        step = qr.steps[st.session_state[state_key]]
        st.subheader(f"Iteración {step.iteration} de {len(qr.steps)}")
        st.write(step.operation)
        st.latex(step.formula)
        st.write(step.interpretation)
        for name, value in step.data.items():
            if isinstance(value, np.ndarray):
                matrix(name, value.reshape(1, -1) if value.ndim == 1 else value)
            else:
                st.caption(f"{name}: {value:.6g}")


def downloadable_report(title: str, description: str, parameters: dict,
                        matrices: dict[str, np.ndarray], metrics: dict,
                        interpretation: str, figures: list[go.Figure],
                        csv_data: pd.DataFrame, key: str):
    st.subheader("Exportar experimento")
    st.download_button("Descargar resultados CSV", csv_data.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{key}.csv", mime="text/csv", key=f"csv_{key}")
    parts = ["<!doctype html><html lang='es'><meta charset='utf-8'>",
             "<title>" + escape(title) + "</title>",
             "<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;color:#152234}"
             "table{border-collapse:collapse;margin:18px 0}td,th{border:1px solid #cbd5e1;padding:6px 10px}"
             "h1,h2{color:#143b60}.plot{margin:25px 0}</style>",
             f"<h1>{escape(title)}</h1><p>{escape(description)}</p>"]
    for section, mapping in (("Parámetros", parameters), ("Métricas", metrics)):
        parts.append(f"<h2>{section}</h2><table>")
        parts.extend(f"<tr><th>{escape(str(k))}</th><td>{escape(str(v))}</td></tr>" for k, v in mapping.items())
        parts.append("</table>")
    parts.append("<h2>Matrices y coeficientes</h2>")
    for name, data in matrices.items():
        frame = pd.DataFrame(np.atleast_2d(data))
        parts.append(f"<h3>{escape(name)}</h3>{frame.to_html(index=False, header=False, float_format=lambda v: f'{v:.6g}')}" )
    parts.append(f"<h2>Interpretación</h2><p>{escape(interpretation)}</p>")
    for figure in figures:
        parts.append("<div class='plot'>" + figure.to_html(full_html=False, include_plotlyjs="cdn") + "</div>")
    parts.append("</html>")
    st.download_button("Descargar reporte HTML", "".join(parts).encode("utf-8"),
                       file_name=f"{key}_reporte.html", mime="text/html", key=f"html_{key}")
