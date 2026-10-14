"""Animaciones de reconstrucción, ortogonalidad y sensibilidad de QR."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from numerical_methods.diagnostics import orthogonality_history
from .ui import COLORS, plot_style


def reconstruction_figure(mgs, hh, column):
    """Acumula términos reales qᵢrᵢⱼ, con todas las componentes de aⱼ."""
    if column not in range(mgs.shape[1]):
        raise ValueError("Columna fuera de la matriz.")
    a = mgs.frames[0].matrix
    x = np.arange(1, a.shape[0]+1)
    terms = [qr.Q[:, :column+1] * qr.R[:column+1, column] for qr in [mgs, hh]]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Gram-Schmidt modificado", "Householder"),
                        horizontal_spacing=.12, shared_yaxes=True)
    frames = []
    for count in range(column+2):
        traces = []
        for method_index, contributions in enumerate(terms):
            term = contributions[:, count-1] if count else np.zeros(len(x))
            cumulative = contributions[:, :count].sum(axis=1)
            current = f"Término q{count} · r{count},{column+1}" if count else "Término por añadir"
            data = [go.Scatter(x=x, y=a[:, column], mode="lines+markers", name=f"Columna a{column+1} original",
                               line=dict(color=COLORS["data"], dash="dash"), marker=dict(size=5)),
                    go.Bar(x=x, y=term, name=current, marker_color=COLORS["fit"], opacity=.65),
                    go.Scatter(x=x, y=cumulative, mode="lines+markers", name="Suma reconstruida",
                               line=dict(color=COLORS["error"], width=3), marker=dict(size=6))]
            for trace in data:
                trace.showlegend = method_index == 0
                trace.hovertemplate = "Fila %{x}: %{y:.10g}<extra>%{fullData.name}</extra>"
                traces.append(trace)
                if count == 0:
                    fig.add_trace(trace, row=1, col=method_index+1)
        title = f"Columna a{column+1} · {count} de {column+1} términos de Q y R"
        frames.append(go.Frame(name=str(count), data=traces, traces=list(range(6)),
                               layout=go.Layout(title=dict(text=title, font=dict(size=14)))))
    extrema = np.concatenate([a[:, column]] + [t.ravel() for t in terms]
                             + [np.cumsum(t, axis=1).ravel() for t in terms])
    low, high = float(extrema.min()), float(extrema.max())
    padding = max(high-low, abs(high), abs(low), 1e-12)*.12
    plot_style(fig, "Fila de A", "Componente de la columna")
    fig.update_xaxes(title="Fila de A")
    fig.update_yaxes(range=[min(0, low)-padding, max(0, high)+padding])
    fig.update_layout(height=500, title=frames[0].layout.title, legend=dict(orientation="h", y=1.24))
    return fig, frames


def orthogonality_figure(mgs, hh):
    """Revela QₖᵀQₖ − I; las columnas todavía no construidas quedan vacías."""
    histories = [orthogonality_history(qr) for qr in [mgs, hh]]
    n = mgs.shape[1]
    scale = max(1e-16, max(np.max(np.abs(h[-1]["error"])) for h in histories))
    fig = make_subplots(rows=1, cols=2, subplot_titles=("MGS · QₖᵀQₖ − I", "Householder · QₖᵀQₖ − I"),
                        horizontal_spacing=.15)
    frames = []
    for k in range(n+1):
        data = []
        for index, history in enumerate(histories):
            matrix = np.full((n, n), np.nan)
            matrix[:k, :k] = history[k]["error"]
            text = [[f"{v:.1e}" if np.isfinite(v) else "" for v in row] for row in matrix]
            trace = go.Heatmap(z=matrix, text=text, texttemplate="%{text}", zmin=-scale, zmax=scale,
                               zmid=0, colorscale="RdBu", showscale=False, hoverongaps=False,
                               x=[f"q{i+1}" for i in range(n)], y=[f"q{i+1}" for i in range(n)], xgap=4, ygap=4,
                               hovertemplate="%{y} · %{x}: %{z:.12g}<extra></extra>")
            data.append(trace)
            if k == 0:
                fig.add_trace(trace, row=1, col=index+1)
        title = ("Inicio · todavía no hay columnas terminadas" if k == 0 else
                 f"{k} columnas terminadas · ‖QₖᵀQₖ − I‖F: MGS {histories[0][k]['norm']:.2e} · HH {histories[1][k]['norm']:.2e}")
        frames.append(go.Frame(name=str(k), data=data, traces=[0, 1], layout=go.Layout(title=dict(text=title, font=dict(size=13)))))
    plot_style(fig, "", "")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False, autorange="reversed")
    fig.update_layout(height=440, title=frames[0].layout.title)
    return fig, frames


def perturbation_figure(study):
    """Revela soluciones calculadas al aumentar la perturbación relativa de y."""
    fig = go.Figure()
    names = list(study["changes"])
    colors = [COLORS["fit"], COLORS["data"], COLORS["ideal"]]
    frames = []
    for count in range(1, len(study["levels"])+1):
        data = []
        for name, color in zip(names, colors):
            # El eje logarítmico no representa ceros: se conservan en el hover.
            values = study["changes"][name][:count]
            trace = go.Scatter(x=study["levels"][:count], y=np.maximum(values, 1e-18), customdata=values,
                               mode="lines+markers", name=name,
                               line=dict(color=color, dash="dash" if name.startswith("SciPy") else "solid"),
                               marker=dict(size=7, symbol="diamond" if name.startswith("SciPy") else "circle"),
                               hovertemplate="δ = %{x:.2e}<br>Cambio relativo β = %{customdata:.6e}<extra>%{fullData.name}</extra>")
            data.append(trace)
            if count == 1:
                fig.add_trace(trace)
        frames.append(go.Frame(name=str(count), data=data, traces=list(range(len(names))),
                               layout=go.Layout(title=dict(text=f"Perturbación relativa δ = {study['levels'][count-1]:.2e}", font=dict(size=14)))))
    plot_style(fig, "Perturbación relativa de y · δ", "Cambio relativo de los coeficientes")
    finite = np.concatenate(list(study["changes"].values()))
    finite = finite[np.isfinite(finite)]
    low = max(float(finite.min()), 1e-18) if len(finite) else 1e-18
    high = max(float(finite.max()), low*10) if len(finite) else 1e-17
    fig.update_xaxes(type="log", tickformat=".0e", range=[np.log10(study["levels"].min())-.2, np.log10(study["levels"].max())+.2])
    fig.update_yaxes(type="log", tickformat=".0e", range=[np.log10(low)-.4, np.log10(high)+.4])
    fig.update_layout(height=470, title=frames[0].layout.title, legend=dict(orientation="h", y=1.18))
    return fig, frames
