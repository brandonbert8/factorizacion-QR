"""Comparación animada de los estados reales de ambos algoritmos QR."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from numerical_methods.teaching import factorization_states
from numerical_methods.diagnostics import aligned_factors, perturbation_study, reference_comparison
from .qr_figures import reconstruction_figure, orthogonality_figure, perturbation_figure
from .ui import matrix, plot_style
from .experience import animated_plot, fingerprint, insight, qr_animation


def comparison_figure(mgs, hh, *, max_rows=6):
    """Una iteración compartida; no representa tiempo ni coste de ejecución."""
    left, right = factorization_states(mgs), factorization_states(hh)
    n = mgs.shape[1]
    m = mgs.shape[0]
    q_title = "Q completa acumulada" if m <= max_rows else "primeras n columnas de Qₖ"
    titles = ("MGS · Q parcial", "MGS · R parcial", "MGS · residuos pendientes V",
              f"Householder · {q_title}", "Householder · matriz B = Hₖ…H₁A", "Householder · Q completa × B")
    fig = make_subplots(rows=2, cols=3, subplot_titles=titles, horizontal_spacing=.09, vertical_spacing=.22)
    frames = []
    a = mgs.frames[0].matrix
    scales = [1., max(np.abs(mgs.R).max(), np.abs(a).max()), np.abs(a).max(),
              1., max(np.abs(hh.R).max(), np.abs(a).max()), np.abs(a).max()]
    for iteration, (ms, hs) in enumerate(zip(left, right)):
        arrays = [ms["Q"], ms["R"], ms["pending"], hs["Q"], hs["R"], hs["Q"] @ hs["R"]]
        traces = []
        for index, (array, scale) in enumerate(zip(arrays, scales)):
            # Mostrar una ventana legible; todos los estados se calculan con A completa.
            if index == 3 and m > max_rows:
                array = array[:, :n]
            array = array[:max_rows]
            trace = go.Heatmap(z=array, text=[[f"{x:.5g}" for x in row] for row in array], texttemplate="%{text}",
                               colorscale="RdBu", zmid=0, zmin=-scale, zmax=scale, showscale=False,
                               x=[f"c{j+1}" for j in range(array.shape[1])],
                               y=[f"f{i+1}" for i in range(array.shape[0])],
                               xgap=3, ygap=3, hovertemplate="%{y}, %{x}: %{z:.10g}<extra></extra>")
            traces.append(trace)
            if iteration == 0:
                fig.add_trace(trace, row=index//3+1, col=index%3+1)
        title = ("Inicio · misma matriz A para ambos métodos" if iteration == 0 else
                 f"Iteración {iteration}/{n} · MGS fija q{iteration} y resta proyecciones; Householder aplica H{iteration}")
        if iteration == n:
            title = "QR completa · MGS: A = QR; Householder: B = [R; 0]"
        frames.append(go.Frame(name=str(iteration), data=traces, traces=list(range(6)),
                               layout=go.Layout(title=dict(text=title, font=dict(size=14)))))
    plot_style(fig, "", "")
    fig.update_layout(height=650, title=dict(text=frames[0].layout.title.text, font=dict(size=14)))
    fig.update_annotations(font_size=12, yshift=10)
    fig.update_xaxes(side="bottom", showgrid=False, zeroline=False)
    fig.update_yaxes(autorange="reversed", showgrid=False, zeroline=False)
    return fig, frames


def qr_resolution(results, method, key, signature, reduced, *, y, context,
                  coefficient_labels=None, model_expression=None):
    """Factorizar y comparar dentro del ejercicio, usando su misma A y su mismo y."""
    selected = results[method]
    a = selected.qr.frames[0].matrix
    st.caption(context)
    st.latex(r"A_{m\times n}=Q_{m\times n}R_{n\times n},\qquad Q^TQ=I_n")
    st.caption(f"Matriz del ejercicio: {a.shape[0]} observaciones × {a.shape[1]} columnas · κ₂(A) = {np.linalg.cond(a):.3e}")
    for warning in selected.qr.warnings:
        st.warning(warning)
    view = st.radio("Resolución QR de este ejemplo", ["Paso a paso", "Comparar métodos", "Reconstruir A", "Precisión y sensibilidad"],
                    horizontal=True, key=f"{key}_view")
    if view == "Reconstruir A":
        render_reconstruction(results, key, reduced)
        return
    if view == "Precisión y sensibilidad":
        render_diagnostics(results, y, key, reduced)
        return
    if view == "Paso a paso":
        if method == "Gram-Schmidt modificado":
            insight("Cómo se factoriza esta matriz", "Normalizar → proyectar → restar → repetir con los residuos actualizados.")
            st.latex(r"r_{kk}=\|v_k\|_2,\quad q_k=v_k/r_{kk},\quad r_{kj}=q_k^Tv_j,\quad v_j\leftarrow v_j-r_{kj}q_k")
            st.caption("Gram-Schmidt modificado de una pasada: cada proyección usa el residuo actualizado.")
        else:
            insight("Cómo se factoriza esta matriz", "Construir un espejo → reflejar el bloque pendiente → anular las entradas bajo la diagonal.")
            st.latex(r"\alpha=-\operatorname{sign}(x_1)\|x\|_2,\quad v=\frac{x-\alpha e_1}{\|x-\alpha e_1\|_2},\quad H=I-2vv^T")
            st.caption("Se toma sign(0) = 1. El signo de α evita cancelación al construir v. Cada reflector actúa en un bloque más pequeño.")
        qr_animation(selected.qr, key, signature, reduced, y=y,
                     coefficient_labels=coefficient_labels, model_expression=model_expression)
        return

    mgs, hh = results["Gram-Schmidt modificado"], results["Householder"]
    insight("Comparación sobre los datos del ejercicio", "MGS construye Q con proyecciones; Householder construye R con reflectores. Ambos parten de la misma A y resuelven el mismo y.")
    fig, frames = comparison_figure(mgs.qr, hh.qr)
    animated_plot(fig, frames, ["Inicio"] + [f"Iteración {k}" for k in range(1, a.shape[1]+1)], reduced,
                  "Construcción de Q y R por ambos métodos sobre la matriz de este ejercicio")
    fig.update_layout(margin=dict(t=110, b=105, l=30, r=25))
    if reduced:
        for trace, data in zip(fig.data, frames[-1].data):
            trace.update(data.to_plotly_json())
        fig.layout.title = frames[-1].layout.title
        fig.layout.sliders[0].active = len(frames)-1
    else:
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1800
        fig.layout.updatemenus[0].buttons[0].args[1]["fromcurrent"] = False
    st.plotly_chart(fig, width="stretch", theme=None, key=f"{key}_comparison_{fingerprint(a)}",
                    alt="Comparación animada de Gram-Schmidt modificado y Householder con la matriz del ejercicio")
    st.caption("Cada avance termina una iteración de ambos métodos; no mide tiempo de ejecución. Las matrices muestran estados calculados.")
    if a.shape[0] > 6:
        st.caption(f"Vista ampliada: primeras 6 de {a.shape[0]} filas. Para Q de Householder se muestran las primeras n columnas de la matriz completa acumulada. Los cálculos y métricas usan todas las filas; los factores finales completos están debajo.")
    left, right = st.columns(2)
    with left:
        st.markdown("**Gram-Schmidt modificado**")
        st.latex(r"A=Q_kR_k+V_{\mathrm{pendiente}}")
        st.write("Q incorpora una dirección por iteración. R guarda normas y proyecciones. Al terminar, los residuos pendientes son cero.")
    with right:
        st.markdown("**Householder**")
        st.latex(r"B_k=H_k\cdots H_1A,\quad Q_k=H_1\cdots H_k,\quad A=Q_kB_k")
        st.write("Cada reflector crea ceros bajo la diagonal. Al terminar B = [R; 0], y las primeras n columnas de Q completa dan la QR económica.")
    n = a.shape[1]
    st.caption(f"Para esta A: MGS realiza {n} normalizaciones y {n*(n-1)//2} proyecciones; Householder aplica {n} reflectores. Son operaciones diferentes; estos recuentos no miden velocidad. Ambos tienen coste O(mn²) para m ≥ n.")
    st.subheader("Comparación final con los datos del ejercicio")
    labels = coefficient_labels or [f"β{i+1}" for i in range(a.shape[1])]
    for column, result in zip(st.columns(2), [mgs, hh]):
        with column:
            st.markdown(f"**{result.qr.method}**")
            st.metric("Error QR · ‖A − QR‖F / ‖A‖F", f"{result.qr.reconstruction_error:.3e}")
            st.metric("Ortogonalidad · ‖QᵀQ − I‖F", f"{result.qr.orthogonality_error:.3e}")
            st.metric("RMSE del sistema ajustado", f"{np.sqrt(np.mean(result.residuals**2)):.6g}")
            st.write(" · ".join(f"{label} = {value:.6g}" for label, value in zip(labels, result.coefficients)))
            with st.expander("Comprobar factores completos de este método"):
                matrix("Q económica", result.qr.Q)
                matrix("R", result.qr.R)
                matrix("QᵀQ", result.qr.Q.T @ result.qr.Q)
                matrix("A − QR", a - result.qr.Q @ result.qr.R)
    st.caption("Menor error QR y menor pérdida de ortogonalidad indican una factorización más precisa. El RMSE mide el ajuste al sistema de este ejercicio.")
    st.caption("Los signos de Q y R pueden diferir sin cambiar QR. Householder suele conservar mejor la ortogonalidad en matrices con columnas casi dependientes; ninguno elimina el mal condicionamiento de A.")


def render_motion(fig, frames, labels, reduced, key, alt, *, duration=1100, top=110):
    """Controles coherentes y resultado visible al reducir movimiento."""
    animated_plot(fig, frames, labels, reduced, alt)
    fig.update_layout(margin=dict(t=top, b=105, l=55, r=30))
    if reduced:
        for index, data in zip(frames[-1].traces, frames[-1].data):
            fig.data[index].update(data.to_plotly_json())
        fig.update_layout(title=frames[-1].layout.title)
        fig.layout.sliders[0].active = len(frames)-1
    else:
        options = fig.layout.updatemenus[0].buttons[0].args[1]
        options["frame"]["duration"] = duration
        options["fromcurrent"] = False
    st.plotly_chart(fig, width="stretch", theme=None, key=key, alt=alt)


def render_reconstruction(results, key, reduced):
    mgs, hh = results["Gram-Schmidt modificado"].qr, results["Householder"].qr
    n = mgs.shape[1]
    st.subheader("Cómo Q y R recuperan una columna de A")
    column = st.selectbox("Columna que se reconstruye", range(n), index=n-1,
                          format_func=lambda j: f"a{j+1} · {j+1} términos", key=f"{key}_column_{n}")
    st.latex(rf"a_{{{column+1}}}=\sum_{{i=1}}^{{{column+1}}}q_i r_{{i,{column+1}}}")
    fig, frames = reconstruction_figure(mgs, hh, column)
    signature = fingerprint(mgs.frames[0].matrix)
    render_motion(fig, frames, ["Inicio"]+[f"+ q{i}r{i},{column+1}" for i in range(1, column+2)],
                  reduced, f"{key}_reconstruction_{column}_{signature}",
                  "Suma de los términos de Q y R para reconstruir una columna de la matriz del ejercicio")
    st.caption("La barra muestra el término que se añade y la línea rosa acumula la suma. Se usan todas las filas. R es triangular: la columna aⱼ necesita solamente q₁,…,qⱼ.")
    a = mgs.frames[0].matrix
    for col, qr in zip(st.columns(2), [mgs, hh]):
        with col:
            st.markdown(f"**{qr.method}**")
            difference = np.linalg.norm(a[:, column] - qr.Q @ qr.R[:, column])
            st.metric("Error absoluto en la columna · ‖aⱼ − Qrⱼ‖₂", f"{difference:.3e}")
    st.caption("Si una columna de Q cambia de signo, la fila correspondiente de R también cambia. El término qᵢrᵢⱼ conserva su valor: comparar factores requiere considerar esa equivalencia.")


def render_diagnostics(results, y, key, reduced):
    mgs, hh = results["Gram-Schmidt modificado"].qr, results["Householder"].qr
    a = mgs.frames[0].matrix
    signature = fingerprint(a, y)
    st.subheader("Ortogonalidad durante la construcción de Q")
    fig, frames = orthogonality_figure(mgs, hh)
    render_motion(fig, frames, ["Inicio"] + [f"{k} columnas" for k in range(1, a.shape[1]+1)],
                  reduced, f"{key}_orthogonality_{signature}",
                  "Error Q transpuesta Q menos identidad de las columnas terminadas en cada iteración")
    st.caption("Los espacios vacíos corresponden a columnas pendientes. Cero es el valor ideal: la diagonal comprueba norma unitaria y las otras celdas comprueban perpendicularidad. Ambos paneles usan la misma escala.")

    st.subheader("Referencia independiente sobre el mismo sistema")
    rows = reference_comparison(a, y, results)
    st.dataframe(pd.DataFrame(rows).style.format({name: "{:.3e}" for name in rows[0] if name != "Método"},
                                                na_rep="—"), hide_index=True, width="stretch")
    st.caption("La referencia usa QR económica de SciPy para los factores y QR con pivoteo para la solución. También calcula en coma flotante; una diferencia pequeña valida concordancia, no exactitud absoluta.")

    with st.expander("Equivalencia de los factores y sus signos"):
        qm, rm = aligned_factors(mgs)
        qh, rh = aligned_factors(hh)
        st.latex(r"A=(QD)(DR),\qquad D=\operatorname{diag}(\pm1),\quad D^2=I")
        st.caption("Se ajustan los signos para que ambas diagonales de R sean positivas. Con rango completo, esta convención permite comparar los factores económicos directamente.")
        left, right = st.columns(2)
        left.metric("Diferencia Q · ‖Qₘ − Qₕ‖F", f"{np.linalg.norm(qm-qh, 'fro'):.3e}")
        right.metric("Diferencia R relativa · ‖Rₘ − Rₕ‖F / ‖Rₕ‖F", f"{np.linalg.norm(rm-rh, 'fro')/np.linalg.norm(rh, 'fro'):.3e}")
        matrix("R de Gram-Schmidt · diagonal positiva", rm)
        matrix("R de Householder · diagonal positiva", rh)

    with st.expander("Respuesta de los coeficientes a perturbaciones en y"):
        st.caption("Se mantiene A y se perturba y en la dirección singular de mayor sensibilidad absoluta. Cada método se compara con su propia solución base; el experimento es controlado, no ruido aleatorio.")
        bounds = st.slider("Intervalo de log₁₀ δ", -14, -1, (-12, -3), key=f"{key}_noise_range")
        if np.linalg.norm(y) == 0:
            st.caption("Con y = 0 no está definida una perturbación relativa. La comparación de QR permanece válida.")
        else:
            study = perturbation_study(a, y, np.logspace(*bounds, 13))
            fig, frames = perturbation_figure(study)
            render_motion(fig, frames, [f"δ={value:.1e}" for value in study["levels"]], reduced,
                          f"{key}_sensitivity_{signature}_{bounds}",
                          "Cambio relativo de los coeficientes de ambos métodos y SciPy al perturbar las observaciones",
                          duration=450)
            st.latex(r"\delta=\frac{\|\Delta y\|_2}{\|y\|_2},\qquad \text{cambio de }\beta=\frac{\|\beta(y+\Delta y)-\beta(y)\|_2}{\|\beta(y)\|_2}")
            st.caption("Curvas parecidas pueden compartir una gran sensibilidad: eso pertenece al sistema. Diferencias entre curvas también incluyen redondeo. Para perturbaciones diminutas, el redondeo puede dominar el cambio medido.")
            if any(np.isnan(values).any() for values in study["changes"].values()):
                st.caption("El cambio relativo de β se omite cuando su solución base es cero.")
            st.caption("Los cambios exactamente cero se sitúan en 10⁻¹⁸ para mostrarlos en escala logarítmica; el valor calculado se conserva al pasar el cursor.")
