import numpy as np
import pytest
from pathlib import Path
from streamlit.testing.v1 import AppTest
from numerical_methods import householder, modified_gram_schmidt
from numerical_methods.teaching import factorization_states, factorization_steps
from components.operations import operation_figure
from components.qr_lab import comparison_figure


@pytest.mark.parametrize("shape", [(3, 2), (3, 3), (8, 3), (1, 1)])
def test_partial_factorizations_reconstruct_original_matrix(shape):
    a = np.random.default_rng(42).normal(size=shape)
    for algorithm in [modified_gram_schmidt, householder]:
        result = algorithm(a)
        states = factorization_states(result)
        assert len(states) == shape[1]+1
        for k, state in enumerate(states):
            reconstructed = state["Q"] @ state["R"]
            if state["pending"] is not None:
                reconstructed += state["pending"]
                np.testing.assert_allclose(state["Q"][:, :k].T @ state["Q"][:, :k], np.eye(k), atol=1e-12)
            else:
                np.testing.assert_allclose(state["Q"].T @ state["Q"], np.eye(shape[0]), atol=1e-12)
                np.testing.assert_allclose(np.tril(state["R"][:, :k], -1), 0, atol=1e-12)
            np.testing.assert_allclose(reconstructed, a, atol=1e-12)
        np.testing.assert_allclose(states[-1]["R"][:shape[1]], result.R, atol=1e-12)


def test_geometry_preserves_angles_and_reflection_mirror():
    a = np.array([[1., 1.], [1., 2.], [1., 4.]])
    for algorithm in [modified_gram_schmidt, householder]:
        for step in factorization_steps(algorithm(a)):
            fig, frames = operation_figure(step)
            start = np.array([fig.data[0].x[-1], fig.data[0].y[-1]])
            end = np.array([frames[-1].data[0].x[-1], frames[-1].data[0].y[-1]])
            assert start @ end == pytest.approx(step.before @ step.after, abs=1e-12)
            if step.kind == "proyectar":
                assert abs(end[0]) < 1e-12
            if step.kind in ("reflector", "reflejar"):
                mirror = next(t for t in fig.data if t.name == "Espejo H")
                direction = np.array([mirror.x[-1], mirror.y[-1]])
                assert (end-start) @ direction == pytest.approx(0, abs=1e-12)
                assert (start+end) @ (start-end) == pytest.approx(0, abs=1e-12)


def test_comparison_animation_finishes_with_factorizations():
    a = np.array([[1., 1.], [1., 2.], [1., 4.]])
    mgs, hh = modified_gram_schmidt(a), householder(a)
    fig, frames = comparison_figure(mgs, hh)
    assert len(fig.data) == 6 and len(frames) == 3
    final = frames[-1].data
    np.testing.assert_allclose(np.array(final[0].z) @ np.array(final[1].z), a, atol=1e-12)
    np.testing.assert_allclose(final[2].z, 0)
    np.testing.assert_allclose(np.array(final[3].z) @ np.array(final[4].z), a, atol=1e-12)


@pytest.mark.parametrize("page,key", [
    ("pages/1_regresion_qr.py", "housing_qr"),
    ("pages/2_estabilidad_numerica.py", "stability_qr"),
    ("pages/3_trayectoria_proyectil.py", "projectile_qr"),
])
def test_qr_comparison_is_inside_each_example(page, key):
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(root / page, default_timeout=30).run()
    next(b for b in app.button if b.label == "Resolver con QR →").click().run()
    assert not app.exception
    next(b for b in app.button if b.label == "Paso →").click().run()
    assert app.session_state[f"{key}_frame"] == 1
    next(s for s in app.selectbox if s.label == "Método QR para observar").set_value("Gram-Schmidt modificado").run()
    assert not app.exception
    assert app.session_state[f"{key}_frame"] == 0
    next(r for r in app.radio if r.label == "Resolución QR de este ejemplo").set_value("Comparar métodos").run()
    assert not app.exception
    assert len(app.get("plotly_chart")) == 1
    assert len(app.metric) == 6
    assert any("datos del ejercicio" in item.value for item in app.markdown)
    next(t for t in app.toggle if t.label == "Reducir movimiento").set_value(True).run()
    assert not app.exception
    next(r for r in app.radio if r.label == "Resolución QR de este ejemplo").set_value("Paso a paso").run()
    assert not app.exception
    assert "▶ Reproducir" not in [b.label for b in app.button]
    jump = next(s for s in app.selectbox if s.label == "Saltar a una operación")
    jump.select_index(len(jump.options)-1).run()
    assert not app.exception
    assert any("Solución obtenida" in item.value for item in app.markdown)


@pytest.mark.parametrize("shape", [(16, 2), (40, 3), (60, 3)])
def test_long_comparison_previews_real_data_and_keeps_all_r_rows(shape):
    a = np.random.default_rng(8).normal(size=shape)
    mgs, hh = modified_gram_schmidt(a), householder(a)
    _, frames = comparison_figure(mgs, hh)
    np.testing.assert_allclose(frames[0].data[2].z, a[:6])
    np.testing.assert_allclose(frames[-1].data[0].z, mgs.Q[:6])
    np.testing.assert_allclose(frames[-1].data[1].z, mgs.R)
    np.testing.assert_allclose(frames[-1].data[3].z, hh.Q[:6], atol=1e-12)
    np.testing.assert_allclose(frames[-1].data[4].z[:shape[1]], hh.R)
    np.testing.assert_allclose(frames[-1].data[5].z, a[:6], atol=1e-12)


@pytest.mark.parametrize("system", ["Altura y(t)", "Posición horizontal x(t)"])
@pytest.mark.parametrize("known_gravity", [False, True])
def test_projectile_comparison_uses_selected_system_and_gravity(system, known_gravity):
    from experiments.projectile import simulated_projectile
    from numerical_methods import least_squares
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(root / "pages/3_trayectoria_proyectil.py", default_timeout=30).run()
    if known_gravity:
        for _ in range(2):
            next(b for b in app.button if b.label == "Siguiente →").click().run()
        next(r for r in app.radio if r.label == "Tratamiento de g").set_value("Usar g conocida").run()
    next(b for b in app.button if b.label == "Resolver con QR →").click().run()
    next(r for r in app.radio if r.label == "Ecuación de movimiento a resolver").set_value(system).run()
    next(r for r in app.radio if r.label == "Resolución QR de este ejemplo").set_value("Comparar métodos").run()
    assert not app.exception
    data = simulated_projectile(35., 42., 3., 9.81, 14, .8, 13)
    t = data.t_s.to_numpy()
    a = np.column_stack([np.ones_like(t), t])
    target = data.x_m.to_numpy() if system == "Posición horizontal x(t)" else data.y_m.to_numpy()
    if system == "Altura y(t)":
        if known_gravity:
            target = target + .5*9.81*t**2
        else:
            a = np.column_stack([a, t**2])
    for index, method in enumerate(["Gram-Schmidt modificado", "Householder"]):
        result = least_squares(a, target, method)
        assert app.metric[3*index].value == f"{result.qr.reconstruction_error:.3e}"
        assert app.metric[3*index+1].value == f"{result.qr.orthogonality_error:.3e}"
        assert app.metric[3*index+2].value == f"{np.sqrt(np.mean(result.residuals**2)):.6g}"


def test_home_links_to_three_original_examples():
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(root / "app.py", default_timeout=30).run()
    assert not app.exception
    assert len(app.get("page_link")) == 3
    assert not (root / "pages/0_factorizacion_qr.py").exists()
