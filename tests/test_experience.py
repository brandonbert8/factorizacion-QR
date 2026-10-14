import numpy as np
import pytest
from pathlib import Path
from streamlit.testing.v1 import AppTest
from numerical_methods import householder, modified_gram_schmidt

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sample_matrix():
    return np.array([[1., 2.], [2., 1.], [3., 4.], [4., 3.]])


@pytest.mark.parametrize("algorithm", [modified_gram_schmidt, householder])
def test_frames_begin_with_input_and_finish_with_computed_r(algorithm, sample_matrix):
    result = algorithm(sample_matrix)
    np.testing.assert_allclose(result.frames[0].matrix, sample_matrix)
    np.testing.assert_allclose(result.frames[-1].matrix, result.R)
    assert result.frames[0].phase == "inicio"
    assert result.frames[-1].phase == "final"


def test_mgs_projection_frames_match_actual_updates(sample_matrix):
    result = modified_gram_schmidt(sample_matrix)
    for previous, current in zip(result.frames, result.frames[1:]):
        if current.phase == "proyectar":
            _, column = current.active_columns
            expected = previous.matrix.copy()
            expected[:, column] -= current.coefficient * current.vector
            np.testing.assert_allclose(current.matrix, expected, atol=1e-14)
        elif current.phase == "normalizar":
            np.testing.assert_allclose(current.matrix, previous.matrix)
            assert np.linalg.norm(current.vector) == pytest.approx(1.)


def test_householder_frames_match_reflections(sample_matrix):
    result = householder(sample_matrix)
    for previous, current in zip(result.frames, result.frames[1:]):
        if current.phase != "reflejar":
            continue
        k = current.iteration - 1
        expected = previous.matrix.copy()
        v = current.vector
        assert np.linalg.norm(v) == pytest.approx(1.)
        expected[k:, k:] -= 2 * np.outer(v, v @ expected[k:, k:])
        expected[k+1:, k] = 0
        np.testing.assert_allclose(current.matrix, expected, atol=1e-13)


@pytest.mark.parametrize("page", ["pages/1_regresion_qr.py", "pages/2_estabilidad_numerica.py",
                                  "pages/3_trayectoria_proyectil.py"])
def test_presentation_scenes_load_without_mode_selector(page):
    app = AppTest.from_file(ROOT / page, default_timeout=30).run()
    assert not app.exception
    scene_key = {"pages/1_regresion_qr.py": "housing_scene",
                 "pages/2_estabilidad_numerica.py": "stability_scene",
                 "pages/3_trayectoria_proyectil.py": "projectile_scene"}[page]
    for stage in range(1, 5):
        next_button = next(button for button in app.button if button.label == "Siguiente →")
        next_button.click().run()
        assert not app.exception, f"{page}, escena {stage}"
        assert app.session_state[scene_key] == stage
    assert not app.get("segmented_control")


def test_qr_step_controls_and_method_reset():
    app = AppTest.from_file(ROOT / "pages/1_regresion_qr.py", default_timeout=30).run()
    for _ in range(3):
        next(button for button in app.button if button.label == "Siguiente →").click().run()
    next(button for button in app.button if button.label == "Paso →").click().run()
    assert app.session_state["housing_qr_frame"] == 1
    app.selectbox[0].set_value("Gram-Schmidt modificado").run()
    assert not app.exception
    assert app.session_state["housing_qr_frame"] == 0


def test_resolution_reaches_coefficients_and_model_expression():
    app = AppTest.from_file(ROOT / "pages/1_regresion_qr.py", default_timeout=30).run()
    next(button for button in app.button if button.label == "Resolver con QR →").click().run()
    select = next(box for box in app.selectbox if box.label == "Saltar a una operación")
    select.select_index(len(select.options)-1).run()
    assert not app.exception
    assert any("Solución obtenida" in item.value for item in app.markdown)
    assert any("p(x)=" in item.value for item in app.latex)


@pytest.mark.parametrize("page,control,new_value,marker", [
    ("pages/1_regresion_qr.py", "Ruido de precios (Bs)", 100000, "Resultado con todos los datos"),
    ("pages/2_estabilidad_numerica.py", "log₁₀ ε: acerca las columnas", -3.0, "Tus parámetros seleccionados"),
    ("pages/3_trayectoria_proyectil.py", "Ruido de medición (m)", 2.0, "Calidad de la reconstrucción"),
])
def test_presentation_controls_recalculate_results(page, control, new_value, marker):
    app = AppTest.from_file(ROOT / page, default_timeout=30).run()
    for _ in range(2):
        next(button for button in app.button if button.label == "Siguiente →").click().run()
    before = next(item.value for item in app.markdown if marker in item.value)
    next(slider for slider in app.slider if slider.label == control).set_value(new_value).run()
    after = next(item.value for item in app.markdown if marker in item.value)
    assert not app.exception
    assert before != after


def test_light_theme_and_reduced_motion_keep_manual_qr_controls():
    app = AppTest.from_file(ROOT / "pages/1_regresion_qr.py", default_timeout=30).run()
    next(toggle for toggle in app.toggle if toggle.label == "Tema claro").set_value(True).run()
    assert app.session_state["lab_light_value"] is True
    for _ in range(3):
        next(button for button in app.button if button.label == "Siguiente →").click().run()
    next(toggle for toggle in app.toggle if toggle.label == "Reducir movimiento").set_value(True).run()
    assert not app.exception
    labels = [button.label for button in app.button]
    assert "Paso →" in labels and "Reiniciar" in labels
    assert "▶ Reproducir" not in labels
