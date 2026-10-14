from pathlib import Path
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest
from numerical_methods import householder, modified_gram_schmidt, least_squares
from numerical_methods.diagnostics import aligned_factors, orthogonality_history, reference_comparison, perturbation_study
from components.qr_figures import reconstruction_figure, orthogonality_figure, perturbation_figure

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("shape", [(3, 2), (8, 3), (60, 3), (1, 1)])
def test_reconstruction_animation_accumulates_actual_terms(shape):
    a = np.random.default_rng(17).normal(size=shape)
    qr_results = [modified_gram_schmidt(a), householder(a)]
    for column in range(shape[1]):
        fig, frames = reconstruction_figure(*qr_results, column)
        assert len(frames) == column+2
        for count, frame in enumerate(frames):
            for offset, qr in zip([0, 3], qr_results):
                np.testing.assert_allclose(frame.data[offset].y, a[:, column])
                expected = qr.Q[:, :count] @ qr.R[:count, column]
                np.testing.assert_allclose(frame.data[offset+2].y, expected, atol=1e-12)
                term = qr.Q[:, count-1]*qr.R[count-1,column] if count else np.zeros(shape[0])
                np.testing.assert_allclose(frame.data[offset+1].y, term)
                if count == column+1:
                    np.testing.assert_allclose(frame.data[offset+2].y, a[:,column], atol=1e-12)


@pytest.mark.parametrize("algorithm", [householder, modified_gram_schmidt])
def test_sign_alignment_preserves_qr_and_history_uses_completed_columns(algorithm):
    a = np.random.default_rng(2).normal(size=(9, 3))
    result = algorithm(a)
    q, r = aligned_factors(result)
    assert np.all(np.diag(r) > 0)
    np.testing.assert_allclose(q@r, result.Q@result.R, atol=1e-12)
    for k, state in enumerate(orthogonality_history(result)):
        expected = result.Q[:, :k].T @ result.Q[:, :k] - np.eye(k)
        np.testing.assert_allclose(state["error"], expected, atol=1e-15)
        assert state["norm"] == pytest.approx(np.linalg.norm(expected,"fro"), abs=1e-15)
    assert orthogonality_history(result)[-1]["norm"] == pytest.approx(result.orthogonality_error, abs=1e-15)


def test_orthogonality_animation_leaves_unbuilt_columns_blank():
    a = np.array([[1.,1.], [1.,1+1e-8], [1.,1-1e-8]])
    results = [modified_gram_schmidt(a), householder(a)]
    _, frames = orthogonality_figure(*results)
    for k, frame in enumerate(frames):
        for data, result in zip(frame.data, results):
            actual = np.asarray(data.z)
            np.testing.assert_allclose(actual[:k,:k], orthogonality_history(result)[k]["error"])
            assert np.isnan(actual[k:,:]).all()
            assert np.isnan(actual[:,k:]).all()


def test_perturbation_study_matches_known_diagonal_system_and_preserves_input():
    a = np.diag([1., .01])
    y = np.array([1., .01])
    original_a, original_y = a.copy(), y.copy()
    study = perturbation_study(a, y, np.array([1e-7, 1e-5, 1e-3]))
    expected = study["levels"] * np.linalg.norm(y) / (.01*np.sqrt(2))
    for changes in study["changes"].values():
        np.testing.assert_allclose(changes, expected, rtol=1e-8)
    np.testing.assert_allclose(study["actual_levels"], study["levels"], rtol=1e-9)
    np.testing.assert_array_equal(a, original_a)
    np.testing.assert_array_equal(y, original_y)
    fig, frames = perturbation_figure(study)
    for data, name in zip(frames[-1].data, study["changes"]):
        np.testing.assert_allclose(data.customdata, study["changes"][name])
    with pytest.raises(ValueError):
        perturbation_study(a, np.zeros(2))
    with pytest.raises(ValueError):
        perturbation_study(a, y, [0])


def test_reference_is_same_system_and_reports_relative_coefficient_difference():
    a = np.random.default_rng(14).normal(size=(12, 3))
    y = np.arange(12.)
    results = {method: least_squares(a,y,method) for method in ["Gram-Schmidt modificado", "Householder"]}
    rows = reference_comparison(a, y, results)
    assert len(rows) == 3
    assert rows[-1]["Método"] == "SciPy · referencia"
    assert all(row["Error QR"] < 1e-12 for row in rows)
    assert all(row["Diferencia β / ref."] < 1e-12 for row in rows)


@pytest.mark.parametrize("page,key", [
    ("pages/1_regresion_qr.py", "housing_qr"),
    ("pages/2_estabilidad_numerica.py", "stability_qr"),
    ("pages/3_trayectoria_proyectil.py", "projectile_qr"),
])
def test_new_animations_and_subtle_observations_in_original_examples(page, key):
    app = AppTest.from_file(ROOT / page, default_timeout=30).run()
    assert not app.exception
    assert "Guion" not in [p.proto.popover.label for p in app.get("popover")]
    next(b for b in app.button if b.label == "Resolver con QR →").click().run()
    for view, count in [("Reconstruir A",1),("Precisión y sensibilidad",2)]:
        next(r for r in app.radio if r.label == "Resolución QR de este ejemplo").set_value(view).run()
        assert not app.exception
        assert len(app.get("plotly_chart")) == count
        next(t for t in app.toggle if t.label == "Reducir movimiento").set_value(True).run()
        assert not app.exception
    next(s for s in app.slider if s.label == "Intervalo de log₁₀ δ").set_value((-10,-2)).run()
    assert not app.exception
    for _ in range(1):
        next(b for b in app.button if b.label == "Siguiente →").click().run()
        assert not app.exception
        assert "Guion" not in [p.proto.popover.label for p in app.get("popover")]


def test_home_has_no_presentation_download_or_extra_page():
    app = AppTest.from_file(ROOT / "app.py", default_timeout=30).run()
    assert not app.exception
    assert len(app.get("page_link")) == 3
    assert not app.get("download_button")
