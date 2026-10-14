import numpy as np
import pytest
from numerical_methods import least_squares
from numerical_methods.teaching import solution_steps
from components.operations import operation_figure


@pytest.mark.parametrize("method", ["Householder", "Gram-Schmidt modificado"])
@pytest.mark.parametrize("shape", [(3, 2), (8, 3)])
def test_teaching_resolves_same_problem_as_algorithm(method, shape):
    rng = np.random.default_rng(12)
    a, y = rng.normal(size=shape), rng.normal(size=shape[0])
    result = least_squares(a, y, method)
    steps = solution_steps(result.qr, y)
    np.testing.assert_allclose(steps[-1].after, result.coefficients, atol=1e-12)
    assert len([s for s in steps if s.kind == "producto"]) == shape[1]
    assert len([s for s in steps if s.kind == "sustituir"]) == shape[1]
    for step in steps:
        if step.kind == "proyectar":
            np.testing.assert_allclose(step.before-step.auxiliary, step.after, atol=1e-12)
        elif step.kind == "reflejar":
            v = step.auxiliary
            np.testing.assert_allclose(step.before-2*v*(v @ step.before), step.after, atol=1e-12)
            assert np.linalg.norm(step.after) == pytest.approx(np.linalg.norm(step.before))
        elif step.kind == "producto":
            assert step.after[-1] == pytest.approx(result.rhs[step.values["column"]])
        elif step.kind == "sustituir":
            row = step.values["row"]
            assert result.qr.R[row] @ step.after == pytest.approx(result.rhs[row])
        if step.kind in ("normalizar", "proyectar", "reflector", "reflejar"):
            figure, frames = operation_figure(step)
            endpoint = np.array([frames[-1].data[0].x[-1], frames[-1].data[0].y[-1]])
            assert np.linalg.norm(endpoint) == pytest.approx(np.linalg.norm(step.after), abs=1e-12)


def test_methods_have_different_operation_sequences():
    a = np.array([[1., 1.], [1., 2.], [1., 4.]])
    y = np.array([1., 3., 4.])
    kinds = {method: {step.kind for step in solution_steps(least_squares(a, y, method).qr, y)}
             for method in ["Householder", "Gram-Schmidt modificado"]}
    assert {"reflector", "reflejar"} <= kinds["Householder"]
    assert {"normalizar", "proyectar"} <= kinds["Gram-Schmidt modificado"]
    assert "reflejar" not in kinds["Gram-Schmidt modificado"]
