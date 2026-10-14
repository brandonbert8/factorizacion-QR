import numpy as np
import pytest
from scipy.linalg import lstsq
from experiments.housing import design_matrix, simulated_housing
from experiments.projectile import simulated_projectile
from experiments.stability import experimental_matrix
from numerical_methods import householder, least_squares, modified_gram_schmidt
from numerical_methods.least_squares import back_substitution
from numerical_methods.metrics import fit_metrics, relative_error


@pytest.mark.parametrize("algorithm", [modified_gram_schmidt, householder])
@pytest.mark.parametrize("shape", [(5, 2), (7, 3), (3, 3)])
def test_qr_economic_reconstruction_and_steps(algorithm, shape):
    a = np.random.default_rng(32).normal(size=shape)
    result = algorithm(a)
    m, n = shape
    assert result.Q.shape == (m, n)
    assert result.R.shape == (n, n)
    assert len(result.steps) == n
    np.testing.assert_allclose(result.Q @ result.R, a, atol=1e-12)
    np.testing.assert_allclose(result.Q.T @ result.Q, np.eye(n), atol=1e-12)
    np.testing.assert_allclose(result.R, np.triu(result.R), atol=1e-14)


@pytest.mark.parametrize("algorithm", [modified_gram_schmidt, householder])
def test_dependent_and_tiny_columns_rejected(algorithm):
    with pytest.raises(ValueError, match="dependiente"):
        algorithm(np.array([[1., 2.], [2., 4.], [3., 6.]]))
    with pytest.raises(ValueError, match="dependiente"):
        algorithm(np.array([[1., 1e-18], [0., 0.], [0., 0.]]))


def test_triangular_solver_and_singularity():
    r = np.array([[2., 3.], [0., 4.]])
    np.testing.assert_allclose(back_substitution(r, np.array([8., 8.])), [1., 2.])
    with pytest.raises(ValueError, match="singular"):
        back_substitution(np.array([[1., 2.], [0., 0.]]), np.array([1., 1.]))


@pytest.mark.parametrize("method", ["Householder", "Gram-Schmidt modificado"])
def test_least_squares_matches_scipy(method):
    rng = np.random.default_rng(8)
    a = rng.normal(size=(12, 3))
    y = rng.normal(size=12)
    result = least_squares(a, y, method)
    np.testing.assert_allclose(result.coefficients, lstsq(a, y)[0], atol=1e-12)
    np.testing.assert_allclose(result.residuals, y - a @ result.coefficients)


def test_simulated_housing_coefficients_reasonable():
    data = simulated_housing(30, 10000, 3)
    a = design_matrix(data.superficie_m2.to_numpy(), 1)
    result = least_squares(a, data.precio_bob.to_numpy())
    assert abs(result.coefficients[1] - 5100) < 100
    assert fit_metrics(data.precio_bob, result.fitted)["R²"] > .99


def test_projectile_recovers_parameters_without_noise():
    data = simulated_projectile(30, 40, 2, 9.81, 16, 0, 4)
    t = data.t_s.to_numpy()
    a = np.column_stack([np.ones_like(t), t, t**2])
    result = least_squares(a, data.y_m.to_numpy())
    np.testing.assert_allclose(result.coefficients[0], 2, atol=1e-12)
    np.testing.assert_allclose(result.coefficients[2], -9.81 / 2, atol=1e-12)


def test_conditioning_increases_solution_sensitivity():
    beta = np.array([2., -1.])
    direction = np.array([0., 1., -1.])
    errors = []
    for epsilon in [1e-2, 1e-6]:
        a = experimental_matrix(epsilon)
        y = a @ beta + 1e-8 * direction
        errors.append(relative_error(least_squares(a, y).coefficients, beta))
    assert errors[1] > errors[0] * 1000


def test_metrics_known_values():
    scores = fit_metrics(np.array([1., 2., 3.]), np.array([1., 2., 2.]))
    assert scores["MSE"] == pytest.approx(1/3)
    assert scores["RMSE"] == pytest.approx(np.sqrt(1/3))
    assert scores["R²"] == pytest.approx(.5)
