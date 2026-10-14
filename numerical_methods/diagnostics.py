"""Comprobaciones de QR y sensibilidad con los datos completos del ejercicio."""

import numpy as np
from scipy.linalg import lstsq, qr as scipy_qr
from .least_squares import least_squares
from .teaching import factorization_states

METHODS = ("Gram-Schmidt modificado", "Householder")


def aligned_factors(result):
    """Fija diagonal positiva sin alterar el producto Q R."""
    signs = np.where(np.diag(result.R) < 0, -1., 1.)
    return result.Q * signs, signs[:, None] * result.R


def orthogonality_history(result):
    """Error de las primeras k columnas ya construidas, sin contar las vacías."""
    history = []
    for k, state in enumerate(factorization_states(result)):
        q = state["Q"][:, :k]
        error = q.T @ q - np.eye(k)
        history.append(dict(iteration=k, error=error, norm=float(np.linalg.norm(error, "fro"))))
    return history


def reference_comparison(a, y, results):
    """SciPy valida los factores y la solución por una implementación independiente."""
    q, r = scipy_qr(a, mode="economic")
    beta = lstsq(a, y, lapack_driver="gelsy")[0]
    rows = []
    for name, result in results.items():
        rows.append({"Método": name, "Error QR": result.qr.reconstruction_error,
                     "Ortogonalidad": result.qr.orthogonality_error,
                     "Diferencia β / ref.": float(np.linalg.norm(result.coefficients-beta) / np.linalg.norm(beta))
                     if np.linalg.norm(beta) > 0 else np.nan})
    rows.append({"Método": "SciPy · referencia", "Error QR": float(np.linalg.norm(a-q@r, "fro")/np.linalg.norm(a, "fro")),
                 "Ortogonalidad": float(np.linalg.norm(q.T@q-np.eye(a.shape[1]), "fro")),
                 "Diferencia β / ref.": 0. if np.linalg.norm(beta) > 0 else np.nan})
    return rows


def perturbation_study(a, y, levels=None):
    """Perturba y en la dirección singular de mayor sensibilidad absoluta.

    No cambia A. Cada método compara sus coeficientes perturbados con su propia
    solución base; la referencia usa QR con pivoteo de SciPy (gelsy).
    """
    levels = np.logspace(-12, -3, 13) if levels is None else np.asarray(levels, float)
    if levels.ndim != 1 or not len(levels) or not np.all(np.isfinite(levels)) or np.any(levels <= 0):
        raise ValueError("Las perturbaciones relativas deben ser positivas y finitas.")
    scale = float(np.linalg.norm(y))
    if scale == 0:
        raise ValueError("La perturbación relativa requiere un vector y no nulo.")
    u, singular, _ = np.linalg.svd(a, full_matrices=False)
    direction = u[:, -1]
    baseline = {name: least_squares(a, y, name).coefficients for name in METHODS}
    baseline["SciPy · referencia"] = lstsq(a, y, lapack_driver="gelsy")[0]
    changes = {name: [] for name in baseline}
    actual_levels = []
    for level in levels:
        perturbed = y + level * scale * direction
        actual_levels.append(float(np.linalg.norm(perturbed-y) / scale))
        for name, base in baseline.items():
            beta = (lstsq(a, perturbed, lapack_driver="gelsy")[0] if name == "SciPy · referencia"
                    else least_squares(a, perturbed, name).coefficients)
            changes[name].append(float(np.linalg.norm(beta-base)/np.linalg.norm(base))
                                 if np.linalg.norm(base) > 0 else np.nan)
    beta_norm = float(np.linalg.norm(baseline["SciPy · referencia"]))
    gain = scale / (singular[-1]*beta_norm) if beta_norm > 0 else np.nan
    return dict(levels=levels, actual_levels=np.array(actual_levels), direction=direction,
                changes={name: np.array(values) for name, values in changes.items()}, gain=gain)
