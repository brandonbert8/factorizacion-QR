"""Mínimos cuadrados por QR y sustitución regresiva explícita."""

from dataclasses import dataclass
import numpy as np
from .qr import QRResult, modified_gram_schmidt, householder


@dataclass
class LeastSquaresResult:
    coefficients: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray
    rhs: np.ndarray
    qr: QRResult


def back_substitution(r: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    r, rhs = np.asarray(r, float), np.asarray(rhs, float)
    if r.ndim != 2 or r.shape[0] != r.shape[1] or rhs.shape != (r.shape[0],):
        raise ValueError("Dimensiones incompatibles para sustitución regresiva.")
    n = len(rhs)
    tolerance = np.finfo(float).eps * n * max(1.0, float(np.linalg.norm(r, ord=np.inf)))
    answer = np.empty(n)
    for i in range(n - 1, -1, -1):
        if abs(r[i, i]) <= tolerance:
            raise ValueError(f"R singular o casi singular en la diagonal {i+1}.")
        answer[i] = (rhs[i] - r[i, i+1:] @ answer[i+1:]) / r[i, i]
    return answer


def least_squares(a: np.ndarray, y: np.ndarray, method: str = "Householder") -> LeastSquaresResult:
    a, y = np.asarray(a, float), np.asarray(y, float)
    if a.ndim != 2 or y.shape != (a.shape[0],) or not np.all(np.isfinite(y)):
        raise ValueError("Dimensiones o valores inválidos para mínimos cuadrados.")
    if method == "Householder":
        qr = householder(a)
    elif method == "Gram-Schmidt modificado":
        qr = modified_gram_schmidt(a)
    else:
        raise ValueError("Método QR desconocido.")
    rhs = qr.Q.T @ y
    coefficients = back_substitution(qr.R, rhs)
    fitted = a @ coefficients
    return LeastSquaresResult(coefficients, fitted, y - fitted, rhs, qr)
