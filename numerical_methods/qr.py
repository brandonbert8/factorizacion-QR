"""Factorizaciones QR económicas con registro de operaciones reales."""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class QRStep:
    iteration: int
    operation: str
    formula: str
    interpretation: str
    data: dict[str, np.ndarray | float] = field(default_factory=dict)


@dataclass
class QRFrame:
    """Estado numérico verificable para una escena de la factorización."""

    iteration: int
    phase: str
    title: str
    formula: str
    explanation: str
    matrix: np.ndarray
    matrix_label: str
    active_columns: tuple[int, ...] = ()
    vector: np.ndarray | None = None
    coefficient: float | None = None


@dataclass
class QRResult:
    Q: np.ndarray
    R: np.ndarray
    shape: tuple[int, int]
    method: str
    steps: list[QRStep]
    reconstruction_error: float
    orthogonality_error: float
    warnings: list[str]
    frames: list[QRFrame] = field(default_factory=list)


def _input(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    a = np.asarray(matrix, dtype=float)
    if a.ndim != 2 or not all(a.shape) or a.shape[0] < a.shape[1]:
        raise ValueError("Se requiere una matriz bidimensional con m ≥ n ≥ 1.")
    if not np.all(np.isfinite(a)):
        raise ValueError("La matriz contiene valores no finitos.")
    scale = float(np.linalg.norm(a, ord=2))
    if scale == 0:
        raise ValueError("La matriz es nula y no tiene rango completo.")
    return a, np.finfo(float).eps * max(a.shape) * scale


def _result(a: np.ndarray, q: np.ndarray, r: np.ndarray, method: str,
            steps: list[QRStep], frames: list[QRFrame]) -> QRResult:
    rec = float(np.linalg.norm(a - q @ r, "fro") / np.linalg.norm(a, "fro"))
    orth = float(np.linalg.norm(q.T @ q - np.eye(q.shape[1]), "fro"))
    condition = float(np.linalg.cond(a))
    warnings = []
    if condition > 1 / np.sqrt(np.finfo(float).eps):
        warnings.append(f"κ(A) = {condition:.2e}: la solución puede ser muy sensible al ruido.")
    return QRResult(q, r, a.shape, method, steps, rec, orth, warnings, frames)


def modified_gram_schmidt(matrix: np.ndarray) -> QRResult:
    """MGS de una pasada. R tiene diagonal positiva por construcción."""
    a, tolerance = _input(matrix)
    m, n = a.shape
    v = a.copy()
    q = np.zeros((m, n))
    r = np.zeros((n, n))
    steps = []
    frames = [QRFrame(0, "inicio", "Matriz de partida", r"A=[a_1,\ldots,a_n]",
                      "Cada columna empieza con sus valores originales.", v.copy(), "Columnas pendientes")]
    for k in range(n):
        norm = float(np.linalg.norm(v[:, k]))
        if norm <= tolerance:
            raise ValueError(f"Columna {k+1} dependiente o casi nula (norma {norm:.2e}; umbral {tolerance:.2e}).")
        r[k, k] = norm
        q[:, k] = v[:, k] / norm
        frames.append(QRFrame(k + 1, "normalizar", f"Normalizar columna {k+1}",
                              r"q_k=v_k/\|v_k\|_2", "Se obtiene una dirección unitaria para proyectar las columnas restantes.",
                              v.copy(), "Vectores residuales V", (k,), q[:, k].copy(), norm))
        projections = {}
        for j in range(k + 1, n):
            coefficient = float(q[:, k] @ v[:, j])
            r[k, j] = coefficient
            v[:, j] -= coefficient * q[:, k]
            projections[f"r{k+1},{j+1}"] = coefficient
            frames.append(QRFrame(k + 1, "proyectar", f"Restar proyección de columna {j+1}",
                                  r"r_{kj}=q_k^Tv_j,\quad v_j\leftarrow v_j-r_{kj}q_k",
                                  "La columna pierde su componente en la nueva dirección ortogonal.",
                                  v.copy(), "Vectores residuales V", (k, j), q[:, k].copy(), coefficient))
        steps.append(QRStep(k + 1, "Normalizar y restar proyecciones",
                            r"r_{kk}=\|v_k\|_2,\ q_k=v_k/r_{kk},\ v_j\leftarrow v_j-(q_k^Tv_j)q_k",
                            "La columna actual forma un vector unitario; las restantes pierden su componente en esa dirección.",
                            {"vector antes": a[:, k].copy(), "vector residual": v[:, k].copy(),
                             "q": q[:, k].copy(), "R parcial": r.copy(), **projections}))
    frames.append(QRFrame(n, "final", "Factorización completa", r"A\approx QR",
                          "Las columnas de Q son ortonormales y R guarda las proyecciones.",
                          r.copy(), "R final"))
    return _result(a, q, r, "Gram-Schmidt modificado", steps, frames)


def householder(matrix: np.ndarray) -> QRResult:
    """QR económica; conserva n reflectores y forma únicamente las n columnas de Q."""
    a, tolerance = _input(matrix)
    m, n = a.shape
    work = a.copy()
    reflectors: list[tuple[int, np.ndarray]] = []
    steps = []
    frames = [QRFrame(0, "inicio", "Matriz de partida", r"R^{(0)}=A",
                      "Los reflectores transformarán esta matriz en una triangular superior.",
                      work.copy(), "R en construcción")]
    for k in range(n):
        x = work[k:, k].copy()
        norm = float(np.linalg.norm(x))
        if norm <= tolerance:
            raise ValueError(f"Columna {k+1} dependiente o casi nula (norma {norm:.2e}; umbral {tolerance:.2e}).")
        # El signo evita cancelación en la construcción de v.
        alpha = -np.copysign(norm, x[0])
        v = x.copy()
        v[0] -= alpha
        v /= np.linalg.norm(v)
        work[k:, k:] -= 2 * np.outer(v, v @ work[k:, k:])
        work[k+1:, k] = 0.0
        reflectors.append((k, v.copy()))
        steps.append(QRStep(k + 1, "Aplicar reflector de Householder",
                            r"H_k=I-2v_kv_k^T,\ R\leftarrow H_kR",
                            "El reflector anula las entradas bajo la diagonal en la columna actual.",
                            {"subcolumna": x, "reflector v": v.copy(), "R parcial": work.copy()}))
        frames.append(QRFrame(k + 1, "reflejar", f"Reflector {k+1}",
                              r"H_k=I-2v_kv_k^T,\quad R\leftarrow H_kR",
                              "La transformación anula las entradas bajo la diagonal de la columna resaltada.",
                              work.copy(), "R en construcción", (k,), v.copy(), alpha))
    q = np.eye(m, n)
    for k, v in reversed(reflectors):
        q[k:, :] -= 2 * np.outer(v, v @ q[k:, :])
    r = np.triu(work[:n, :])
    frames.append(QRFrame(n, "final", "Factorización completa", r"A\approx QR",
                          "R es triangular superior; los reflectores aplicados en orden inverso forman Q.",
                          r.copy(), "R final"))
    return _result(a, q, r, "Householder", steps, frames)
