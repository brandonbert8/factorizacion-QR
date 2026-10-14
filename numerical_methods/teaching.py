"""Operaciones de enseñanza extraídas de QR y de la solución calculada."""

from dataclasses import dataclass, field
import numpy as np
from .qr import QRResult


def number(value: float) -> str:
    text = f"{float(value):.5g}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        return rf"{mantissa}\times10^{{{int(exponent)}}}"
    return text


def vector_latex(vector: np.ndarray, limit: int = 3) -> str:
    values = np.asarray(vector).ravel()
    entries = [number(value) for value in values[:limit]]
    if len(values) > limit:
        entries.append(r"\vdots")
    return r"\begin{bmatrix}" + r"\\".join(entries) + r"\end{bmatrix}"


@dataclass
class TeachingStep:
    phase: str
    kind: str
    title: str
    formula: str
    explanation: str
    before: np.ndarray
    after: np.ndarray
    auxiliary: np.ndarray | None = None
    values: dict = field(default_factory=dict)


def factorization_steps(qr: QRResult) -> list[TeachingStep]:
    """Operaciones geométricas de QR, independientes de mínimos cuadrados."""
    steps = []
    for previous, frame in zip(qr.frames, qr.frames[1:]):
        k = frame.iteration - 1
        if frame.phase == "normalizar":
            before = previous.matrix[:, k].copy()
            q = frame.vector.copy()
            norm = frame.coefficient
            steps.append(TeachingStep("Construir Q y R", "normalizar", f"MGS · convertir v{k+1} en q{k+1}",
                rf"r_{{{k+1},{k+1}}}=\|v_{{{k+1}}}\|_2={number(norm)},\qquad q_{{{k+1}}}=\frac{{{vector_latex(before)}}}{{{number(norm)}}}={vector_latex(q)}",
                "Dividimos cada componente por la norma. La dirección se conserva y la longitud pasa a uno.",
                before, q, values={"norm": norm, "column": k}))
        elif frame.phase == "proyectar":
            _, j = frame.active_columns
            before = previous.matrix[:, j].copy()
            projection = frame.coefficient * frame.vector
            after = frame.matrix[:, j].copy()
            steps.append(TeachingStep("Construir Q y R", "proyectar", f"MGS · quitar de v{j+1} la dirección q{k+1}",
                rf"r_{{{k+1},{j+1}}}=q_{{{k+1}}}^Tv_{{{j+1}}}={number(frame.coefficient)}"
                rf",\qquad v_{{{j+1}}}^{{nuevo}}={vector_latex(before)}-{vector_latex(projection)}={vector_latex(after)}",
                "Azul: vector original. Amarillo: componente paralela a q. Rosa: lo que queda al restarla. El residuo debe ser perpendicular a q.",
                before, after, projection, {"q": frame.vector.copy(), "coefficient": frame.coefficient,
                                            "orthogonality": float(frame.vector @ after), "column": j}))
        elif frame.phase == "reflejar":
            x = previous.matrix[k:, k].copy()
            v = frame.vector.copy()
            alpha = frame.coefficient
            steps.append(TeachingStep("Construir Q y R", "reflector", f"Householder · construir el espejo H{k+1}",
                rf"\alpha={number(alpha)},\quad v=\frac{{x-\alpha e_1}}{{\|x-\alpha e_1\|_2}}={vector_latex(v)},\quad H=I-2vv^T",
                "Elegimos un espejo cuya normal es v. Ese reflector lleva la subcolumna x al eje de la primera componente.",
                x, frame.matrix[k:, k].copy(), v, {"alpha": alpha, "column": k}))
            for j in range(k, previous.matrix.shape[1]):
                before = previous.matrix[k:, j].copy()
                removed = 2 * v * (v @ before)
                after = frame.matrix[k:, j].copy()
                steps.append(TeachingStep("Construir Q y R", "reflejar", f"Householder · reflejar columna {j+1} con H{k+1}",
                    rf"Hz=z-2v(v^Tz)={vector_latex(before)}-{vector_latex(removed)}={vector_latex(after)}",
                    "El mismo espejo se aplica a todas las columnas pendientes. En la columna objetivo se anulan las entradas bajo la diagonal; las longitudes se conservan.",
                    before, after, v, {"removed": removed, "column": j, "target_column": k,
                                      "norm_before": float(np.linalg.norm(before)), "norm_after": float(np.linalg.norm(after))}))

    return steps


def solution_steps(qr: QRResult, y: np.ndarray) -> list[TeachingStep]:
    """Expone QR, productos Qᵀy y sustitución regresiva."""
    y = np.asarray(y, dtype=float)
    if y.shape != (qr.shape[0],):
        raise ValueError("El vector y debe tener una entrada por fila de A.")
    steps = factorization_steps(qr)
    rhs = qr.Q.T @ y
    for j in range(qr.shape[1]):
        terms = qr.Q[:, j] * y
        steps.append(TeachingStep("Calcular Qᵀy", "producto", f"Proyectar las observaciones sobre q{j+1}",
            rf"b_{{{j+1}}}=q_{{{j+1}}}^Ty=\sum_i q_{{i,{j+1}}}y_i={number(rhs[j])}",
            "Cada barra aporta un producto qᵢ·yᵢ. La suma acumulada construye una entrada del nuevo lado derecho b=Qᵀy.",
            terms, np.cumsum(terms), values={"rhs": float(rhs[j]), "column": j, "q": qr.Q[:, j].copy(), "y": y.copy()}))
    coefficients = np.zeros(qr.shape[1])
    for i in range(qr.shape[1]-1, -1, -1):
        before = coefficients.copy()
        contribution = float(qr.R[i, i+1:] @ coefficients[i+1:])
        coefficients[i] = (rhs[i] - contribution) / qr.R[i, i]
        steps.append(TeachingStep("Resolver Rβ=b", "sustituir", f"Sustitución regresiva · obtener β{i+1}",
            rf"\beta_{{{i+1}}}=\frac{{b_{{{i+1}}}-\sum_{{j>{i+1}}}r_{{{i+1},j}}\beta_j}}{{r_{{{i+1},{i+1}}}}}"
            rf"=\frac{{{number(rhs[i])}-({number(contribution)})}}{{{number(qr.R[i,i])}}}={number(coefficients[i])}",
            "Empezamos por la última fila de R. Usamos los coeficientes ya calculados y despejamos uno nuevo en cada paso.",
            before, coefficients.copy(), values={"row": i, "R": qr.R.copy(), "rhs": rhs.copy(), "contribution": contribution}))
    return steps


def factorization_states(qr: QRResult) -> list[dict]:
    """Estados al terminar cada iteración, reconstruidos de los registros reales.

    MGS: A = Q_k R_k + V_pendiente. Householder: A = Q_k B_k,
    conservando Q completa durante la construcción.
    """
    a = qr.frames[0].matrix
    m, n = a.shape
    states = []
    if qr.method == "Householder":
        q = np.eye(m)
        states.append(dict(Q=q.copy(), R=a.copy(), pending=None))
        for frame in qr.frames:
            if frame.phase != "reflejar":
                continue
            k, v = frame.iteration - 1, frame.vector
            q[:, k:] -= 2 * np.outer(q[:, k:] @ v, v)
            states.append(dict(Q=q.copy(), R=frame.matrix.copy(), pending=None))
    else:
        q = np.zeros((m, n))
        states.append(dict(Q=q.copy(), R=np.zeros((n, n)), pending=a.copy()))
        for step in qr.steps:
            k = step.iteration - 1
            q[:, k] = step.data["q"]
            last_frame = next(frame for frame in reversed(qr.frames)
                              if frame.iteration == k + 1 and frame.phase != "final")
            pending = last_frame.matrix.copy()
            pending[:, :k+1] = 0
            states.append(dict(Q=q.copy(), R=step.data["R parcial"].copy(), pending=pending))
    return states
