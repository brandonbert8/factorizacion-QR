"""Métricas y verificaciones compartidas."""

import numpy as np


def fit_metrics(observed: np.ndarray, fitted: np.ndarray) -> dict[str, float]:
    observed, fitted = np.asarray(observed, float), np.asarray(fitted, float)
    if observed.shape != fitted.shape or observed.size == 0:
        raise ValueError("Las series deben tener la misma longitud no nula.")
    residual = observed - fitted
    mse = float(np.mean(residual**2))
    total = float(np.sum((observed - observed.mean())**2))
    return {"MSE": mse, "RMSE": float(np.sqrt(mse)),
            "R²": float(1 - np.sum(residual**2) / total) if total > 0 else float("nan")}


def relative_error(estimate: np.ndarray, reference: np.ndarray) -> float:
    reference = np.asarray(reference, float)
    denominator = float(np.linalg.norm(reference))
    return float(np.linalg.norm(np.asarray(estimate, float) - reference) / denominator) if denominator else float("nan")
