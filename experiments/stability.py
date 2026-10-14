import numpy as np


def experimental_matrix(epsilon: float) -> np.ndarray:
    return np.array([[1, 1], [1, 1 + epsilon], [1, 1 - epsilon]], dtype=float)


def observations(epsilon: float, beta: np.ndarray, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    a = experimental_matrix(epsilon)
    rng = np.random.default_rng(seed)
    return a @ beta + rng.normal(0, noise, 3), a
