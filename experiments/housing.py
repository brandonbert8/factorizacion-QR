import numpy as np
import pandas as pd


def simulated_housing(count: int = 16, noise: float = 25000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    area = np.sort(rng.uniform(45, 220, count))
    price = 85000 + 5100 * area + rng.normal(0, noise, count)
    return pd.DataFrame({"superficie_m2": np.round(area, 1), "precio_bob": np.round(price, 0)})


def design_matrix(area: np.ndarray, degree: int) -> np.ndarray:
    if degree not in (1, 2):
        raise ValueError("El grado debe ser 1 o 2.")
    area = np.asarray(area, float)
    return np.column_stack([area**power for power in range(degree + 1)])
