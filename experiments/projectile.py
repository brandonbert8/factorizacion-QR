import numpy as np
import pandas as pd


def flight_time(v0: float, angle_deg: float, y0: float, gravity: float) -> float:
    vy = v0 * np.sin(np.deg2rad(angle_deg))
    return float((vy + np.sqrt(vy**2 + 2 * gravity * y0)) / gravity)


def simulated_projectile(v0: float, angle_deg: float, y0: float, gravity: float,
                         count: int, noise: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, flight_time(v0, angle_deg, y0, gravity), count)
    angle = np.deg2rad(angle_deg)
    x = v0 * np.cos(angle) * t
    y = y0 + v0 * np.sin(angle) * t - 0.5 * gravity * t**2
    return pd.DataFrame({"t_s": t, "x_m": x + rng.normal(0, noise, count),
                         "y_m": y + rng.normal(0, noise, count)})


def physical_quantities(y0: float, vy: float, vx: float, gravity: float) -> dict[str, float]:
    if gravity <= 0 or vx <= 0:
        raise ValueError("La gravedad y velocidad horizontal estimadas deben ser positivas.")
    discriminant = vy**2 + 2 * gravity * y0
    if discriminant < 0:
        raise ValueError("La trayectoria estimada no cruza el suelo en tiempo real.")
    tf = float((vy + np.sqrt(discriminant)) / gravity)
    return {"v₀ (m/s)": float(np.hypot(vx, vy)), "θ (°)": float(np.rad2deg(np.arctan2(vy, vx))),
            "y₀ (m)": float(y0), "g (m/s²)": float(gravity),
            "altura máxima (m)": float(y0 + max(vy, 0)**2 / (2 * gravity)),
            "tiempo de vuelo (s)": tf, "alcance (m)": float(vx * tf)}
