from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from sklearn.svm import SVR


@dataclass
class HHOSVRConfig:
    C_bounds: tuple[float, float] = (0.1, 100.0)
    epsilon_bounds: tuple[float, float] = (0.001, 0.1)
    gamma_bounds: tuple[float, float] = (1e-4, 0.05)
    population_size: int = 10
    iterations: int = 15
    seed: int = 42


def _levy_flight(rng: np.random.Generator, dim: int, beta: float = 1.5) -> np.ndarray:
    sigma = (
        math.gamma(1 + beta)
        * math.sin(math.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = rng.normal(0, sigma, size=dim)
    v = rng.normal(0, 1, size=dim)
    step = u / (np.abs(v) ** (1 / beta))
    return step


def _decode_params(pos: np.ndarray, bounds: np.ndarray) -> dict:
    # pos in log10 space for C, gamma, epsilon
    C = 10 ** pos[0]
    gamma = 10 ** pos[1]
    epsilon = 10 ** pos[2]
    C = float(np.clip(C, bounds[0, 0], bounds[0, 1]))
    gamma = float(np.clip(gamma, bounds[1, 0], bounds[1, 1]))
    epsilon = float(np.clip(epsilon, bounds[2, 0], bounds[2, 1]))
    return {"C": C, "gamma": gamma, "epsilon": epsilon}


def optimize_hho_svr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: HHOSVRConfig,
) -> dict:
    rng = np.random.default_rng(config.seed)
    dim = 3

    # log10 bounds for C, gamma, epsilon
    bounds = np.array(
        [
            [config.C_bounds[0], config.C_bounds[1]],
            [config.gamma_bounds[0], config.gamma_bounds[1]],
            [config.epsilon_bounds[0], config.epsilon_bounds[1]],
        ],
        dtype=np.float64,
    )
    log_bounds = np.log10(bounds)

    n = int(config.population_size)
    T = int(config.iterations)

    positions = rng.uniform(log_bounds[:, 0], log_bounds[:, 1], size=(n, dim))

    def fitness(pos: np.ndarray) -> float:
        params = _decode_params(pos, bounds)
        try:
            model = SVR(C=params["C"], gamma=params["gamma"], epsilon=params["epsilon"], kernel="rbf")
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            rmse = float(np.sqrt(np.mean((preds - y_val) ** 2)))
            return rmse
        except Exception:
            return float("inf")

    fitness_vals = np.array([fitness(p) for p in positions], dtype=np.float64)
    best_idx = int(np.argmin(fitness_vals))
    rabbit = positions[best_idx].copy()
    rabbit_fit = float(fitness_vals[best_idx])

    for t in range(T):
        E0 = rng.uniform(-1, 1, size=n)
        E = 2.0 * E0 * (1.0 - (t + 1) / T)
        X_mean = positions.mean(axis=0)
        for i in range(n):
            r = rng.random()
            q = rng.random()
            J = 2.0 * (1.0 - rng.random())

            if abs(E[i]) >= 1:
                if q >= 0.5:
                    rand_idx = int(rng.integers(0, n))
                    X_rand = positions[rand_idx]
                    r1 = rng.random()
                    r2 = rng.random()
                    new_pos = X_rand - r1 * np.abs(X_rand - 2.0 * r2 * positions[i])
                else:
                    r3 = rng.random()
                    r4 = rng.random()
                    new_pos = (rabbit - X_mean) - r3 * (log_bounds[:, 0] + r4 * (log_bounds[:, 1] - log_bounds[:, 0]))
            else:
                if r >= 0.5 and abs(E[i]) >= 0.5:
                    new_pos = rabbit - E[i] * np.abs(rabbit - positions[i])
                elif r >= 0.5 and abs(E[i]) < 0.5:
                    new_pos = rabbit - E[i] * np.abs(rabbit - X_mean)
                elif r < 0.5 and abs(E[i]) >= 0.5:
                    Y = rabbit - E[i] * np.abs(J * rabbit - positions[i])
                    Z = Y + 0.01 * _levy_flight(rng, dim)
                    fit_Y = fitness(Y)
                    fit_Z = fitness(Z)
                    if fit_Y < fitness_vals[i] or fit_Z < fitness_vals[i]:
                        new_pos = Y if fit_Y <= fit_Z else Z
                    else:
                        new_pos = positions[i]
                else:
                    Y = rabbit - E[i] * np.abs(J * rabbit - X_mean)
                    Z = Y + 0.01 * _levy_flight(rng, dim)
                    fit_Y = fitness(Y)
                    fit_Z = fitness(Z)
                    if fit_Y < fitness_vals[i] or fit_Z < fitness_vals[i]:
                        new_pos = Y if fit_Y <= fit_Z else Z
                    else:
                        new_pos = positions[i]

            new_pos = np.clip(new_pos, log_bounds[:, 0], log_bounds[:, 1])
            new_fit = fitness(new_pos)
            if new_fit < fitness_vals[i]:
                positions[i] = new_pos
                fitness_vals[i] = new_fit

        best_idx = int(np.argmin(fitness_vals))
        if fitness_vals[best_idx] < rabbit_fit:
            rabbit_fit = float(fitness_vals[best_idx])
            rabbit = positions[best_idx].copy()

    best_params = _decode_params(rabbit, bounds)
    return best_params
