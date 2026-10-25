"""Reward-maximizing exploration rate of a lone agent over (d, R) (Figures 2B, 2C, S3)."""

import numpy as np
from tqdm import tqdm

from competitive_exploration.paths import DATA_DIR
from competitive_exploration.single_agent import site_occupancy_grid

T_VALUES = np.array([1000, 5000, 100000])
D_VALUES = np.arange(1, 31)
R_VALUES = np.linspace(1.05, 20, 200)
EPS_GRID = np.linspace(0, 1, 201)


def main():
    opt_eps = np.empty((len(T_VALUES), len(D_VALUES), len(R_VALUES)))
    for t, T in enumerate(T_VALUES):
        for a, d in enumerate(tqdm(D_VALUES, desc=f"T={T}")):
            alpha, beta = site_occupancy_grid(d, EPS_GRID, 1 - 1 / T)
            payoff = alpha[:, None] + beta[:, None] * R_VALUES
            opt_eps[t, a] = EPS_GRID[np.argmax(payoff, axis=0)]

    np.savez(
        DATA_DIR / "opt_eps_single.npz",
        opt_eps=opt_eps,
        T_values=T_VALUES,
        d_values=D_VALUES,
        R_values=R_VALUES,
        eps_grid=EPS_GRID,
    )


if __name__ == "__main__":
    main()
