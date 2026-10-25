"""Reward-maximizing rate when the distance is random, d ~ Poisson(lambda) + 1,
over a fine grid of mean distances (Figure S3, top)."""

import numpy as np
from tqdm import tqdm

from competitive_exploration.paths import DATA_DIR
from competitive_exploration.poisson import optimal_eps_poisson

T_VALUES = np.array([1000, 5000, 100000])
R_VALUES = np.linspace(1.05, 20, 1000)
MEAN_DISTANCES = np.linspace(1, 30, 1000)
EPS_GRID = np.linspace(0, 1, 201)


def main():
    opt_eps = np.array(
        [
            optimal_eps_poisson(1 - 1 / T, R_VALUES, EPS_GRID, MEAN_DISTANCES - 1)
            for T in tqdm(T_VALUES)
        ]
    )

    np.savez(
        DATA_DIR / "opt_eps_poisson.npz",
        opt_eps=opt_eps,
        T_values=T_VALUES,
        R_values=R_VALUES,
        mean_distances=MEAN_DISTANCES,
        eps_grid=EPS_GRID,
    )


if __name__ == "__main__":
    main()
