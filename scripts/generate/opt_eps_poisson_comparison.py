"""Optimal rate with a random distance minus the optimal rate at the matching fixed
distance, on the grid of opt_eps_single.npz (Figure S3, bottom)."""

import numpy as np

from competitive_exploration.paths import DATA_DIR
from competitive_exploration.poisson import optimal_eps_poisson


def main():
    fixed = np.load(DATA_DIR / "opt_eps_single.npz")
    T_values, d_values, R_values = (
        fixed["T_values"],
        fixed["d_values"],
        fixed["R_values"],
    )

    poisson_opt_eps = np.array(
        [
            optimal_eps_poisson(1 - 1 / T, R_values, fixed["eps_grid"], d_values - 1)
            for T in T_values
        ]
    )

    np.savez(
        DATA_DIR / "opt_eps_poisson_comparison.npz",
        difference=poisson_opt_eps - fixed["opt_eps"],
        T_values=T_values,
        d_values=d_values,
        R_values=R_values,
    )


if __name__ == "__main__":
    main()
