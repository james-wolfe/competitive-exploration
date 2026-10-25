"""Rate at the positive local maximum of a lone agent's reward curve, on the grid of
the fine sweep (Figure S4)."""

import numpy as np
from tqdm import tqdm

from competitive_exploration.paths import DATA_DIR
from competitive_exploration.single_agent import site_occupancy_grid


def positive_local_max(payoff, eps_grid):
    """The one rate eps > 0 at a local maximum of payoff, or NaN if there is none."""
    is_max = np.ones(len(payoff), dtype=bool)
    is_max[1:] &= payoff[1:] >= payoff[:-1]
    is_max[:-1] &= payoff[:-1] >= payoff[1:]
    found = np.flatnonzero(is_max & (eps_grid > 0))
    if len(found) > 1:
        raise RuntimeError(f"several positive local maxima at {eps_grid[found]}")
    return eps_grid[found[0]] if len(found) else np.nan


def main():
    sweep = np.load(DATA_DIR / "sweep_nash_social_dilemma_fine.npz")
    eps_grid = sweep["eps_grid"]
    T_values, d_values, R_values = (
        sweep["T_values"],
        sweep["d_values"],
        sweep["R_values"],
    )

    local_max_eps = np.empty((len(T_values), len(d_values), len(R_values)))
    for t, T in enumerate(T_values):
        for a, d in enumerate(tqdm(d_values, desc=f"T={T}")):
            alpha, beta = site_occupancy_grid(d, eps_grid, 1 - 1 / T)
            for r, R in enumerate(R_values):
                local_max_eps[t, a, r] = positive_local_max(alpha + R * beta, eps_grid)

    np.savez(
        DATA_DIR / "local_max_eps_single.npz",
        local_max_eps=local_max_eps,
        T_values=T_values,
        d_values=d_values,
        R_values=R_values,
    )


if __name__ == "__main__":
    main()
