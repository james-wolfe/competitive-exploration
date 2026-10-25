"""Payoffs, Nash equilibria, and local best-response basins in four regimes (Figures 4, S8)."""

import numpy as np

from competitive_exploration.equilibria import find_nash, social_optimum
from competitive_exploration.local_br import basin_map
from competitive_exploration.paths import DATA_DIR
from competitive_exploration.two_agent import payoff_grid

T = 5000
REGIMES = [(4, 2.1), (6, 4.0), (8, 3.0), (20, 3.0)]  # (d, R)
EPS_GRID = np.linspace(0, 1, 201)


def main():
    results = {key: [] for key in ["Z", "codes", "endpoints", "nash", "social_optimum"]}
    for d, R in REGIMES:
        B0, Bd = payoff_grid(d, EPS_GRID, 1 - 1 / T)
        Z = B0 + R * Bd
        codes, endpoints = basin_map(Z)
        results["Z"].append(Z)
        results["codes"].append(codes)
        results["endpoints"].append(endpoints)
        results["nash"].append(find_nash(Z)[1])
        results["social_optimum"].append(social_optimum(Z)[1])

    np.savez(
        DATA_DIR / "two_agent_local_br.npz",
        **{key: np.array(value) for key, value in results.items()},
        d_values=np.array([d for d, _ in REGIMES]),
        R_values=np.array([R for _, R in REGIMES]),
        eps_grid=EPS_GRID,
        T=T,
    )


if __name__ == "__main__":
    main()
