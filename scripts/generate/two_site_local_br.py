"""Local best-response basins in the two-site model at d = 4, R = 3 (Figure S8)."""

import numpy as np

from competitive_exploration.equilibria import social_optimum
from competitive_exploration.local_br import basin_map
from competitive_exploration.paths import DATA_DIR
from competitive_exploration.two_site import payoff_matrix

d = 4
R = 3.0
T = 5000
EPS_GRID = np.linspace(0, 1, 201)


def main():
    Z = payoff_matrix(EPS_GRID, R, d, 1 - 1 / T)
    codes, endpoints = basin_map(Z)

    np.savez(
        DATA_DIR / "two_site_local_br.npz",
        Z=Z,
        codes=codes,
        endpoints=endpoints,
        social_optimum=social_optimum(Z)[1],
        eps_grid=EPS_GRID,
        d=d,
        R=R,
        T=T,
    )


if __name__ == "__main__":
    main()
