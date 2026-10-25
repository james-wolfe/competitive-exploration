"""Player 2's payoff over a fine grid of rate pairs, split by reward site (Figure 3)."""

import numpy as np

from competitive_exploration.paths import DATA_DIR
from competitive_exploration.two_agent import payoff_grid

d = 4
R = 2.1
T = 5000
EPS_GRID = np.linspace(0, 1, 1001)


def main():
    B0, Bd = payoff_grid(d, EPS_GRID, 1 - 1 / T)

    # Z[i, j]: player 2's payoff with eps2 = EPS_GRID[i] and eps1 = EPS_GRID[j].
    np.savez(
        DATA_DIR / "two_agent_payoffs.npz",
        Z_0=B0,
        Z_d=R * Bd,
        eps_grid=EPS_GRID,
        d=d,
        R=R,
        T=T,
    )


if __name__ == "__main__":
    main()
