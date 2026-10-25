"""Single-agent reward curves, split by reward site (Figure 2A)."""

import numpy as np

from competitive_exploration.paths import DATA_DIR
from competitive_exploration.single_agent import site_occupancy_grid

T = 5000
D_VALUES = np.array([10, 5])  # rows of the figure
R_VALUES = np.array([2.5, 5.0])  # columns of the figure
EPS_GRID = np.linspace(0, 1, 201)


def main():
    gamma = 1 - 1 / T
    # rewards[a, r, e] = (reward from site 0, reward from site d)
    rewards = np.empty((len(D_VALUES), len(R_VALUES), len(EPS_GRID), 2))
    for a, d in enumerate(D_VALUES):
        alpha, beta = site_occupancy_grid(d, EPS_GRID, gamma)
        for r, R in enumerate(R_VALUES):
            rewards[a, r] = np.column_stack([alpha, R * beta])

    np.savez(
        DATA_DIR / "single_agent_rewards.npz",
        rewards=rewards,
        d_values=D_VALUES,
        R_values=R_VALUES,
        eps_grid=EPS_GRID,
        T=T,
    )


if __name__ == "__main__":
    main()
