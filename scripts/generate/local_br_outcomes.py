"""Collective payoff reached by the local best-response dynamics, averaged over all
starting pairs on the grid of the baseline sweep (Figure S6). Trajectories that end
in a cycle count the first pair of the cycle they reach."""

import numpy as np
from tqdm import tqdm

from competitive_exploration.local_br import local_br_endpoints
from competitive_exploration.paths import DATA_DIR


def main():
    sweep = np.load(DATA_DIR / "sweep_nash_social_dilemma.npz")
    Z_all = sweep["Z"]

    mean_payoff = np.empty(Z_all.shape[:3])
    cycle_fraction = np.empty(Z_all.shape[:3])
    for index in tqdm(list(np.ndindex(mean_payoff.shape))):
        Z = Z_all[index]
        end_i, end_j, fixed = local_br_endpoints(Z)
        mean_payoff[index] = (Z + Z.T)[end_i, end_j].mean()
        cycle_fraction[index] = 1 - fixed.mean()

    np.savez(
        DATA_DIR / "local_br_outcomes.npz",
        mean_payoff=mean_payoff,
        cycle_fraction=cycle_fraction,
        global_optimum_payoff=sweep["global_optimum_payoff"],
        T_values=sweep["T_values"],
        d_values=sweep["d_values"],
        R_values=sweep["R_values"],
    )


if __name__ == "__main__":
    main()
