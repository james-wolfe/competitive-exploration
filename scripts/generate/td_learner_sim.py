"""Simulated TD learners against the Markov chain expectation (Figure S1)."""

import numpy as np

from competitive_exploration.paths import DATA_DIR
from competitive_exploration.single_agent import finite_horizon_reward
from competitive_exploration.td_learner import simulate_rewards

d = 8
R = 5.0
T = 5000
N_RUNS = 5000
SIM_EPS = np.arange(0, 1.01, 0.05)
CHAIN_EPS = np.linspace(0, 1, 101)


def main():
    mean, standard_error = simulate_rewards(
        d, T, SIM_EPS, N_RUNS, gamma=1 - 1 / T, Rd=R, seed=1
    )
    expected = [finite_horizon_reward(d, eps, T, Rd=R) for eps in CHAIN_EPS]

    np.savez(
        DATA_DIR / "td_learner_sim.npz",
        sim_eps=SIM_EPS,
        mean_reward=mean,
        standard_error=standard_error,
        chain_eps=CHAIN_EPS,
        expected_reward=expected,
        d=d,
        R=R,
        T=T,
        n_runs=N_RUNS,
    )


if __name__ == "__main__":
    main()
