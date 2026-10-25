"""Which kinds of social dilemma occur, collective and Pareto, over a sweep of nine
discount factors (Table S1; Results, "Strict dilemmas usually involve a conflict of
interest").

For each (gamma, d, R), records six booleans: dilemmas of the first kind, second kind
and strict kind (Results, "Competition produces a social dilemma"), each judged by
collective payoff and by Pareto dominance.
"""

import numpy as np
from tqdm import tqdm

from competitive_exploration.equilibria import find_locally_stable, find_nash
from competitive_exploration.paths import DATA_DIR
from competitive_exploration.two_agent import payoff_grid

T_VALUES = np.array([100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000])
D_VALUES = np.arange(2, 31)
R_VALUES = np.linspace(2.1, 20.0, 200)
# Rates down to 1e-8 resolve optima close to zero.
EPS_GRID = np.union1d(np.linspace(0, 1, 41), np.logspace(-8, -2, 7))

KINDS = ["first", "first_pareto", "second", "second_pareto", "strict", "strict_pareto"]


def pareto_dominated(targets, candidates):
    """Whether some candidate payoff pair is strictly better for both agents than some
    target payoff pair."""
    return bool((candidates[:, None] > targets[None]).all(axis=2).any())


def dilemma_kinds(Z):
    W = Z + Z.T
    local_points, local = find_locally_stable(Z)
    nash_points, nash = find_nash(Z)

    # Each agent's payoff at every strategy pair (i, j).
    payoffs = np.stack([Z, Z.T], axis=-1)
    local_payoffs = payoffs[local]
    nash_payoffs = payoffs[nash]

    return [
        bool((W[local] < W.max()).any()),
        pareto_dominated(local_payoffs, payoffs.reshape(-1, 2)),
        bool(W[local].min() < W[local].max()),
        pareto_dominated(local_payoffs, local_payoffs),
        bool(nash_points and (W[nash] < W[local].max()).any()),
        pareto_dominated(nash_payoffs, local_payoffs),
    ]


def main():
    kinds = np.empty(
        (len(T_VALUES), len(D_VALUES), len(R_VALUES), len(KINDS)), dtype=bool
    )
    for t, T in enumerate(T_VALUES):
        for a, d in enumerate(tqdm(D_VALUES, desc=f"T={T}")):
            B0, Bd = payoff_grid(d, EPS_GRID, 1 - 1 / T)
            for r, R in enumerate(R_VALUES):
                kinds[t, a, r] = dilemma_kinds(B0 + R * Bd)

    np.savez(
        DATA_DIR / "dilemma_kinds.npz",
        **{kind: kinds[..., k] for k, kind in enumerate(KINDS)},
        T_values=T_VALUES,
        d_values=D_VALUES,
        R_values=R_VALUES,
        eps_grid=EPS_GRID,
    )


if __name__ == "__main__":
    main()
