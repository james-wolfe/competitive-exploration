"""Probability and time for a walker starting at site 1, before it has found site d,
to reach site d before site 0: gambler's-ruin formulas and simulation (Figure S9)."""

import numpy as np
from scipy.linalg import solve_banded

from competitive_exploration.paths import DATA_DIR
from competitive_exploration.single_agent import state_sites, transition_matrix
from competitive_exploration.two_site import hit_probability, mean_hitting_time

d = 4
N_PATHS = 10_000_000
EPS_GRID = np.linspace(0, 1, 51)


def exact_stats(eps):
    """P(hit d first), E[T], E[T | hit d first], E[T | hit 0 first]."""
    p = eps / 2
    prob_d = hit_probability(p, d)
    mean_T = mean_hitting_time(p, d)
    if p == 0:
        return prob_d, mean_T, np.inf, mean_T

    # h[k] = E[T * 1{hit d first} | start at k] solves h = u + p h[k+1] + q h[k-1],
    # where u[k] is the probability of hitting d first from k.
    q = 1 - p
    k = np.arange(1, d)
    u = k / d if np.isclose(p, q) else (1 - (q / p) ** k) / (1 - (q / p) ** d)
    bands = np.zeros((3, d - 1))
    bands[0, 1:] = -p
    bands[1] = 1
    bands[2, :-1] = -q
    h1 = solve_banded((1, 1), bands, u)[0]
    return prob_d, mean_T, h1 / prob_d, (mean_T - h1) / (1 - prob_d)


def simulate_stats(eps, seed):
    """Monte Carlo estimates of exact_stats from N_PATHS walks of the lattice chain."""
    P = transition_matrix(d, eps)
    sites = state_sites(d)
    # Every state has at most two successors; store the first with its probability.
    first = np.array([P.indices[P.indptr[s]] for s in range(P.shape[0])])
    last = np.array([P.indices[P.indptr[s + 1] - 1] for s in range(P.shape[0])])
    p_first = np.array([P.data[P.indptr[s]] for s in range(P.shape[0])])

    rng = np.random.default_rng(seed)
    state = np.full(N_PATHS, 1)  # site 1, site d not yet found
    time = np.zeros(N_PATHS, dtype=int)
    active = np.ones(N_PATHS, dtype=bool)
    while active.any():
        paths = np.flatnonzero(active)
        current = state[paths]
        take_first = rng.random(paths.size) < p_first[current]
        state[paths] = np.where(take_first, first[current], last[current])
        time[paths] += 1
        active[paths] = (sites[state[paths]] != 0) & (sites[state[paths]] != d)

    hit_d = sites[state] == d
    return (
        hit_d.mean(),
        time.mean(),
        time[hit_d].mean() if hit_d.any() else np.nan,
        time[~hit_d].mean() if (~hit_d).any() else np.nan,
    )


def main():
    exact = np.array([exact_stats(eps) for eps in EPS_GRID])
    simulated = np.array(
        [simulate_stats(eps, seed=1000 * i) for i, eps in enumerate(EPS_GRID)]
    )

    np.savez(
        DATA_DIR / "hitting_stats.npz",
        exact=exact,
        simulated=simulated,
        columns=np.array(
            ["P(hit d first)", "E[T]", "E[T | d first]", "E[T | 0 first]"]
        ),
        eps_grid=EPS_GRID,
        d=d,
        n_paths=N_PATHS,
    )


if __name__ == "__main__":
    main()
