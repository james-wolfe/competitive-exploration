"""A lone agent on the lattice {0, ..., d} (Model, "Markov chain formulation"; Materials
and Methods)."""

import numpy as np
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import splu


def transition_matrix(d, eps):
    """Transition matrix of the two-phase biased random walk.

    States are pairs (x, B): the agent's site x and whether it has visited site d
    (B = 1) or not (B = 0). State (x, B) has index B * (d + 1) + x, so the start state
    (0, 0) has index 0. Reaching site d sets B = 1 on arrival. State (d, 0) is never
    reached and is kept only so that the matrix has the size 2(d + 1) used in the text.

    The rules are those of the high-discount regime (gamma^d R > 1, or gamma^d R / 2 > 1
    with two agents), in which an agent that has found site d exploits toward it from
    every site. All results in the paper are in this regime.
    """
    n = d + 1
    rows, cols, probs = [], [], []

    def add(x, B, x_next, p):
        if p > 0:
            rows.append(B * n + x)
            cols.append((B or x_next == d) * n + x_next)
            probs.append(p)

    for x in range(n):
        # Before discovery the agent exploits toward site 0.
        if x == 0:
            add(x, 0, 0, 1 - eps)
            add(x, 0, 1, eps)
        elif x < d:
            add(x, 0, x - 1, 1 - eps / 2)
            add(x, 0, x + 1, eps / 2)
        else:
            add(x, 0, d, 1.0)

        # After discovery the agent exploits toward site d.
        if x == 0:
            add(x, 1, 1, 1.0)
        elif x < d:
            add(x, 1, x - 1, eps / 2)
            add(x, 1, x + 1, 1 - eps / 2)
        else:
            add(x, 1, d, 1 - eps)
            add(x, 1, d - 1, eps)

    return csr_matrix((probs, (rows, cols)), shape=(2 * n, 2 * n))


def state_sites(d):
    """Site x of each state, in the order used by transition_matrix."""
    return np.tile(np.arange(d + 1), 2)


def site_occupancy(d, eps, gamma):
    """Discounted time a lone agent spends at site 0 and at site d.

    These are alpha-hat and beta-hat of Section S4: the agent's payoff is
    R0 * alpha + Rd * beta, so one linear solve gives the payoff for every pair of
    reward values (Materials and Methods, "Linearity of the payoff in the two reward
    values").
    """
    P = transition_matrix(d, eps)
    sites = state_sites(d)
    indicators = np.column_stack([sites == 0, sites == d]).astype(float)
    lu = splu((identity(P.shape[0], format="csc") - gamma * P).tocsc())
    alpha, beta = lu.solve(indicators)[0]
    return float(alpha), float(beta)


def site_occupancy_grid(d, eps_grid, gamma):
    """site_occupancy for each rate in eps_grid, returned as two arrays."""
    alpha, beta = np.array([site_occupancy(d, eps, gamma) for eps in eps_grid]).T
    return alpha, beta


def finite_horizon_reward(d, eps, T, R0=1.0, Rd=1.0):
    """Expected undiscounted reward over time steps 0, ..., T (Model, "Markov chain
    formulation")."""
    P = transition_matrix(d, eps)
    sites = state_sites(d)
    reward = R0 * (sites == 0) + Rd * (sites == d)

    p = np.zeros(P.shape[0])
    p[0] = 1.0
    total = p @ reward
    for _ in range(T):
        p = P.T @ p
        total += p @ reward
    return float(total)
