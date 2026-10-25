"""The two-site model without explicit spatial structure (Section S7).

Each agent is in one of three states: U (has not found the high reward, at site 1),
H (at site 2, the high reward) or L (has found the high reward, back at site 1).
Distance enters only through gambler's-ruin hitting probabilities and times.
"""

import numpy as np

U, H, L = range(3)
SITE = np.array([1, 2, 1])  # site occupied in each state


def hit_probability(p, d):
    """Probability that a walk from site 1 that steps right with probability p
    reaches site d before site 0."""
    q = 1 - p
    if p == 0:
        return 0.0
    if q == 0:
        return 1.0
    if np.isclose(p, q, rtol=0, atol=1e-14):
        return 1 / d
    r = q / p
    return (r - 1) / (r**d - 1)


def mean_hitting_time(p, d):
    """Expected steps for the same walk to reach site 0 or site d."""
    q = 1 - p
    if p == 0:
        return 1.0
    if q == 0 or np.isclose(p, q, rtol=0, atol=1e-14):
        return float(d - 1)
    return (1 - d * hit_probability(p, d)) / (q - p)


def transition_matrix(eps, d):
    """3 x 3 transition matrix over (U, H, L).

    From U and H an agent attempts to switch with probability eps, and from L it
    always attempts to return to H. Attempts succeed with the hitting probability of a
    walk biased away from the target (U to H, H to L) or toward it (L to H).
    """
    away = hit_probability(eps / 2, d)
    toward = hit_probability(1 - eps / 2, d)
    return np.array(
        [
            [1 - eps * away, eps * away, 0.0],
            [0.0, 1 - eps * away, eps * away],
            [0.0, toward, 1 - toward],
        ]
    )


def joint_chain(eps1, eps2, R2, d):
    """Joint transition matrix and both agents' expected one-step rewards.

    The reward depends on the sites after a transition (site 1 pays 1, site 2 pays
    R2, split when both agents are at the same site), minus the expected cost of an
    attempt: its mean hitting time, paid with probability eps in U and H and always
    in L.
    """
    P1, P2 = transition_matrix(eps1, d), transition_matrix(eps2, d)
    T = np.kron(P1, P2)

    site1, site2 = np.meshgrid(SITE, SITE, indexing="ij")
    split = np.where(site1 == site2, 2, 1)
    r1 = T @ (np.where(site1 == 1, 1.0, R2) / split).ravel()
    r2 = T @ (np.where(site2 == 1, 1.0, R2) / split).ravel()

    def attempt_cost(eps):
        return np.array(
            [
                eps * mean_hitting_time(eps / 2, d),
                eps * mean_hitting_time(eps / 2, d),
                mean_hitting_time(1 - eps / 2, d),
            ]
        )

    r1 -= np.repeat(attempt_cost(eps1), 3)
    r2 -= np.tile(attempt_cost(eps2), 3)
    return T, r1, r2


def payoff_matrix(eps_grid, R2, d, gamma):
    """Payoff matrix Z[i, j] of an agent with rate eps_grid[i] against eps_grid[j]."""
    m = len(eps_grid)
    Z = np.empty((m, m))
    for i, eps_i in enumerate(eps_grid):
        for j, eps_j in enumerate(eps_grid):
            T, r_i, _ = joint_chain(eps_i, eps_j, R2, d)
            Z[i, j] = np.linalg.solve(np.eye(9) - gamma * T, r_i)[0]
    return Z
