"""Local best-response dynamics (Results, "Locally stable equilibria").

In each round both agents simultaneously move to whichever of their current rate and
its two grid neighbors pays best against the competitor's current rate (the lower
rate if tied). Strategy pairs and payoff matrices follow the convention of two_agent.
"""

import numpy as np

from .equilibria import OTHER, outcome_type


def local_best_move(Z):
    """next_rate[k, l]: an agent's next rate index from rate k against competitor l."""
    shifted = np.full((3,) + Z.shape, -np.inf)
    shifted[0, 1:] = Z[:-1]
    shifted[1] = Z
    shifted[2, :-1] = Z[1:]
    return np.arange(Z.shape[0])[:, None] + np.argmax(shifted, axis=0) - 1


def local_br_endpoints(Z):
    """Where the dynamics end from every starting pair.

    Returns (end_i, end_j, fixed): arrays of shape Z.shape giving the pair reached
    from each starting pair (i, j), and whether it is a fixed point. A few
    trajectories end in a cycle instead; for those the endpoint is the first pair of
    the cycle that the trajectory reaches.
    """
    move = local_best_move(Z)

    def step(i, j):
        return move[i, j], move[j, i]

    # A pair is on a cycle (or is a fixed point) if the dynamics return to it.
    rounds = 4 * Z.shape[0]
    i0, j0 = np.indices(Z.shape)
    on_cycle = np.zeros(Z.shape, dtype=bool)
    i, j = i0, j0
    for _ in range(rounds):
        i, j = step(i, j)
        on_cycle |= (i == i0) & (j == j0)

    i, j = i0, j0
    for _ in range(rounds):
        moving = ~on_cycle[i, j]
        if not moving.any():
            break
        i_next, j_next = step(i, j)
        i, j = np.where(moving, i_next, i), np.where(moving, j_next, j)
    if not on_cycle[i, j].all():
        raise RuntimeError("some trajectories did not reach a cycle or fixed point")

    i_next, j_next = step(i, j)
    return i, j, (i_next == i) & (j_next == j)


def basin_map(Z):
    """Outcome type reached from each starting pair (OTHER for cycles), and a mask of
    the fixed points that are reached."""
    end_i, end_j, fixed = local_br_endpoints(Z)
    codes = np.where(fixed, np.vectorize(outcome_type)(end_i, end_j), OTHER)
    endpoints = np.zeros(Z.shape, dtype=bool)
    endpoints[end_i[fixed], end_j[fixed]] = True
    return codes, endpoints
