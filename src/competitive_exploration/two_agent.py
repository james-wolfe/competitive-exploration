"""Two agents sharing the lattice and splitting rewards (Model, "Two competing agents";
Materials and Methods).

Payoff matrices follow one convention throughout: Z[i, j] is the payoff to an agent
with exploration rate eps_grid[i] whose competitor has rate eps_grid[j]. Plotted with
rows on the vertical axis, Z is player 2's payoff over (eps1, eps2), and the
collective payoff of the pair is Z + Z.T.
"""

import numpy as np
from scipy.sparse import identity, kron
from scipy.sparse.linalg import splu

from .single_agent import state_sites, transition_matrix


def reward_shares(d):
    """Reward share of each agent at each joint state, for unit rewards.

    Joint state (s1, s2) has index s1 * n + s2, matching the Kronecker product. Returns
    an array of shape (n * n, 4) with columns: agent 1's share of the site-0 reward,
    agent 1's share of the site-d reward, and the same two for agent 2. An agent alone
    at a reward site receives all of it, and two agents at the same site split it.
    """
    x1, x2 = np.meshgrid(state_sites(d), state_sites(d), indexing="ij")

    def share(mine, other, site):
        return ((mine == site) * (1 - (other == site) / 2)).ravel()

    return np.column_stack(
        [share(x1, x2, 0), share(x1, x2, d), share(x2, x1, 0), share(x2, x1, d)]
    )


def two_agent_occupancy(d, eps1, eps2, gamma, shares=None):
    """Discounted reward each agent collects from each site, with R0 = Rd = 1.

    Returns a 2 x 2 array S, where S[k, 0] and S[k, 1] are what agent k + 1 collects
    from site 0 and site d. Agent k + 1's payoff is R0 * S[k, 0] + Rd * S[k, 1]
    (Materials and Methods, "Linearity of the payoff in the two reward values").
    """
    if shares is None:
        shares = reward_shares(d)
    K = kron(transition_matrix(d, eps1), transition_matrix(d, eps2), format="csc")
    lu = splu(identity(K.shape[0], format="csc") - gamma * K)
    return lu.solve(shares)[0].reshape(2, 2)


def payoff_grid(d, eps_grid, gamma):
    """Payoff matrices per unit reward over a grid of exploration-rate pairs.

    Returns (B0, Bd) such that the payoff matrix for rewards R0 and Rd is
    Z = R0 * B0 + Rd * Bd. Each solve gives both agents' values, so only pairs with
    i <= j are solved.
    """
    m = len(eps_grid)
    shares = reward_shares(d)
    B0 = np.empty((m, m))
    Bd = np.empty((m, m))
    for i in range(m):
        for j in range(i, m):
            S = two_agent_occupancy(d, eps_grid[i], eps_grid[j], gamma, shares)
            B0[i, j], Bd[i, j] = S[0]
            B0[j, i], Bd[j, i] = S[1]
    return B0, Bd
