"""A random distance to the distant reward, d ~ Poisson(lambda) + 1 (Section S1)."""

import numpy as np
from scipy.stats import poisson

from .single_agent import site_occupancy_grid


def shifted_poisson_weights(lambda_values, d_values):
    """P(d = a) for each lambda (rows) and a in d_values (columns).

    The weights are renormalized over d_values, so d_values should extend well
    beyond the largest mean distance.
    """
    weights = poisson.pmf(
        np.asarray(d_values)[None, :] - 1, np.asarray(lambda_values)[:, None]
    )
    return weights / weights.sum(axis=1, keepdims=True)


# Distances included in the expectation. P(d > 80) < 1e-13 for the means used here
# (lambda < 30).
D_SUPPORT = np.arange(1, 81)


def optimal_eps_poisson(
    gamma, reward_ratios, eps_grid, lambda_values, d_values=D_SUPPORT
):
    """Reward-maximizing rate in eps_grid, shape (len(lambda_values), len(reward_ratios))."""
    occupancy = np.array([site_occupancy_grid(d, eps_grid, gamma) for d in d_values])
    payoff = occupancy[:, 0, :, None] + occupancy[:, 1, :, None] * reward_ratios
    weights = shifted_poisson_weights(lambda_values, d_values)
    best = [np.argmax(np.tensordot(w, payoff, axes=1), axis=0) for w in weights]
    return eps_grid[np.array(best)]
