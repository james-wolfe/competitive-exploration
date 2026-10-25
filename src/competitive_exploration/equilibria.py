"""Nash equilibria, locally stable points, and social optima on a grid of rates.

Payoff matrices follow the convention of two_agent: Z[i, j] is the payoff to an agent
with rate index i against a competitor with rate index j. A strategy pair (i, j) lists
player 2's index first, so that it matches the rows and columns of Z.
"""

import numpy as np
from scipy.ndimage import maximum_filter

# Outcome types used to classify equilibria (Figures 4, 5, S7, S8).
BOTH_STAY, BOTH_EXPLORE, ANTI_COORDINATION, MULTIPLE, OTHER = range(5)


def best_response(Z):
    """Index of the best response to each competitor rate (first index if tied)."""
    return np.argmax(Z, axis=0)


def find_nash(Z):
    """Pure-strategy Nash equilibria: pairs of mutual best responses."""
    br = best_response(Z)
    points = [(int(br[j]), j) for j in range(Z.shape[1]) if br[br[j]] == j]
    mask = np.zeros(Z.shape, dtype=bool)
    for point in points:
        mask[point] = True
    return points, mask


def find_locally_stable(Z):
    """Pairs at which neither agent gains by moving its rate one grid step."""
    own_best = np.ones(Z.shape, dtype=bool)
    own_best[1:] &= Z[1:] >= Z[:-1]
    own_best[:-1] &= Z[:-1] >= Z[1:]
    mask = own_best & own_best.T
    return [tuple(map(int, p)) for p in np.argwhere(mask)], mask


def social_optimum(Z, among="locally stable"):
    """Pairs with the largest collective payoff, among locally stable pairs or all pairs."""
    W = Z + Z.T
    if among == "locally stable":
        _, candidates = find_locally_stable(Z)
    else:
        candidates = np.ones(Z.shape, dtype=bool)
    if not candidates.any():
        return [], candidates
    mask = candidates & (W == W[candidates].max())
    return [tuple(map(int, p)) for p in np.argwhere(mask)], mask


def outcome_type(i, j, diag_band=1):
    """Classify one strategy pair. Pairs within diag_band grid steps of the diagonal
    count as both exploring, which absorbs rounding in the location of the
    symmetric point."""
    if i == 0 and j == 0:
        return BOTH_STAY
    if i == 0 or j == 0:
        return ANTI_COORDINATION
    if abs(i - j) <= diag_band:
        return BOTH_EXPLORE
    return OTHER


def classify_points(points):
    """Split strategy pairs into (both explore, anti-coordination, both stay, other)."""
    groups = {BOTH_EXPLORE: [], ANTI_COORDINATION: [], BOTH_STAY: [], OTHER: []}
    for point in points:
        groups[outcome_type(*point)].append(point)
    return tuple(groups.values())


def outcome_code(points):
    """Type shared by a set of strategy pairs, MULTIPLE if they differ, OTHER if empty."""
    types = {outcome_type(*point) for point in points}
    if len(types) == 1:
        return types.pop()
    return MULTIPLE if types else OTHER


def nash_4d(U1, U2, rtol=1e-12, atol=1e-12):
    """Nash equilibria of the signaling game; arrays indexed as in signaling."""
    best1 = np.isclose(U1, U1.max(axis=(0, 1), keepdims=True), rtol=rtol, atol=atol)
    best2 = np.isclose(U2, U2.max(axis=(2, 3), keepdims=True), rtol=rtol, atol=atol)
    return best1 & best2


def locally_stable_4d(U1, U2, rtol=1e-12, atol=1e-12):
    """Locally stable points of the signaling game.

    Each agent's neighborhood is the 3 x 3 block of grid steps in its own two rates,
    including changes to both rates at once.
    """

    def local_max(U, size):
        neighborhood_max = maximum_filter(U, size=size, mode="constant", cval=-np.inf)
        return np.isclose(U, neighborhood_max, rtol=rtol, atol=atol)

    return local_max(U1, (3, 3, 1, 1)) & local_max(U2, (1, 1, 3, 3))


def signaling_summary(U1, U2, tol=1e-12):
    """Worst Nash and best locally stable collective payoffs of the signaling game,
    the numbers of each, and whether a strict dilemma remains."""
    W = U1 + U2
    nash = nash_4d(U1, U2, tol, tol)
    local = locally_stable_4d(U1, U2, tol, tol)
    worst_nash = W[nash].min() if nash.any() else np.nan
    best_local = W[local].max() if local.any() else np.nan
    dilemma = bool(
        nash.any()
        and local.any()
        and not np.isclose(worst_nash, best_local, rtol=tol, atol=tol)
    )
    return worst_nash, best_local, int(nash.sum()), int(local.sum()), dilemma
