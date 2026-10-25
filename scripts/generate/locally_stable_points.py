"""Locally stable points with and without signaling at d = 4, R = 2.1 (Table 3)."""

import numpy as np

from competitive_exploration.equilibria import (
    find_locally_stable,
    find_nash,
    locally_stable_4d,
    nash_4d,
)
from competitive_exploration.paths import DATA_DIR
from competitive_exploration.signaling import signaling_payoff_grid

d = 4
R = 2.1
T = 5000
EPS_GRID = np.linspace(0, 1, 51)


def main():
    U1, U2 = signaling_payoff_grid(d, EPS_GRID, 1 - 1 / T, Rd=R, progress=True)
    W = U1 + U2
    nash = nash_4d(U1, U2)
    local = locally_stable_4d(U1, U2)

    # Signaling: one row per set of equivalent locally stable points. An agent whose
    # competitor never explores never receives a signal, so its post-signal rate is
    # irrelevant; such points are listed with that rate equal to the initial rate.
    groups = {}
    for i1, s1, i2, s2 in np.argwhere(local):
        key = (i1, i1 if i2 == 0 else s1, i2, i2 if i1 == 0 else s2)
        groups.setdefault(key, []).append((i1, s1, i2, s2))
    keys = list(groups)

    # The baseline game is the signaling game restricted to strategies that never
    # switch. Its symmetric Nash equilibrium is also evaluated under signaling.
    k = np.arange(len(EPS_GRID))
    Z = U1[k[:, None], k[:, None], k[None, :], k[None, :]]
    baseline_points, _ = find_locally_stable(Z)
    _, baseline_nash = find_nash(Z)
    for i, j in find_nash(Z)[0]:
        if i == j and (i, i, i, i) not in groups:
            keys.append((i, i, i, i))
            groups[(i, i, i, i)] = [(i, i, i, i)]

    np.savez(
        DATA_DIR / "locally_stable_points.npz",
        baseline_eps=EPS_GRID[np.array(baseline_points)],
        baseline_payoff=[Z[i, j] + Z[j, i] for i, j in baseline_points],
        baseline_nash=[baseline_nash[p] for p in baseline_points],
        signaling_eps=EPS_GRID[np.array(keys)],
        signaling_payoff=[max(W[p] for p in groups[key]) for key in keys],
        signaling_nash=[any(nash[p] for p in groups[key]) for key in keys],
        signaling_locally_stable=[any(local[p] for p in groups[key]) for key in keys],
        d=d,
        R=R,
        T=T,
        eps_grid=EPS_GRID,
    )


if __name__ == "__main__":
    main()
