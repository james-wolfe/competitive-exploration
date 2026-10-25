"""Equilibria, social optima, and strict social dilemmas over the (gamma, d, R) sweep.

The default variant (41 exploration rates) stores the payoff matrices, which
local_br_outcomes.py (Figure S6) reads. Three variants do not store them: "fine401" uses
401 exploration rates plus the seven rates 1e-8, ..., 1e-2 at every distance and feeds
Figures 5, 6 and S7 and Tables 2 and S2; "fine" uses 101 exploration rates, and "small_d"
adds rates down to 1e-8 for the shortest distances, where the socially optimal rates are
smallest; these two feed Figures S4 and S5.

The 41-rate grid misses the strict dilemmas at distances d = 2 and 3, and 101 evenly
spaced rates still miss nearly all of them at d = 2; adding the seven small rates
recovers them. "fine401" takes about 6 CPU-hours; --workers N runs the (gamma, d) cells
in parallel.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm import tqdm

from competitive_exploration.equilibria import (
    classify_points,
    find_locally_stable,
    find_nash,
    outcome_code,
    social_optimum,
)
from competitive_exploration.paths import DATA_DIR
from competitive_exploration.two_agent import payoff_grid

T_VALUES = np.array([1000, 5000, 100000])
R_VALUES = np.linspace(2.1, 20, 200)
TOL = 1e-12

# variant: (exploration-rate grid, distances)
VARIANTS = {
    "": (np.linspace(0, 1, 41), np.arange(1, 31)),
    "fine": (np.linspace(0, 1, 101), np.arange(1, 31)),
    "small_d": (
        np.union1d(np.linspace(0, 1, 101), np.logspace(-8, -2, 7)),
        np.arange(1, 6),
    ),
    "fine401": (
        np.union1d(np.linspace(0, 1, 401), np.logspace(-8, -2, 7)),
        np.arange(1, 31),
    ),
}


def analyze(Z, eps_grid):
    """Summarize the equilibria of one payoff matrix."""
    W = Z + Z.T
    nash_points, nash = find_nash(Z)
    local_points, local = find_locally_stable(Z)
    optimal_points, optimal = social_optimum(Z)
    global_points, _ = social_optimum(Z, among="all")
    symmetric, anti_coordination, _, _ = classify_points(local_points)

    out = {
        "nash_type": outcome_code(nash_points),
        "social_optimum_type": outcome_code(optimal_points),
        "global_optimum_type": outcome_code(global_points),
        "worst_nash_payoff": W[nash].min() if nash_points else np.nan,
        "worst_local_payoff": W[local].min() if local_points else np.nan,
        "best_local_payoff": W[local].max() if local_points else np.nan,
        "global_optimum_payoff": W.max(),
        "symmetric_eps": np.mean([eps_grid[[i, j]].mean() for i, j in symmetric])
        if symmetric
        else np.nan,
        "worst_nash_type": -1,
        "strict_dilemma": False,
        "pareto_dilemma": False,
    }

    # Table 2: is each locally stable point Nash (row) and socially optimal (column)?
    for name, points in [
        ("symmetric", symmetric),
        ("anti_coordination", anti_coordination),
    ]:
        counts = np.zeros((2, 2), dtype=int)
        for point in points:
            counts[int(not nash[point]), int(not optimal[point])] += 1
        out[f"{name}_counts"] = counts

    # Strict dilemma: the worst Nash equilibrium is collectively worse than the best
    # locally stable point. It is a Pareto dilemma if some locally stable point is
    # strictly better for both agents than some Nash equilibrium.
    if (
        nash_points
        and local_points
        and not np.isclose(
            out["worst_nash_payoff"], out["best_local_payoff"], rtol=TOL, atol=TOL
        )
    ):
        worst = np.argwhere(
            nash & np.isclose(W, out["worst_nash_payoff"], rtol=TOL, atol=TOL)
        )
        out["strict_dilemma"] = True
        out["worst_nash_type"] = outcome_code(map(tuple, worst))
        for i, j in np.argwhere(nash):
            both = (Z > Z[i, j] + TOL) & (Z.T > Z[j, i] + TOL)
            out["pareto_dilemma"] |= bool((local & both).any())
    return out


def analyze_cell(args):
    """Payoff matrices and summaries at every reward ratio for one (gamma, d)."""
    T, d, eps_grid = args
    B0, Bd = payoff_grid(d, eps_grid, 1 - 1 / T)
    return B0, Bd, [analyze(B0 + R * Bd, eps_grid) for R in R_VALUES]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", nargs="?", default="", choices=VARIANTS)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    variant = args.variant
    eps_grid, d_values = VARIANTS[variant]

    shape = (len(T_VALUES), len(d_values), len(R_VALUES))
    results = {}
    Z_all = np.empty(shape + (len(eps_grid),) * 2) if variant == "" else None

    cells = [(T, d, eps_grid) for T in T_VALUES for d in d_values]
    with ProcessPoolExecutor(args.workers) as pool:
        outputs = pool.map(analyze_cell, cells)
        for k, (B0, Bd, rows) in enumerate(tqdm(outputs, total=len(cells))):
            t, a = divmod(k, len(d_values))
            for r, R in enumerate(R_VALUES):
                if Z_all is not None:
                    Z_all[t, a, r] = B0 + R * Bd
                for key, value in rows[r].items():
                    if key not in results:
                        results[key] = np.empty(
                            shape + np.shape(value), np.asarray(value).dtype
                        )
                    results[key][t, a, r] = value

    if Z_all is not None:
        results["Z"] = Z_all
    np.savez(
        DATA_DIR / f"sweep_nash_social_dilemma{'_' + variant if variant else ''}.npz",
        **results,
        eps_grid=eps_grid,
        T_values=T_VALUES,
        d_values=d_values,
        R_values=R_VALUES,
    )


if __name__ == "__main__":
    main()
