"""Threshold reward ratio R*(d, T) at which exploring first pays, and the exploration rate
the optimum jumps to there, over distances and horizons (Figure S2).

R* is the minimum over eps of L(eps) / beta(eps), the discounted time away from site 0
over the discounted time at site d, found on a grid of rates and refined with a bounded
one-dimensional search. The grid is log-spaced below 0.01, down to 1e-13, because at
small d and long horizons the minimizing rate is far below 0.01 (about 5e-5 at d = 2,
T = 1e9). The resolvent I - gamma P is assembled as (I - P) + P/T, so 1 - gamma = 1/T is
never formed by subtraction.

The thresholds are also computed at the paper's three horizons, which the grid of
horizons does not all contain, for the checks of Section S2.
"""

from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import splu
from tqdm import tqdm

from competitive_exploration.paths import DATA_DIR
from competitive_exploration.single_agent import state_sites, transition_matrix

D_VALUES = np.arange(1, 41)
# Twelve horizons per decade, so every decade is on the grid and the cells of the
# Figure S2 heatmap are roughly square.
T_VALUES = np.logspace(2, 9, 85)
T_PAPER = np.array([1000, 5000, 100000])
EPS_GRID = np.union1d(np.logspace(-13, -2, 111), np.linspace(0.01, 0.9995, 300))


def loss_gain_ratio(d, eps, T):
    """L / beta: discounted time away from site 0 over discounted time at site d."""
    P = transition_matrix(d, eps).tocsc()
    off_diagonal = P - diags(P.diagonal())
    generator = diags(np.asarray(off_diagonal.sum(axis=1)).ravel()) - off_diagonal  # I - P
    lu = splu(csc_matrix(generator + P / T))
    sites = state_sites(d)
    loss, gain = lu.solve(np.column_stack([sites != 0, sites == d]).astype(float))[0]
    return loss / gain if gain > 0 else np.inf


def threshold(d_and_T):
    """R* = min over eps of the loss/gain ratio, and the minimizing eps.

    At d = 1 the ratio is 1 for every rate, so the returned rate is arbitrary."""
    d, T = d_and_T
    ratios = np.array([loss_gain_ratio(d, eps, T) for eps in EPS_GRID])
    i = int(np.argmin(ratios))
    bounds = EPS_GRID[max(i - 1, 0)], EPS_GRID[min(i + 1, len(EPS_GRID) - 1)]
    result = minimize_scalar(
        lambda u: loss_gain_ratio(d, np.exp(u), T),
        bounds=np.log(bounds),
        method="bounded",
        options={"xatol": 1e-8},
    )
    if result.fun <= ratios[i]:
        return result.fun, np.exp(result.x)
    return ratios[i], EPS_GRID[i]


def sweep(T_values, pool):
    """(R*, eps*) for every horizon and distance, each of shape (T, d)."""
    jobs = [(d, T) for T in T_values for d in D_VALUES]
    out = list(tqdm(pool.imap(threshold, jobs, chunksize=4), total=len(jobs)))
    out = np.array(out).reshape(len(T_values), len(D_VALUES), 2)
    return out[..., 0], out[..., 1]


def main():
    with Pool() as pool:
        R_star, eps_star = sweep(T_VALUES, pool)
        R_star_paper, eps_star_paper = sweep(T_PAPER, pool)
    np.savez(
        DATA_DIR / "threshold_sweep.npz",
        R_star=R_star,
        eps_star=eps_star,
        T_values=T_VALUES,
        R_star_paper=R_star_paper,
        eps_star_paper=eps_star_paper,
        T_paper=T_PAPER,
        d_values=D_VALUES,
        eps_grid=EPS_GRID,
    )


if __name__ == "__main__":
    main()
