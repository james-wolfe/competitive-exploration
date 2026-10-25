"""Whether signaling removes each strict dilemma whose worst Nash equilibrium has both
agents exploring (Figure 6, Table S2, SI Section S8.2), on the grid of the "fine401"
baseline sweep (401 rates plus 1e-8, ..., 1e-2), using signaling_fast.

Every dilemma is evaluated on the grid of the baseline sweep; grid_size records that
grid's size, and coarse_worst_nash_payoff repeats worst_nash_payoff.

Usage: python scripts/generate/signaling_fine401.py [--workers N]
       [--base FILE --out FILE]   (defaults: the fine401 sweep and signaling_fine401.npz)
       [--checkpoint DIR]   (save each (gamma, d) when done; a rerun skips finished ones)

About 350 CPU-hours and 5 GB of memory per worker.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from competitive_exploration.equilibria import BOTH_EXPLORE
from competitive_exploration.paths import DATA_DIR
from competitive_exploration.signaling_fast import (
    combine,
    signaling_components,
    signaling_summary_fast,
)

SOLVED, NO_NASH, DILEMMA_REMAINS, NO_LOCAL = range(4)
D_MIN = 2


def status(nash_count, local_count, dilemma):
    if local_count == 0:
        return NO_LOCAL
    if nash_count == 0:
        return NO_NASH
    return DILEMMA_REMAINS if dilemma else SOLVED


def evaluate(args):
    """Signaling summaries at every candidate reward ratio of one (gamma, d). With a
    checkpoint directory, each (gamma, d) is saved when done and reused on a rerun."""
    T, d, eps_grid, R_list, checkpoint = args
    path = checkpoint and os.path.join(checkpoint, f"T{T}_d{d:02d}.npy")
    if path and os.path.exists(path):
        return [tuple(row) for row in np.load(path, allow_pickle=True)]
    components = signaling_components(d, eps_grid, 1 - 1 / T)
    rows = [signaling_summary_fast(*combine(components, 1.0, R)) for R in R_list]
    if path:
        np.save(path + ".tmp.npy", np.array(rows, dtype=object), allow_pickle=True)
        os.replace(path + ".tmp.npy", path)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--base", default=str(DATA_DIR / "sweep_nash_social_dilemma_fine401.npz"))
    parser.add_argument("--out", default=str(DATA_DIR / "signaling_fine401.npz"))
    parser.add_argument("--checkpoint", default="", help="directory for per-(gamma, d) results")
    args = parser.parse_args()

    base = np.load(args.base)
    eps_grid = base["eps_grid"]
    T_values, d_values, R_values = base["T_values"], base["d_values"], base["R_values"]
    candidate = base["strict_dilemma"] & (base["worst_nash_type"] == BOTH_EXPLORE)
    # Distances d = 2-30 only: at d = 1 every strict dilemma is a shortfall of at most
    # 0.022% in collective payoff between neighboring near-symmetric equilibria on the grid.
    candidate &= (d_values >= D_MIN)[None, :, None]

    shape = candidate.shape
    worst_nash = np.full(shape, np.nan)
    best_local = np.full(shape, np.nan)
    nash_count = np.zeros(shape, dtype=int)
    local_count = np.zeros(shape, dtype=int)
    dilemma = np.zeros(shape, dtype=bool)
    final_status = np.full(shape, -1)

    cells = [(t, a) for t, a in np.ndindex(shape[:2]) if candidate[t, a].any()]
    # Largest distances first, since they take longest.
    cells.sort(key=lambda c: -d_values[c[1]])
    if args.checkpoint:
        os.makedirs(args.checkpoint, exist_ok=True)
    jobs = [(int(T_values[t]), int(d_values[a]), eps_grid, R_values[candidate[t, a]], args.checkpoint)
            for t, a in cells]
    with ProcessPoolExecutor(args.workers) as pool:
        for (t, a), rows in zip(cells, pool.map(evaluate, jobs)):
            for r, row in zip(np.flatnonzero(candidate[t, a]), rows):
                worst_nash[t, a, r], best_local[t, a, r] = row[0], row[1]
                nash_count[t, a, r], local_count[t, a, r], dilemma[t, a, r] = row[2:]
                final_status[t, a, r] = status(*row[2:])
            print(f"T={T_values[t]} d={d_values[a]}: {len(rows)} dilemmas", flush=True)

    np.savez(
        args.out,
        candidate=candidate,
        status=final_status,
        grid_size=np.where(candidate, len(eps_grid), 0),
        worst_nash_payoff=worst_nash,
        best_local_payoff=best_local,
        coarse_worst_nash_payoff=worst_nash,
        nash_count=nash_count,
        local_count=local_count,
        dilemma=dilemma,
        status_codes=np.array(["solved", "no Nash", "dilemma remains", "no locally stable"]),
        eps_grid=eps_grid,
        T_values=T_values,
        d_values=d_values,
        R_values=R_values,
    )


if __name__ == "__main__":
    main()
