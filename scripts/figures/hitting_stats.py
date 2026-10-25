"""Figure S9: probability and time for a walker to reach site d before returning to
site 0, from the gambler's-ruin formulas (curves) and simulation (points)."""

import matplotlib.pyplot as plt
import numpy as np

from competitive_exploration.paths import DATA_DIR
from competitive_exploration.plotting import save_figure, use_paper_style

use_paper_style()
data = np.load(DATA_DIR / "hitting_stats.npz")
eps_grid, exact, simulated = data["eps_grid"], data["exact"], data["simulated"]

fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")

left.plot(eps_grid, exact[:, 0], color="black", label="Expectation")
left.scatter(
    eps_grid, simulated[:, 0], color="black", s=18, zorder=3, label="Simulation"
)
left.set_title(r"$\mathrm{P}(\mathrm{hit}\ d\ \mathrm{before}\ 0)$")
left.set_ylabel("Probability")

# Columns 1-3: unconditional time, time given d first, time given 0 first.
for column, color, label in [
    (1, "C0", r"$\tau$"),
    (3, "C1", r"$\tau_0$"),
    (2, "C2", r"$\tau_d$"),
]:
    right.plot(eps_grid, exact[:, column], color=color, label=label)
    right.scatter(eps_grid, simulated[:, column], color=color, s=18, zorder=3)
right.set_title("Hitting times")
right.set_ylabel("Time (steps)")

for ax in (left, right):
    ax.set_xlabel(r"Exploration rate $\varepsilon$")
    ax.legend()
    ax.grid(True)

save_figure(fig, "hitting_stats.pdf")
