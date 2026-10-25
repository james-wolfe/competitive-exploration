"""Figure S7: type of the strategy pair that maximizes the collective payoff."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from competitive_exploration.paths import DATA_DIR
from competitive_exploration.plotting import (
    OUTCOME_CMAP,
    OUTCOME_LABELS,
    OUTCOME_NORM,
    centers_to_edges,
    save_figure,
    use_paper_style,
)

use_paper_style()
data = np.load(DATA_DIR / "sweep_nash_social_dilemma_fine401.npz")
optimum_type = data["global_optimum_type"]

fig, axes = plt.subplots(
    1,
    len(data["T_values"]),
    figsize=(9.75, 3.5),
    sharex=True,
    sharey=True,
    layout="constrained",
)
for t, T in enumerate(data["T_values"]):
    axes[t].pcolormesh(
        centers_to_edges(data["R_values"]),
        centers_to_edges(data["d_values"]),
        optimum_type[t],
        cmap=OUTCOME_CMAP,
        norm=OUTCOME_NORM,
        alpha=0.95,
        antialiased=False,
        rasterized=True,
    )
    axes[t].set_title(rf"$\gamma = 1 - 1/{T}$")
    axes[t].set_xlabel(r"Reward ratio $R$")
    axes[t].set_xticks([5, 10, 15, 20])
    axes[t].set_yticks([5, 10, 15, 20, 25, 30])
axes[0].set_ylabel(r"Distance $d$")

axes[-1].legend(
    handles=[
        Patch(color=OUTCOME_CMAP(code), label=OUTCOME_LABELS[code])
        for code in np.unique(optimum_type)
    ],
    loc="center right",
)

save_figure(fig, "global_social_optimum_classification.pdf")
