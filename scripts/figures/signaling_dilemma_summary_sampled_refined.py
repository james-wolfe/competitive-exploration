"""Figure 6: which strict dilemmas with both agents exploring at the worst Nash
equilibrium are removed by signaling, and by how much the worst Nash payoff rises.

For legibility the top row shows every seventh candidate reward ratio at each
distance; the bottom row shows the removed dilemmas among those."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from competitive_exploration.paths import DATA_DIR
from competitive_exploration.plotting import (
    PANEL_SIZE,
    place_panels,
    save_figure,
    use_paper_style,
)

use_paper_style(final_size=True)
# Bottom-row axis limits, by T, equal on both axes. Two removed dilemmas lie partly
# outside them (SI Section S8.2): limits fitted to every point would stretch the
# T = 100000 panel to 1.9e5-8.8e5 because of the dilemma at d = 3, R = 9.03.
LIMITS = {1000: (1834, 2191), 5000: (9211, 11445), 100000: (190217, 225800)}
# Tick and legend labels at the smallest PNAS size, and short ticks, to save space.
TICK = 6
plt.rcParams.update(
    {
        "xtick.labelsize": TICK,
        "ytick.labelsize": TICK,
        "legend.fontsize": TICK,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.pad": 2,
        "ytick.major.pad": 2,
    }
)
base = np.load(DATA_DIR / "sweep_nash_social_dilemma_fine401.npz")
signaling = np.load(DATA_DIR / "signaling_fine401.npz")
T_values, d_values, R_values = base["T_values"], base["d_values"], base["R_values"]

SOLVED = 0
STRIDE = 7
worst = signaling["worst_nash_payoff"]
improvement = 100 * (worst - base["worst_nash_payoff"]) / base["worst_nash_payoff"]

# Every STRIDE-th candidate R at each (gamma, d), among candidates at which a Nash
# equilibrium remains after signaling (SI Section S8.2).
shown = np.zeros_like(signaling["candidate"])
for t, a in np.ndindex(shown.shape[:2]):
    r = np.flatnonzero(signaling["candidate"][t, a] & np.isfinite(worst[t, a]))
    shown[t, a, r[::STRIDE]] = True
solved = shown & (signaling["status"] == SOLVED)
remains = shown & ~solved
norm = plt.Normalize(np.nanmin(improvement[shown]), np.nanmax(improvement[shown]))


# Square panels the size of those in Figure 4 (and Figure 5).
width, height, left, bottom, wspace, hspace = 5.73, 3.82, 0.36, 0.32, 0.28, 0.48
fig, axes = plt.subplots(2, len(T_values), figsize=(width, height))
place_panels(fig, axes, PANEL_SIZE, left, bottom, wspace, hspace)
for ax in axes[0, 1:]:
    ax.sharex(axes[0, 0])
    ax.sharey(axes[0, 0])
    ax.tick_params(labelleft=False)

for t, T in enumerate(T_values):
    ax = axes[0, t]
    for mask, marker in [(solved[t], "o"), (remains[t], "X")]:
        a, r = np.nonzero(mask)
        order = np.argsort(improvement[t, a, r])
        ax.scatter(
            R_values[r[order]],
            d_values[a[order]],
            c=improvement[t, a, r][order],
            cmap="viridis",
            norm=norm,
            marker=marker,
            s=11,
            edgecolor="black",
            linewidth=0.3,
            alpha=0.9,
            zorder=2,
        )
    ax.set_title(rf"$\gamma = 1 - 1/{T}$")
    ax.set_yticks([10, 20, 30])
    ax.grid(True)

    ax = axes[1, t]
    mask = solved[t] & np.isfinite(worst[t])
    order = np.argsort(improvement[t][mask])
    x = base["best_local_payoff"][t][mask][order]
    before = base["worst_nash_payoff"][t][mask][order]
    after = worst[t][mask][order]
    ax.vlines(x, before, after, color="0.65", linewidth=0.5, alpha=0.7, zorder=1)
    ax.scatter(
        x,
        before,
        marker="s",
        s=4,
        facecolor="white",
        edgecolor="0.35",
        linewidth=0.4,
        zorder=2,
    )
    ax.scatter(
        x,
        after,
        c=improvement[t][mask][order],
        cmap="viridis",
        norm=norm,
        s=18,
        edgecolor="black",
        linewidth=0.3,
        alpha=0.9,
        zorder=3,
    )
    limits = LIMITS[int(T)]
    ax.plot(limits, limits, "--", color="0.65", linewidth=0.7, zorder=0)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.ticklabel_format(style="sci", scilimits=(0, 0), useMathText=True)
    step = 10 ** np.floor(np.log10(limits[1])) / 10
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_major_locator(MultipleLocator(step))
    # The axes share one scale, so the power of ten is shown once, above the y-axis.
    ax.xaxis.get_offset_text().set_visible(False)
    ax.grid(True)

axes[0, 1].set_xlabel(r"Reward ratio $R$")
axes[1, 1].set_xlabel("Baseline social optimum")
axes[0, 0].set_ylabel(r"Distance $d$")
axes[1, 0].set_ylabel("Worst Nash payoff")


def marker(label, shape, face="0.75", edge="black", width=0.3, size=3):
    return Line2D(
        [],
        [],
        marker=shape,
        linestyle="none",
        markerfacecolor=face,
        markeredgecolor=edge,
        markeredgewidth=width,
        markersize=size,
        label=label,
    )


LEGEND_STYLE = dict(
    loc="upper left",
    handletextpad=0.2,
    borderpad=0.3,
    labelspacing=0.2,
    borderaxespad=0.3,
    framealpha=0.9,
)
axes[0, 0].legend(
    handles=[marker("dilemma resolved", "o"), marker("dilemma remains", "X")],
    **LEGEND_STYLE,
)
# In the last panel, whose upper left is empty; elsewhere it would cover data.
axes[1, 2].legend(
    handles=[
        marker("baseline", "s", face="white", edge="0.35", width=0.4, size=2.5),
        marker("with signaling", "o", size=4.2),
    ],
    **LEGEND_STYLE,
)

colorbar = fig.colorbar(
    plt.cm.ScalarMappable(norm, "viridis"),
    cax=fig.add_axes(
        [
            (left + 3 * PANEL_SIZE + 2 * wspace + 0.1) / width,
            (bottom + PANEL_SIZE + hspace) / height,
            0.07 / width,
            PANEL_SIZE / height,
        ]
    ),
)
colorbar.set_label("Worst Nash\nimprovement (\\%)", labelpad=4)

save_figure(fig, "signaling_dilemma_summary_sampled_refined.pdf")
