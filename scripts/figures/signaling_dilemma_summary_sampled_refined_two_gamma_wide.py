"""Figure 6: which strict dilemmas with both agents exploring at the worst Nash
equilibrium are removed by signaling, and by how much the worst Nash payoff rises, for
two discount factors. SI Appendix shows the shortest horizon as well.

For legibility the top row shows every seventh candidate reward ratio at each
distance; the bottom row shows the removed dilemmas among those."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from competitive_exploration.paths import DATA_DIR
from competitive_exploration.plotting import (
    PNAS_WIDTHS,
    TWO_GAMMA_PANEL_SIZE,
    place_panels,
    save_figure,
    use_paper_style,
)

use_paper_style(final_size=True)
# Bottom-row axis limits, by T, equal on both axes. Two removed dilemmas lie partly
# outside them (SI Section S10.2): limits fitted to every point would stretch the
# T = 100000 panel to 1.9e5-8.8e5 because of the dilemma at d = 3, R = 9.03.
LIMITS = {5000: (9211, 11445), 100000: (190217, 225800)}
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
t_keep = [list(base["T_values"]).index(T) for T in LIMITS]
T_values, d_values, R_values = base["T_values"][t_keep], base["d_values"], base["R_values"]
base = {key: base[key][t_keep] for key in ["worst_nash_payoff", "best_local_payoff"]}
signaling = {key: signaling[key][t_keep] for key in ["worst_nash_payoff", "candidate", "status"]}

SOLVED = 0
STRIDE = 7
worst = signaling["worst_nash_payoff"]
improvement = 100 * (worst - base["worst_nash_payoff"]) / base["worst_nash_payoff"]

# Every STRIDE-th candidate R at each (gamma, d), among candidates at which a Nash
# equilibrium remains after signaling (SI Section S10.2).
shown = np.zeros_like(signaling["candidate"])
for t, a in np.ndindex(shown.shape[:2]):
    r = np.flatnonzero(signaling["candidate"][t, a] & np.isfinite(worst[t, a]))
    shown[t, a, r[::STRIDE]] = True
solved = shown & (signaling["status"] == SOLVED)
remains = shown & ~solved
norm = plt.Normalize(np.nanmin(improvement[shown]), np.nanmax(improvement[shown]))


# Square panels, so that the diagonal of the bottom row lies at 45 degrees, placed as in
# Figure 5, with the colorbar on the right, at the width of a 3/4-page figure.
panel = TWO_GAMMA_PANEL_SIZE
width, left, bottom, top = PNAS_WIDTHS["one_and_half"], 0.58, 0.32, 0.2
wspace, hspace = 0.3, 0.48
height = bottom + 2 * panel + hspace + top
fig, axes = plt.subplots(2, len(T_values), figsize=(width, height))
place_panels(fig, axes, panel, left, bottom, wspace, hspace)
for ax in axes[0, 1:]:
    ax.sharex(axes[0, 0])
    ax.sharey(axes[0, 0])
    ax.tick_params(labelleft=False)

for t, T in enumerate(T_values):
    ax = axes[0, t]
    for mask, marker, size in [(solved[t], "o", 11), (remains[t], "X", 8)]:
        a, r = np.nonzero(mask)
        order = np.argsort(improvement[t, a, r])
        ax.scatter(
            R_values[r[order]],
            d_values[a[order]],
            c=improvement[t, a, r][order],
            cmap="viridis",
            norm=norm,
            marker=marker,
            s=size,
            edgecolor="black",
            linewidth=0.3,
            alpha=0.9,
            zorder=2,
        )
    ax.set_title(rf"$\gamma = 1 - 1/{T}$")
    ax.set_xlabel(r"Reward ratio $R$")
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

fig.text(
    (left + panel + wspace / 2) / width,
    0.02 / height,
    "Social optimum without signaling",
    ha="center",
    va="bottom",
)
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
    handlelength=1,
    handletextpad=0.2,
    borderpad=0.3,
    labelspacing=0.2,
    borderaxespad=0.3,
    framealpha=0.9,
)
axes[0, 0].legend(
    handles=[marker("dilemma resolved", "o"), marker("dilemma remains", "X", size=2.6)],
    **LEGEND_STYLE,
)
# In the last panel, whose upper left is empty; elsewhere it would cover data.
axes[1, -1].legend(
    handles=[
        marker("without signaling", "s", face="white", edge="0.35", width=0.4, size=2.5),
        marker("with signaling", "o", size=4.2),
    ],
    **LEGEND_STYLE,
)

# Centered on the two rows, over most of their height.
bar_height = 0.7 * (2 * panel + hspace)
colorbar = fig.colorbar(
    plt.cm.ScalarMappable(norm, "viridis"),
    cax=fig.add_axes(
        [
            (left + 2 * panel + wspace + 0.1) / width,
            (bottom + (2 * panel + hspace - bar_height) / 2) / height,
            0.07 / width,
            bar_height / height,
        ]
    ),
    ticks=[5, 10, 15, 20],
)
colorbar.set_label("Worst Nash improvement (\\%)", labelpad=4)

save_figure(fig, "signaling_dilemma_summary_sampled_refined_two_gamma_wide.pdf")
