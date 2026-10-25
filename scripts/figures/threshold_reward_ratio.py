"""Figure S2: the threshold reward ratio R* at which exploring first pays, over the distance
d and the horizon T = 1/(1 - gamma), with slices at fixed T and at fixed d."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.interpolate import RectBivariateSpline

from competitive_exploration.paths import DATA_DIR
from competitive_exploration.plotting import CM, PNAS_WIDTHS, save_figure, use_paper_style

use_paper_style(final_size=True)
data = np.load(DATA_DIR / "threshold_sweep.npz")
R, T, d = data["R_star"], data["T_values"], data["d_values"]  # R has shape (T, d)
logT = np.log10(T)

SHOWN_T = 10.0 ** np.arange(2, 10)
SHOWN_D = [2, 3, 5, 10, 20, 40]
LEVELS = [1.1, 1.5, 2, 3, 5, 10, 30, 100, 300, 1000]
LEGEND_KW = dict(handlelength=1.5, handletextpad=0.4, labelspacing=0.25, borderpad=0.7)


def contour_midpoints(ax, cs):
    """Label positions halfway along each level's longest piece, in screen distance.

    Levels whose contour is too short to hold a label (a corner of the plot) are left
    unlabeled."""
    points = []
    for segs in cs.allsegs:
        if not segs:
            continue
        xy = ax.transData.transform(max(segs, key=len))
        arc = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
        if arc[-1] >= 0.25 * ax.bbox.height:
            points.append(ax.transData.inverted().transform(xy[np.searchsorted(arc, arc[-1] / 2)]))
    return points


fig = plt.figure(figsize=(PNAS_WIDTHS["double"], 17.5 * CM), layout="constrained")
top, bottom = fig.subfigures(2, 1, height_ratios=[1.08, 1])
sub_b, sub_c = bottom.subfigures(1, 2)
for subfig, letter in [(top, "A"), (sub_b, "B"), (sub_c, "C")]:
    subfig.text(0, 1, rf"\textbf{{{letter}}}", ha="left", va="top", fontsize=8)

# A: heatmap over (T, d). The x axis is log10 T on a linear scale, because clabel cuts the
# wrong stretch of line on a log axis, so the decade labels and minor ticks are set by hand.
# Contours come from a cubic spline of log R*; levels below R*(d = 2) are omitted, since
# R* = 1 at d = 1 and such a contour would only interpolate between those two rows.
ax = top.subplots()
mesh = ax.pcolormesh(logT, d, R.T, norm=LogNorm(1, R.max()), shading="nearest", rasterized=True)
spline = RectBivariateSpline(logT, d, np.log(R))
logT_fine, d_fine = np.linspace(logT[0], logT[-1], 600), np.linspace(d[0], d[-1], 400)
levels = [v for v in LEVELS if R[:, 1].min() < v < R.max()]
cs = ax.contour(logT_fine, d_fine, np.exp(spline(logT_fine, d_fine)).T, levels=levels, colors="w", linewidths=0.5)
half_step = (logT[1] - logT[0]) / 2
ax.set_xlim(logT[0] - half_step, logT[-1] + half_step)
ax.set_ylim(d[0] - 0.5, d[-1] + 0.5)
decades = np.arange(np.ceil(logT[0]), np.floor(logT[-1]) + 1)
ax.set_xticks(decades, [rf"$10^{{{t:.0f}}}$" for t in decades])
minor = np.log10(np.outer(10.0**decades, np.arange(2, 10)).ravel())
ax.set_xticks(minor[(minor > logT[0]) & (minor < logT[-1])], minor=True)
ax.set_xlabel(r"Horizon $T = 1/(1-\gamma)$")
ax.set_ylabel(r"Distance $d$")
colorbar = top.colorbar(mesh, ax=ax, pad=0.01, fraction=0.03, aspect=20)
colorbar.ax.set_title(r"$R^*$", fontsize=plt.rcParams["font.size"])
fig.canvas.draw()  # fix the layout before placing the labels in screen coordinates
ax.clabel(cs, fmt=lambda v: f"{v:g}", fontsize=6, inline_spacing=2, manual=contour_midpoints(ax, cs))

# B: R* against d at one horizon per decade.
ax_b = sub_b.subplots()
for T_shown, color in zip(SHOWN_T, plt.cm.viridis(np.linspace(0, 0.9, len(SHOWN_T)))):
    row = int(np.argmin(np.abs(np.log(T / T_shown))))
    ax_b.plot(d, R[row], "o-", ms=1.6, lw=0.6, color=color, label=rf"$10^{{{np.log10(T_shown):.0f}}}$")
ax_b.set_xlim(0, d[-1] + 1)
ax_b.set_xlabel(r"Distance $d$")
ax_b.legend(loc="upper left", title=r"$T$", **LEGEND_KW)

# C: R* against T at select distances.
ax_c = sub_c.subplots()
for d_shown, color in zip(SHOWN_D, plt.cm.viridis(np.linspace(0, 0.9, len(SHOWN_D)))):
    col = list(d).index(d_shown)
    ax_c.plot(T, R[:, col], "o-", ms=1.0, lw=0.5, color=color, label=rf"${d_shown}$")
ax_c.set_xscale("log")
ax_c.set_xlabel(r"Horizon $T = 1/(1-\gamma)$")
ax_c.legend(loc="upper right", title=r"$d$", **LEGEND_KW)

# R* >= 1, so the data fill log10 R* in [0, span]; pad 3% of the panel height at each end.
span = np.log10(R.max())
for ax in [ax_b, ax_c]:
    ax.set_box_aspect(1)
    ax.set_yscale("log")
    ax.set_ylim(10.0 ** (-0.03 * span), 10.0 ** (1.03 * span))
    ax.set_ylabel(r"Threshold reward ratio $R^*$")
    ax.grid(True, lw=0.4)

save_figure(fig, "threshold_reward_ratio.pdf")
