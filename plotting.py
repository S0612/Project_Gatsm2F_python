"""Shared matplotlib style and the figures used by estimate.py and monte_carlo.py."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3de"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "legend.frameon": False, "lines.linewidth": 1.3,
})


def _grid(n, ncols=3, size=(4.0, 3.0)):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(size[0] * ncols, size[1] * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes


def yield_fit_figure(dates, y, yhat, mats_months, path):
    fig, axes = _grid(len(mats_months))
    for i, ax in enumerate(axes[:len(mats_months)]):
        ax.plot(dates, y[:, i] * 100, color=INK2, lw=1.6, label="Actual")
        ax.plot(dates, yhat[:, i] * 100, color=BLUE, lw=0.9, label="Model")
        ax.set_title(f"{mats_months[i] / 12:g} year yield", loc="left")
        ax.set_ylabel("Yield (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def decomposition_figure(dates, yhat, rexp, tp, mats_months, path, pick=(24, 120)):
    idx = [list(mats_months).index(m) for m in pick if m in list(mats_months)]
    if not idx:
        idx = [0, len(mats_months) - 1]
    fig, axes = plt.subplots(1, len(idx), figsize=(5.2 * len(idx), 3.6))
    axes = np.atleast_1d(axes)
    for ax, i in zip(axes, idx):
        ax.plot(dates, yhat[:, i] * 100, color=INK, lw=1.4, label="Fitted yield")
        ax.plot(dates, rexp[:, i] * 100, color=BLUE, lw=1.4, label="Expected short rate")
        ax.plot(dates, tp[:, i] * 100, color=ORANGE, lw=1.4, label="Term premium")
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_title(f"{mats_months[i] / 12:g} year yield", loc="left")
        ax.set_ylabel("Percent")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def mc_histograms(est, true, names, labels, path, title=None):
    """est: (R, P) estimates, true: (P,). One histogram per listed parameter."""
    fig, axes = _grid(len(names), ncols=3, size=(3.8, 2.8))
    for ax, name, lab in zip(axes, names, labels):
        j = list(PARAMS).index(name)
        x = est[:, j]
        ax.hist(x, bins=30, density=True, color=BLUE, edgecolor=SURFACE, linewidth=0.5,
                label="Estimates")
        ax.axvline(true[j], color=INK, lw=1.6, label="True value")
        ax.axvline(np.mean(x), color=ORANGE, lw=1.6, ls="--", label="Mean of estimates")
        ax.set_title(lab, loc="left")
        ax.set_yticks([])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(False)
        ax.spines["left"].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96 if title else 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


PARAMS = [
    "phiQ11", "phiQ22", "alpha", "muP1", "muP2",
    "phiP11", "phiP12", "phiP21", "phiP22",
    "sigma11", "sigma21", "sigma22", "stdY",
]
