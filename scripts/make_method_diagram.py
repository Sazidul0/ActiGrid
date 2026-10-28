"""Draw the ActiGrid method diagram (paper Fig. 1, the acquisition loop).

The figure is drawn from coordinates with matplotlib, in the same style as
scripts/make_paper_figures.py, and written as a vector PDF (plus a PNG
preview) to figures/fig_method. It shows the TU Delft sources with only the
resource source acquired, which is the policy's first action.

Usage (from the repository root):
    python scripts/make_method_diagram.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)

BLUE = "#2d68a6"  # the policy and its loop, the only accent
INK, MUTED, RULE, FILL = "#222222", "#6b6b6b", "#c8c8c8", "#efefef"

W, H = 3.5, 1.9  # IEEE single column, inches

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# The drawn state: only the resource source has been acquired so far
# (the TU Delft policy acquires it first).
SOURCES = [("Physical", False), ("Resource", True), ("Network", False)]
ROWS = [1.42, 1.04, 0.66]


def box(ax, x, y, w, h, text=None, edge=INK, face="white", lw=0.7, size=7, color=INK, ls="-"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, lw=lw, ls=ls, zorder=2))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
                color=color, zorder=3, linespacing=1.05)


def arrow(ax, pts, color=INK, lw=0.7, ls="-", head=True):
    """Polyline through pts; the last segment carries the arrow head."""
    for (x0, y0), (x1, y1) in zip(pts[:-2], pts[1:-1]):
        ax.plot([x0, x1], [y0, y1], color=color, lw=lw, ls=ls, solid_capstyle="butt", zorder=1)
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>" if head else "-",
                                 mutation_scale=6, color=color, lw=lw, ls=ls,
                                 shrinkA=0, shrinkB=0, zorder=1))


def main():
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    # Frozen classifier region.
    ax.add_patch(Rectangle((0.70, 0.30), 1.88, 1.42, facecolor=FILL, edgecolor="none", zorder=0))
    ax.text(2.52, 0.38, "frozen classifier", fontsize=6.5, style="italic", color=MUTED,
            ha="right", va="center")

    # Concatenated input z(a): one contiguous cell per source, the mask appended below.
    cx, cw, ch = 1.70, 0.11, ROWS[0] - ROWS[1]

    for (name, seen), y in zip(SOURCES, ROWS):
        path = INK if seen else RULE
        box(ax, 0.04, y - 0.12, 0.54, 0.24, name)
        box(ax, 0.82, y - 0.12, 0.26, 0.24, f"$f_{SOURCES.index((name, seen)) + 1}$")
        arrow(ax, [(0.58, y), (0.82, y)])
        arrow(ax, [(1.08, y), (1.30, y)], color=path, ls="-" if seen else (0, (2, 1.5)))
        ax.add_patch(Circle((1.36, y), 0.06, facecolor="white", edgecolor=INK, lw=0.7, zorder=2))
        ax.text(1.36, y - 0.005, r"$\times$", ha="center", va="center", fontsize=6.5, zorder=3)
        arrow(ax, [(1.42, y), (cx, y)], color=path, ls="-" if seen else (0, (2, 1.5)))
        box(ax, cx, y - ch / 2, cw, ch, face="#9a9a9a" if seen else "white", edge=INK, lw=0.6)

    # Mask a: brackets the gates, and is also appended to z(a).
    ax.add_patch(Rectangle((1.27, 0.50), 0.18, 1.08, facecolor="none", edgecolor=BLUE, lw=0.7,
                           ls=(0, (3, 1.5)), zorder=1))
    ax.text(1.36, 0.43, "$a$", ha="center", va="center", fontsize=7.5, color=BLUE)
    mask_y = ROWS[-1] - ch / 2 - 0.12
    box(ax, cx, mask_y, cw, 0.12, edge=BLUE, lw=0.6)
    ax.text(cx + cw / 2, mask_y + 0.06, "$a$", ha="center", va="center", fontsize=6, color=BLUE, zorder=3)
    ax.text(cx + cw / 2, ROWS[0] + ch / 2 + 0.08, "$z(a)$", ha="center", va="center", fontsize=7)

    # Fusion head and posterior.
    box(ax, 1.98, 0.86, 0.46, 0.36, "fusion\nhead")
    arrow(ax, [(cx + cw, 1.04), (1.98, 1.04)])
    bars = [0.06, 0.24, 0.05, 0.03, 0.10, 0.04]
    for i, b in enumerate(bars):
        ax.add_patch(Rectangle((2.66 + i * 0.045, 0.86), 0.032, b * 1.3,
                               facecolor=MUTED, edgecolor="none", zorder=2))
    ax.plot([2.64, 2.94], [0.86, 0.86], color=INK, lw=0.6, zorder=2)
    ax.text(2.79, 0.76, "$p,\\;H$", ha="center", va="center", fontsize=7)
    arrow(ax, [(2.44, 1.04), (2.62, 1.04)])

    # Policy. Its state s_t = [z, p, H, a, 1 - C_t]: the gated latent and mask
    # branch off z(a), the posterior and entropy come from the head, and the
    # cost headroom enters on its own.
    box(ax, 2.72, 1.38, 0.62, 0.30, "Double DQN", edge=BLUE, lw=0.9, color=BLUE)
    bx = 1.895
    ax.add_patch(Circle((bx, 1.04), 0.018, facecolor=INK, edgecolor="none", zorder=3))
    arrow(ax, [(bx, 1.04), (bx, 1.53), (2.72, 1.53)])
    ax.text(2.66, 1.57, "$s_t$", ha="right", va="bottom", fontsize=7)
    arrow(ax, [(2.79, 1.24), (2.79, 1.38)])
    arrow(ax, [(3.22, 1.24), (3.22, 1.38)])
    ax.text(3.22, 1.18, "$1-C_t$", ha="center", va="center", fontsize=6.5)

    # Acquire: back to the mask along the top.
    arrow(ax, [(3.03, 1.68), (3.03, 1.82), (1.36, 1.82), (1.36, 1.58)], color=BLUE, lw=0.8)
    ax.text(2.20, 1.86, "acquire source $m$", ha="center", va="bottom", fontsize=6.5, color=BLUE)

    # Stop: diagnose with the current posterior.
    arrow(ax, [(3.34, 1.53), (3.44, 1.53), (3.44, 0.30), (2.79, 0.30), (2.79, 0.18)],
          color=BLUE, lw=0.8)
    ax.text(3.40, 0.36, "STOP", ha="right", va="bottom", fontsize=6.5, color=BLUE)
    ax.text(2.79, 0.10, r"diagnosis $\hat{y}$", ha="center", va="center", fontsize=7)

    fig.savefig(OUT / "fig_method.pdf", pad_inches=0.01, bbox_inches="tight")
    fig.savefig(OUT / "fig_method.png", pad_inches=0.01, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote", OUT / "fig_method.pdf")


if __name__ == "__main__":
    main()
