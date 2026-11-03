"""
Schematic diagrams for derivation.md.   python make_diagrams.py  ->  figures/diagram_*.png

Every diagram is drawn with plain matplotlib so it can be edited and re-run.
Vertical scales are exaggerated for clarity.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle, Arc

import plotstyle as ps
from viscoelastic_beam import IceShelf, ViscoelasticBeam, parabolic_lake

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
ps.apply()
plt.rcParams["axes.grid"] = False


def canvas(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def arrow(ax, p0, p1, color=ps.INK, lw=1.6, style="-|>", ms=12, **kw):
    kw.setdefault("zorder", 6)
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                        shrinkA=0, shrinkB=0, **kw)
    ax.add_patch(a)
    return a


def dim(ax, p0, p1, text, color=ps.INK2, off=(0.12, 0), fs=11, **kw):
    arrow(ax, p0, p1, color=color, lw=1.1, style="<|-|>", ms=8)
    mx, my = (p0[0] + p1[0]) / 2 + off[0], (p0[1] + p1[1]) / 2 + off[1]
    ax.text(mx, my, text, color=color, fontsize=fs, va="center", **kw)


def spring(ax, x, y0, y1, n=6, width=0.12, color=ps.BUOY, lw=1.4):
    ys = np.linspace(y0, y1, 2 * n + 3)
    xs = np.full_like(ys, x)
    xs[2:-2:2] += width
    xs[3:-2:2] -= width
    ax.plot(xs, ys, color=color, lw=lw, solid_joinstyle="round")


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=170)
    plt.close(fig)
    print("  wrote figures/" + name)


def lake_deflection(xs, centre, half_width, depth_scale):
    """Use the real model to get a realistic elastic dip (then exaggerate)."""
    shelf = IceShelf()
    x = np.linspace(-15e3, 15e3, 1501)
    beam = ViscoelasticBeam(x, shelf)
    w, _ = beam.elastic(shelf.lake_load(parabolic_lake(x, 0, 1500, 4)))
    return np.interp((xs - centre) / half_width * 1500, x, w) / w.max() * depth_scale


# ---------------------------------------------------------------------------
def diagram_setup():
    fig, ax = canvas(11, 4.6, (-4.2, 20.5), (-4.2, 3.6))
    H, rho = 2.0, 917 / 1028
    top0, bot0 = H * (1 - rho), -H * rho
    xs = np.linspace(0, 17, 600)
    w = lake_deflection(xs, 9.0, 1.6, 0.55)
    top, bot = top0 - w, bot0 - w
    # ocean
    ax.add_patch(Rectangle((0.0, -4.0), 20.3, 4.0, color=ps.OCEAN_LIGHT, zorder=0))
    ax.plot([0.0, 20.3], [0, 0], color=ps.OCEAN, lw=1.2, ls=(0, (6, 3)))
    ax.text(20.3, 0.12, "sea level", ha="right", va="bottom", color=ps.OCEAN, fontsize=10)
    # bed and grounded ice
    bedx = np.array([-4.2, -1.0, 0.0, 3.0, 20.5, 20.5, -4.2])
    bedy = np.array([-1.4, -2.0, -2.25, -3.5, -3.9, -4.2, -4.2])
    ax.add_patch(Polygon(np.c_[bedx, bedy], color="#b8a58c", zorder=1))
    gx = np.array([-4.2, 0.0, 0.0, -1.0, -4.2])
    gy = np.array([top0 + 1.0, top0, bot0, -2.0, -1.4])
    ax.add_patch(Polygon(np.c_[gx, gy], fc=ps.ICE_FILL, ec=ps.ICE_EDGE, lw=1.2, zorder=2))
    # floating shelf
    shelf = np.r_[np.c_[xs, top], np.c_[xs[::-1], bot[::-1]]]
    ax.add_patch(Polygon(shelf, fc=ps.ICE_FILL, ec=ps.ICE_EDGE, lw=1.2, zorder=2))
    # undeflected reference
    ax.plot(xs, np.full_like(xs, top0), color=ps.MUTED, lw=0.9, ls=":", zorder=3)
    ax.plot(xs, np.full_like(xs, bot0), color=ps.MUTED, lw=0.9, ls=":", zorder=3)
    # lake
    level = top0 - 0.08
    inl = top < level
    ax.fill_between(xs[inl], top[inl], level, color=ps.WATER, alpha=0.9, zorder=4, lw=0)
    # labels
    ax.text(-2.3, -0.4, "grounded\nice", ha="center", fontsize=10, color=ps.INK2)
    ax.text(-2.6, -3.3, "bed", ha="center", fontsize=10, color="#6d5d48")
    ax.plot([0, 0], [bot0 - 0.15, top0 + 1.9], color=ps.INK, lw=1.0, ls="--")
    ax.text(0.1, top0 + 1.95, "grounding line\n(clamped)", fontsize=10, va="bottom")
    ax.text(17.1, top0 + 1.95, "calving front\n(free)", fontsize=10, va="bottom", ha="center")
    ax.plot([17, 17], [top0 + 0.1, top0 + 1.9], color=ps.INK, lw=1.0, ls="--")
    ax.text(4.0, -1.0, "floating ice shelf", fontsize=11, color=ps.INK2, ha="center")
    ax.text(14.0, -3.0, "ocean", fontsize=11, color=ps.OCEAN, ha="center")
    ax.annotate("meltwater lake\ndepth $d(x)$", (9.0, level - 0.15), (11.4, 2.6), fontsize=10.5,
                color=ps.INK, arrowprops=dict(arrowstyle="-|>", color=ps.INK, lw=1.0), ha="left")
    dim(ax, (14.5, bot0), (14.5, top0), "$H$", off=(0.15, 0))
    i = np.argmin(np.abs(xs - 9))
    arrow(ax, (9.0, bot0), (9.0, bot[i]), color=ps.ACCENT, lw=1.6, ms=10)
    ax.text(9.15, bot0 - 0.35, "$w(x)$", color=ps.ACCENT, fontsize=11)
    # axes
    arrow(ax, (17.8, -3.2), (19.2, -3.2), lw=1.3)
    arrow(ax, (17.8, -3.2), (17.8, -1.8), lw=1.3)
    ax.text(19.3, -3.2, "$x$", va="center", fontsize=11)
    ax.text(17.8, -1.65, "$z$", ha="center", fontsize=11)
    ax.text(-4.0, -4.05, "vertical scale exaggerated", ha="left", fontsize=8.5, color="#6d5d48")
    save(fig, "diagram_setup.png")


# ---------------------------------------------------------------------------
def diagram_buoyancy():
    fig, ax = canvas(10, 4.0, (-0.5, 15.3), (-4.0, 2.6))
    rho = 0.62  # cartoon density ratio so the freeboard is visible
    H = 2.6
    ax.add_patch(Rectangle((-0.6, -3.8), 7.6, 3.8, color=ps.OCEAN_LIGHT, zorder=0))
    ax.plot([-0.6, 7.0], [0, 0], color=ps.OCEAN, lw=1.0, ls=(0, (6, 3)))
    ax.text(-0.5, 0.08, "sea level", fontsize=8.5, color=ps.OCEAN, va="bottom")
    for x0, w, ttl in [(0.9, 0.0, "floating freely"), (4.4, 0.55, "loaded: sinks by $w$")]:
        top, bot = H * (1 - rho) - w, -H * rho - w
        ax.add_patch(Rectangle((x0, bot), 1.2, H, fc=ps.ICE_FILL, ec=ps.ICE_EDGE, lw=1.2, zorder=2))
        ax.text(x0 + 0.6, 2.25, ttl, ha="center", fontsize=10.5)
        if w > 0:
            ax.add_patch(Rectangle((x0, top), 1.2, 0.35, fc=ps.WATER, lw=0, zorder=3))
            arrow(ax, (x0 + 0.6, top + 1.15), (x0 + 0.6, top + 0.4), color=ps.LOAD, lw=2.0)
            ax.text(x0 + 0.75, top + 0.85, "load $q$", color=ps.LOAD, fontsize=10)
            ax.add_patch(Rectangle((x0, bot), 1.2, w, fc=ps.BUOY, alpha=0.35, lw=0, zorder=3))
            ax.plot([x0 - 0.2, x0 + 1.4], [bot + w, bot + w], color=ps.MUTED, lw=0.9, ls=":")
            dim(ax, (x0 + 1.55, bot), (x0 + 1.55, bot + w), "$w$", off=(0.12, 0))
            for xa in (x0 + 0.3, x0 + 0.9):
                arrow(ax, (xa, bot - 0.95), (xa, bot - 0.05), color=ps.BUOY, lw=2.0)
            ax.text(x0 + 0.6, bot - 1.35, r"extra upthrust $\rho_{sw} g\, w$", ha="center",
                    color=ps.BUOY, fontsize=10.5)
    # spring analogy
    x0 = 9.2
    ax.text(x0 + 2.9, 2.25, "...acts like a bed of springs", ha="center", fontsize=10.5)
    ax.add_patch(Rectangle((x0, 0.0), 5.8, 0.6, fc=ps.ICE_FILL, ec=ps.ICE_EDGE, lw=1.2))
    for xs in np.linspace(x0 + 0.35, x0 + 5.45, 9):
        spring(ax, xs, -2.0, 0.0)
    ax.add_patch(Rectangle((x0 - 0.2, -2.3), 6.2, 0.3, fc="none", ec=ps.INK2, hatch="////",
                           lw=0))
    ax.plot([x0 - 0.2, x0 + 6.0], [-2.0, -2.0], color=ps.INK2, lw=1.2)
    ax.text(x0 + 2.9, -2.85, r"stiffness $k = \rho_{sw}\, g \approx 10^4$ Pa m$^{-1}$",
            ha="center", fontsize=10.5, color=ps.BUOY)
    ax.text(x0 + 2.9, -3.45, "(a Winkler foundation)", ha="center", fontsize=9.5, color=ps.INK2)
    ax.text(-0.5, -3.75, "cartoon, not to scale", fontsize=8, color=ps.INK2, va="bottom")
    save(fig, "diagram_buoyancy.png")


# ---------------------------------------------------------------------------
def curved_arrow(ax, centre, r, a0, a1, color, lw=1.8):
    """Circular arrow from angle a0 to a1 (degrees), arrow head at a1."""
    th = np.radians(np.linspace(a0, a1, 40))
    xs, ys = centre[0] + r * np.cos(th), centre[1] + r * np.sin(th)
    ax.plot(xs[:-3], ys[:-3], color=color, lw=lw)
    arrow(ax, (xs[-4], ys[-4]), (xs[-1], ys[-1]), color=color, lw=lw, ms=12)


def diagram_free_body():
    fig, ax = canvas(10, 4.6, (-4.6, 9.4), (-3.2, 3.2))
    w_, h_ = 3.0, 1.4
    ax.add_patch(Rectangle((0, -h_ / 2), w_, h_, fc=ps.ICE_FILL, ec=ps.ICE_EDGE, lw=1.4))
    ax.plot([0, w_], [0, 0], color=ps.MUTED, ls="--", lw=0.9)
    dim(ax, (0, -h_ / 2 - 0.35), (w_, -h_ / 2 - 0.35), "", off=(0, 0))
    ax.text(w_ / 2, -h_ / 2 - 0.62, "$dx$", ha="center", va="top", fontsize=11)
    # load
    for xa in np.linspace(0.3, w_ - 0.3, 5):
        arrow(ax, (xa, h_ / 2 + 1.0), (xa, h_ / 2 + 0.05), color=ps.LOAD, lw=1.8)
    ax.text(w_ / 2, h_ / 2 + 1.15, r"lake load  $q\,dx$", ha="center", va="bottom",
            color=ps.LOAD, fontsize=11)
    # buoyancy
    for xa in np.linspace(0.3, w_ - 0.3, 5):
        arrow(ax, (xa, -h_ / 2 - 2.1 + 0.6), (xa, -h_ / 2 - 0.95 + 0.0), color=ps.BUOY, lw=1.8)
    ax.text(w_ / 2, -h_ / 2 - 1.75, r"buoyancy  $k\,w\,dx$", ha="center", va="top",
            color=ps.BUOY, fontsize=11)
    # shear
    arrow(ax, (-0.25, 0.7), (-0.25, -0.7), color=ps.INK, lw=2)
    ax.text(-0.4, 0.0, "$V$", ha="right", va="center", fontsize=12)
    arrow(ax, (w_ + 0.25, -0.7), (w_ + 0.25, 0.7), color=ps.INK, lw=2)
    ax.text(w_ + 0.4, 0.0, "$V + dV$", ha="left", va="center", fontsize=12)
    # moments
    curved_arrow(ax, (0, 0), 1.25, 120, 240, ps.ACCENT)   # counter-clockwise on left face
    ax.text(-1.45, 1.05, "$M$", color=ps.ACCENT, fontsize=12, ha="right")
    curved_arrow(ax, (w_, 0), 1.25, 60, -60, ps.ACCENT)   # clockwise on right face
    ax.text(w_ + 1.45, 1.05, "$M + dM$", color=ps.ACCENT, fontsize=12)
    # equations
    ax.text(6.0, 1.3, "forces:", fontsize=11, color=ps.INK2)
    ax.text(6.0, 0.7, r"$\dfrac{dV}{dx} = q - k\,w$", fontsize=13)
    ax.text(6.0, -0.2, "moments:", fontsize=11, color=ps.INK2)
    ax.text(6.0, -0.8, r"$\dfrac{dM}{dx} = V$", fontsize=13)
    ax.text(6.0, -2.0, r"$\Rightarrow\;\; M'' + k\,w = q$", fontsize=14, color=ps.INK,
            bbox=dict(fc="white", ec=ps.ACCENT, boxstyle="round,pad=0.4"))
    save(fig, "diagram_free_body.png")


# ---------------------------------------------------------------------------
def diagram_bending():
    fig, ax = canvas(11, 3.5, (-4.5, 12.5), (-3.0, 2.4))
    # an arc-shaped (hogging) beam segment
    R, half, h = 9.0, 0.36, 1.3
    th = np.linspace(-half, half, 100)
    cx, cy = 0.0, -R
    outer = np.c_[cx + (R + h / 2) * np.sin(th), cy + (R + h / 2) * np.cos(th)]
    inner = np.c_[cx + (R - h / 2) * np.sin(th), cy + (R - h / 2) * np.cos(th)]
    ax.add_patch(Polygon(np.r_[outer, inner[::-1]], fc=ps.ICE_FILL, ec=ps.ICE_EDGE, lw=1.4))
    mid = np.c_[cx + R * np.sin(th), cy + R * np.cos(th)]
    ax.plot(mid[:, 0], mid[:, 1], color=ps.MUTED, ls="--", lw=1)
    ax.text(mid[-1, 0] + 0.15, mid[-1, 1], "neutral\naxis", fontsize=9, color=ps.MUTED,
            va="center")
    for s in (-1, 1):  # cross-section lines stay straight
        a = s * half * 0.55
        p0 = (cx + (R - h / 2) * np.sin(a), cy + (R - h / 2) * np.cos(a))
        p1 = (cx + (R + h / 2) * np.sin(a), cy + (R + h / 2) * np.cos(a))
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=ps.ICE_EDGE, lw=1.0)
    arrow(ax, (-1.7, outer[30, 1] + 0.45), (-2.6, outer[5, 1] + 0.45), color=ps.TENSION, lw=1.5)
    arrow(ax, (1.7, outer[30, 1] + 0.45), (2.6, outer[5, 1] + 0.45), color=ps.TENSION, lw=1.5)
    ax.text(0, outer[50, 1] + 0.45, "top stretched", ha="center", color=ps.TENSION, fontsize=10)
    ax.text(0, inner[50, 1] - 0.5, "bottom squashed", ha="center", va="top", color=ps.VISCO,
            fontsize=10)
    ax.text(0, -2.6, r"curvature $\kappa = 1/R \approx w''$  (here $w''>0$: a hump)",
            ha="center", fontsize=10.5)
    # strain / stress profile
    x0, z0 = 6.0, 0.0
    hh = 2.4
    ax.plot([x0, x0], [z0 - hh / 2, z0 + hh / 2], color=ps.INK, lw=1.2)
    ax.plot([x0 - 1.4, x0 + 1.4], [z0 - hh / 2, z0 + hh / 2], color=ps.ACCENT, lw=2)
    for zz in np.linspace(-hh / 2 + 0.15, hh / 2 - 0.15, 7):
        xx = x0 + 1.4 * zz / (hh / 2)
        if abs(xx - x0) > 0.15:
            arrow(ax, (x0, zz), (xx, zz), color=ps.TENSION if zz > 0 else ps.VISCO, lw=1.1, ms=7)
    ax.text(x0, z0 + hh / 2 + 0.3, "fibre stress across the thickness", ha="center", fontsize=10)
    ax.text(x0 + 1.5, z0 + hh / 2 - 0.1, r"$\sigma = E'\,z\,\kappa$", color=ps.ACCENT,
            fontsize=12)
    ax.text(x0 - 0.15, z0 - hh / 2 - 0.1, "$-H/2$", ha="right", fontsize=9.5, va="top")
    ax.text(x0 - 0.15, z0 + hh / 2, "$+H/2$", ha="right", fontsize=9.5, va="bottom")
    ax.text(x0 + 0.1, z0 - 0.25, "$z$", fontsize=10, color=ps.INK2)
    ax.text(8.7, 0.6, r"$M = \int_{-H/2}^{H/2} \sigma\, z\, dz$", fontsize=12.5)
    ax.text(8.7, -0.35, r"$\;\;\; = \dfrac{E' H^3}{12}\,\kappa = D\,\kappa$", fontsize=12.5)
    ax.text(8.7, -1.5, r"$E' = E/(1-\nu^2)$  (plane strain)", fontsize=10, color=ps.INK2)
    save(fig, "diagram_bending.png")


# ---------------------------------------------------------------------------
def diagram_maxwell():
    fig = plt.figure(figsize=(11, 4.0))
    ax = fig.add_axes([0.0, 0.0, 0.42, 1.0])
    ax.set_xlim(0, 7)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")
    ax.axis("off")
    y = 0.4
    ax.plot([0.3, 1.0], [y, y], color=ps.INK, lw=1.6)
    xs = np.linspace(1.0, 2.8, 13)
    ys = y + np.r_[0, np.tile([0.3, -0.3], 5), 0.3, 0][: xs.size] * 1.0
    ys[0] = ys[-1] = y
    ax.plot(xs, ys, color=ps.ELASTIC, lw=2)
    ax.plot([2.8, 3.5], [y, y], color=ps.INK, lw=1.6)
    # dashpot
    ax.plot([3.5, 3.5], [y - 0.45, y + 0.45], color=ps.VISCO, lw=2)
    ax.plot([3.5, 4.9], [y + 0.45, y + 0.45], color=ps.VISCO, lw=2)
    ax.plot([3.5, 4.9], [y - 0.45, y - 0.45], color=ps.VISCO, lw=2)
    ax.plot([4.3, 4.3], [y - 0.32, y + 0.32], color=ps.VISCO, lw=3)
    ax.plot([4.3, 6.3], [y, y], color=ps.INK, lw=1.6)
    arrow(ax, (6.3, y), (6.9, y), lw=1.8)
    arrow(ax, (0.3, y), (-0.3 + 0.3, y), lw=1.8)
    ax.text(1.9, y + 0.65, "spring (elastic)\n$D$", ha="center", color=ps.ELASTIC, fontsize=10.5)
    ax.text(4.2, y + 0.65, r"dashpot (viscous)" + "\n" + r"$\Lambda$", ha="center",
            color=ps.VISCO, fontsize=10.5)
    ax.text(3.5, -0.75, r"$\dot\kappa \;=\; \dfrac{\dot M}{D} \;+\; \dfrac{M}{\Lambda}$",
            ha="center", fontsize=15)
    ax.text(3.5, -1.85, r"$\tau = \Lambda / D$  (Maxwell time)", ha="center", fontsize=11.5,
            color=ps.INK2)
    ax.text(3.5, 1.85, "Maxwell element: the bending rates add", ha="center", fontsize=11)
    # creep and relaxation
    t = np.linspace(-0.5, 4, 400)
    a1 = fig.add_axes([0.5, 0.2, 0.22, 0.62])
    a2 = fig.add_axes([0.77, 0.2, 0.22, 0.62])
    k = np.where(t < 0, 0, 1 + t)
    a1.plot(t, k, color=ps.VISCO)
    a1.plot(t, np.where(t < 0, 0, 1), color=ps.ELASTIC, ls="--", lw=1.3)
    a1.set_title("hold the moment fixed", fontsize=10.5)
    a1.set_xlabel(r"time / $\tau$")
    a1.set_ylabel(r"curvature $\kappa$")
    a1.text(0.3, 2.7, "viscous creep", color=ps.VISCO, fontsize=10)
    a1.text(1.4, 0.65, "elastic jump", color=ps.ELASTIC, fontsize=9.5)
    m = np.where(t < 0, 0, np.exp(-t))
    a2.plot(t, m, color=ps.VISCO)
    a2.set_title("hold the curvature fixed", fontsize=10.5)
    a2.set_xlabel(r"time / $\tau$")
    a2.set_ylabel(r"moment $M$")
    a2.text(1.0, 0.55, r"relaxes as $e^{-t/\tau}$", color=ps.VISCO, fontsize=10)
    for a in (a1, a2):
        a.set_yticks([])
        a.grid(True)
    save(fig, "diagram_maxwell.png")


# ---------------------------------------------------------------------------
def diagram_limits():
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))
    xs = np.linspace(-6, 6, 600)
    H, rho = 1.6, 917 / 1028
    shelf = IceShelf()
    x = np.linspace(-15e3, 15e3, 1501)
    q = shelf.lake_load(parabolic_lake(x, 0, 1500, 4))
    w_el, _ = ViscoelasticBeam(x, shelf).elastic(q)
    w_iso = q / shelf.k
    s = 1.2 / w_iso.max()
    for ax, w, ttl, col in [(axs[0], w_el, r"$t = 0^+$: elastic plate spreads the load", ps.ELASTIC),
                            (axs[1], w_iso, r"$t \to \infty$: columns float independently", ps.ISO)]:
        ax.set_xlim(-6, 6)
        ax.set_ylim(-4.0, 1.6)
        ax.set_aspect("equal")
        ax.axis("off")
        ww = np.interp(xs * 1000, x, w) * s
        top, bot = H * (1 - rho) - ww, -H * rho - ww
        ax.add_patch(Rectangle((-6, -3.2), 12, 3.2, color=ps.OCEAN_LIGHT, zorder=0))
        ax.plot([-6, 6], [0, 0], color=ps.OCEAN, lw=1, ls=(0, (6, 3)))
        if w is w_iso:
            edges = np.linspace(-6, 6, 25)
            for a_, b_ in zip(edges[:-1], edges[1:]):
                m = (xs >= a_) & (xs <= b_)
                ax.add_patch(Polygon(np.r_[np.c_[xs[m], top[m]], np.c_[xs[m][::-1], bot[m][::-1]]],
                                     fc=ps.ICE_FILL, ec=ps.ICE_EDGE, lw=0.8))
        else:
            ax.add_patch(Polygon(np.r_[np.c_[xs, top], np.c_[xs[::-1], bot[::-1]]],
                                 fc=ps.ICE_FILL, ec=ps.ICE_EDGE, lw=1.2))
        level = H * (1 - rho)  # flat lake surface at the undisturbed ice-surface level
        ax.fill_between(xs, top, level, where=top < level - 1e-3, color=ps.WATER, lw=0)
        ax.set_title(ttl, fontsize=11, color=ps.INK)
        ax.plot(xs, bot, color=col, lw=2)
    axs[0].text(0, -3.65, r"$D\,w'''' + k\,w = q$ : dish $\approx$ lake width $+$ a few $\ell$,"
                "\nwith low flexural bulges either side", ha="center", fontsize=9.5, color=ps.INK2)
    axs[1].text(0, -3.65, r"$k\,w = q$ : the dip copies the lake exactly" + "\n(local isostasy)",
                ha="center", fontsize=9.5, color=ps.INK2)
    fig.tight_layout()
    save(fig, "diagram_limits.png")


# ---------------------------------------------------------------------------
def diagram_grid():
    fig, ax = canvas(11.5, 5.4, (-1.6, 12.6), (-4.4, 2.6))
    n = 9
    xs = np.arange(n) * 1.25 + 0.5
    # beam line
    ax.plot([xs[0], xs[-1]], [0, 0], color=ps.ICE_EDGE, lw=6, alpha=0.35, solid_capstyle="butt")
    for i, x in enumerate(xs):
        ax.plot(x, 0, "o", color=ps.VISCO, ms=9, zorder=3)
        lab = {0: "", 1: "$w_1$", n - 1: "$w_{N}$"}.get(i, "")
        if i == n // 2:
            lab = "$w_j$"
        ax.text(x, 0.35, lab, ha="center", fontsize=11)
    # ghost nodes
    for x in (xs[0] - 1.25, xs[-1] + 1.25):
        ax.plot(x, 0, "o", mfc="white", mec=ps.MUTED, ms=9, mew=1.5, zorder=3)
    ax.text(xs[0] - 1.25, -0.55, "ghost\n$w_{-1}$", ha="center", va="top", fontsize=9, color=ps.MUTED)
    ax.text(xs[-1] + 1.25, -0.55, "ghost\n$M_{N+1}$", ha="center", va="top", fontsize=9,
            color=ps.MUTED)
    ax.text(xs[0] + 0.15, -0.55, "$w_0$", ha="left", va="top", fontsize=11)
    dim(ax, (xs[3], -0.45), (xs[4], -0.45), "", off=(0, 0))
    ax.text((xs[3] + xs[4]) / 2, -0.75, r"$\Delta x$", ha="center", va="top", fontsize=11)
    # left: clamped, right: free
    ax.add_patch(Rectangle((xs[0] - 0.45, -1.0), 0.35, 2.0, fc="none", ec=ps.INK, hatch="////"))
    ax.text(xs[0] - 0.3, 1.25, "grounding line: clamped\n"
            r"$w_0 = 0,\;\; w_{-1} = w_1$" + "\n(a kinematic condition: lives in $L$)",
            fontsize=9.5, va="bottom")
    ax.text(xs[-1] + 0.4, 1.25, "calving front: free\n"
            r"$M_N = 0,\;\; M_{N+1} = M_{N-1}$" + "\n(a dynamic condition: lives in $L^*$)",
            fontsize=9.5, va="bottom", ha="right")
    # operator chain
    y = -2.35
    boxes = [(0.2, "deflection\n$w$", ps.VISCO), (3.4, "curvature\n$\\kappa$", ps.INK2),
             (6.6, "moment\n$M$", ps.ACCENT), (9.8, "net force\n$q - k\\,w$", ps.LOAD)]
    for x, t, c in boxes:
        ax.text(x + 0.9, y, t, ha="center", va="center", fontsize=10.5, color=c,
                bbox=dict(fc="white", ec=c, boxstyle="round,pad=0.45"))
    labs = [r"$\kappa = L\,w$", r"Maxwell: $\dot M = D\dot\kappa - M/\tau$", r"$L^* M$"]
    for (x0, _, _), (x1, _, _), lab in zip(boxes[:-1], boxes[1:], labs):
        arrow(ax, (x0 + 1.95, y), (x1 - 0.15, y), lw=1.5)
        ax.text((x0 + x1) / 2 + 0.9, y + 0.62, lab, ha="center", fontsize=9.5)
    ax.text(5.5, -3.75, r"$L$ and $L^*$ are both the stencil $[\,1,\;-2,\;1\,]/\Delta x^2$;"
            r" they differ only in their end rows (the boundary conditions)",
            ha="center", fontsize=10, color=ps.INK2)
    save(fig, "diagram_grid.png")


def diagram_timestep():
    fig, ax = canvas(10.5, 3.6, (0, 14), (-2.6, 2.4))
    steps = [
        (1.6, "known at $t_n$:\n$w^n,\\;M^n$", ps.INK2),
        (5.4, "solve ONE banded\nlinear system\nfor $w^{n+1}$", ps.VISCO),
        (9.2, "update moment\n$M^{n+1}$\n(one line)", ps.ACCENT),
        (12.5, "stress\n$\\sigma = 6M/H^2$", ps.TENSION),
    ]
    for x, t, c in steps:
        ax.text(x, 0.9, t, ha="center", va="center", fontsize=10.5, color=c,
                bbox=dict(fc="white", ec=c, boxstyle="round,pad=0.5"))
    for (x0, _, _), (x1, _, _) in zip(steps[:-1], steps[1:]):
        arrow(ax, (x0 + 1.35, 0.9), (x1 - 1.35, 0.9), lw=1.5)
    ax.add_patch(FancyArrowPatch((9.2, 0.0), (1.6, 0.0), connectionstyle="arc3,rad=-0.25",
                                 arrowstyle="-|>", mutation_scale=12, color=ps.MUTED, lw=1.2))
    ax.text(5.4, -1.6, r"repeat with $n \to n+1$  (matrix factorised once: each step costs $O(N)$)",
            ha="center", fontsize=10, color=ps.INK2)
    save(fig, "diagram_timestep.png")


if __name__ == "__main__":
    diagram_setup()
    diagram_buoyancy()
    diagram_free_body()
    diagram_bending()
    diagram_maxwell()
    diagram_limits()
    diagram_grid()
    diagram_timestep()
