"""Shared colours and matplotlib style for the flexure figures and diagrams."""
import matplotlib as mpl

# data-series colours (categorical slots 1-3, validated colour-blind safe)
VISCO = "#2a78d6"     # viscoelastic model
ELASTIC = "#eb6834"   # instantaneous elastic limit
ISO = "#1baf7a"       # isostatic (fully relaxed) limit
ACCENT = "#6250d6"    # moments / annotations
TENSION = "#e34948"
SEQ = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
       "#184f95", "#104281", "#0d366b"]  # sequential blue, light -> dark

# physical "scene" colours for schematics
ICE_FILL = "#e3eef8"
ICE_EDGE = "#5b7a95"
OCEAN = "#2f5f8a"
OCEAN_LIGHT = "#b9d3ea"
WATER = "#3fa9f5"
LOAD = "#eb6834"
BUOY = "#1baf7a"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def apply():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "lines.linewidth": 2.0,
        "font.size": 10.5,
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    })


# ---------------------------------------------------------------------------
# Map helpers for (y, x) DataArrays in polar-stereographic metres
# ---------------------------------------------------------------------------
def map_extent(da, scale=1e-3):
    """imshow extent [km] of a (y, x) DataArray with regular coordinates."""
    import numpy as np
    x, y = np.asarray(da.x) * scale, np.asarray(da.y) * scale
    hx, hy = abs(x[1] - x[0]) / 2, abs(y[1] - y[0]) / 2
    return [x.min() - hx, x.max() + hx, y.min() - hy, y.max() + hy]


def show_map(ax, da, title=None, cbar=None, grid=False, **kw):
    """Draw a (y, x) DataArray on ``ax`` with axes in km; returns the image.
    Call it again on the same axes to overlay another field."""
    import numpy as np
    flip = float(da.y[0]) < float(da.y[-1])
    im = ax.imshow(np.asarray(da)[::-1] if flip else np.asarray(da), extent=map_extent(da),
                   interpolation="nearest", **kw)
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_aspect("equal")
    ax.grid(grid)
    if cbar:
        import matplotlib.pyplot as plt
        plt.colorbar(im, ax=ax, shrink=0.8, label=cbar)
    return im


def hillshade(dem, azimuth=315.0, altitude=35.0, exaggeration=10.0):
    """Grey-scale hillshade (0-1) of a (y, x) DEM DataArray."""
    import numpy as np
    z = np.asarray(dem, float) * exaggeration
    dx, dy = abs(float(dem.x[1] - dem.x[0])), abs(float(dem.y[1] - dem.y[0]))
    gy, gx = np.gradient(z, dy, dx)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(azimuth), np.radians(altitude)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return dem.copy(data=np.clip(hs, 0, 1))
