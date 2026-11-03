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
