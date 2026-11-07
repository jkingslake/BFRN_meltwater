"""Shared helpers for the experiment scripts: output paths and plot style."""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "outputs"          # git-ignored
FIG = OUT / "figures"

sys.path.insert(0, str(HERE.parents[1]))   # the flexure folder, for plotstyle
import plotstyle as ps  # noqa: E402

ps.apply()


def save(ds, name):
    """Write a Dataset to outputs/<name>.nc (compressed) and return the path."""
    OUT.mkdir(parents=True, exist_ok=True)
    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars if ds[v].dtype.kind in "fiu"}
    path = OUT / f"{name}.nc"
    ds.to_netcdf(path, encoding=enc)
    return path


def savefig(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / f"{name}.png"
    fig.savefig(path)
    return path


def flag(text):
    """Tag for figure captions marking a placeholder assumption."""
    return f"[placeholder: {text}]"


__all__ = ["save", "savefig", "ps", "OUT", "FIG", "flag", "os"]


def km_axis(ax, ticks=(0.2, 0.5, 1, 2, 5)):
    """Log width axis in km with plain tick labels."""
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    ax.set_xscale("log")
    ax.set_xticks(list(ticks))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
