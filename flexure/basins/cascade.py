"""
basins/cascade.py
=================

A chain of N basins along a flow line.  The head basin receives the
catchment's melt; whatever basin i spills flows into basin i+1 in the same
time step (routing is instantaneous); whatever the last basin spills leaves
the chain.  Each basin has its own width, area, sill height and response
(``basin.Basin``).

Metric: the drainage-system **extent** in a year is the index of the
furthest basin that received water that year (0 = only the head basin).
It equals N if the chain overflowed at its far end.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import xarray as xr

from .basin import Basin, State, advance, capacity, deflection, freeze
from .physics import DAY, YEAR


# ---------------------------------------------------------------------------
# Forcing sequences (annual melt, as a multiple of a reference year)
# ---------------------------------------------------------------------------
def forcing_cycles(n_high=3, n_cool=8, n_cycles=3, high=1.0, cool=0.1, spin_up=0):
    """``spin_up`` cool years, then n_cycles of (n_high high years, n_cool cool years)."""
    one = [high] * n_high + [cool] * n_cool
    return np.array([cool] * spin_up + one * n_cycles, float)


def forcing_constant(n_years=33, value=0.5):
    return np.full(n_years, float(value))


def forcing_stochastic(n_years=33, mean=0.5, cv=0.6, seed=0):
    """Log-normal annual melt with the given mean and coefficient of
    variation (PLACEHOLDER for RACMO-like statistics)."""
    rng = np.random.default_rng(seed)
    s2 = np.log(1.0 + cv**2)
    return mean * rng.lognormal(-0.5 * s2, np.sqrt(s2), n_years)


# ---------------------------------------------------------------------------
# Building a chain
# ---------------------------------------------------------------------------
def make_chain(widths, D0s, H, eta, area_factor=1.0, E=1.0e9, nu=0.3, k_factor=np.pi,
               method="layer", gap=500.0):
    """Basins of the given widths and sill heights, with responses from
    ``physics.basin_response``.  Returns (basins, x), x being the
    along-flow position [m] of each basin's centre (basins separated by
    ``gap``)."""
    widths, D0s = np.broadcast_arrays(np.asarray(widths, float), np.asarray(D0s, float))
    basins = [Basin.from_width(W, H, eta, D0, area=area_factor * W**2, E=E, nu=nu,
                               k_factor=k_factor, method=method) for W, D0 in zip(widths, D0s)]
    x = np.cumsum(widths + gap) - 0.5 * widths - gap
    return basins, x


def random_widths(n, median=1000.0, spread=0.6, wmin=200.0, wmax=5000.0, seed=1):
    """Log-normal widths clipped to [wmin, wmax] (PLACEHOLDER distribution)."""
    rng = np.random.default_rng(seed)
    return np.clip(median * np.exp(spread * rng.standard_normal(n)), wmin, wmax)


# ---------------------------------------------------------------------------
# Running a chain
# ---------------------------------------------------------------------------
def run_cascade(basins, melt, V_ref, t_m=45 * DAY, dt=0.5 * DAY, refreeze=1.0,
                winter_points=4, x=None) -> xr.Dataset:
    """Run a chain of basins for len(melt) years.  The head basin receives
    ``melt[y] * V_ref`` [m^3] in year y, at a constant rate over the first
    t_m of the year.

    Returns an ``xarray.Dataset`` with per-year (dims year, basin) inflow,
    spill and free space at the start of each season; the yearly extent;
    and time series (dims time, basin) of the free space, plug and
    deflection, sampled once a day in the season and ``winter_points``
    times in the winter."""
    N = len(basins)
    states = [State() for _ in basins]
    n_season = max(1, int(round(t_m / dt)))
    dts = t_m / n_season
    melt = np.asarray(melt, float)
    ny = melt.size
    inflow = np.zeros((ny, N)); spill = np.zeros((ny, N)); C0 = np.zeros((ny, N))
    t_rec, C_rec, p_rec, w_rec = [], [], [], []
    T = 0.0
    every = max(1, int(round(DAY / dts)))

    def rec():
        t_rec.append(T / YEAR)
        C_rec.append([capacity(b, s) for b, s in zip(basins, states)])
        p_rec.append([s.p for s in states])
        w_rec.append([deflection(b, s) for b, s in zip(basins, states)])

    rec()
    for y in range(ny):
        C0[y] = [capacity(b, s) for b, s in zip(basins, states)]
        q_head = melt[y] * V_ref / t_m
        for n in range(n_season):
            q = q_head
            for i, b in enumerate(basins):
                inflow[y, i] += q * dts
                states[i], sp = advance(b, states[i], q, dts)
                spill[y, i] += sp
                q = sp / dts
            T += dts
            if (n + 1) % every == 0:
                rec()
        states = [freeze(s, refreeze) for s in states]
        for _ in range(winter_points):
            for i, b in enumerate(basins):
                states[i], _ = advance(b, states[i], 0.0, (YEAR - t_m) / winter_points)
            T += (YEAR - t_m) / winter_points
            rec()
    tiny = 1e-6 * np.array([b.A for b in basins])
    received = inflow > tiny
    extent = np.array([np.flatnonzero(r).max() if r.any() else -1 for r in received])
    extent = np.where(spill[:, -1] > tiny[-1], N, extent)
    ds = xr.Dataset(
        {"inflow": (("year", "basin"), inflow, {"units": "m3"}),
         "spill": (("year", "basin"), spill, {"units": "m3"}),
         "C_start": (("year", "basin"), C0, {"units": "m", "long_name": "free space at the start of the season"}),
         "extent": ("year", extent, {"long_name": "furthest basin receiving water (N = overflowed the chain)"}),
         "melt": ("year", melt),
         "capacity": (("time", "basin"), np.array(C_rec), {"units": "m"}),
         "plug": (("time", "basin"), np.array(p_rec), {"units": "m"}),
         "deflection": (("time", "basin"), np.array(w_rec), {"units": "m"}),
         "W": ("basin", np.array([b.W for b in basins]), {"units": "m"}),
         "A": ("basin", np.array([b.A for b in basins]), {"units": "m2"}),
         "D0": ("basin", np.array([b.D0 for b in basins]), {"units": "m"}),
         "tau": ("basin", np.array([b.tau for b in basins]) / YEAR, {"units": "yr"}),
         "alpha": ("basin", np.array([b.alpha for b in basins])),
         "alpha_e": ("basin", np.array([b.alpha_e for b in basins]))},
        coords={"year": np.arange(ny), "basin": np.arange(N), "time": ("time", np.array(t_rec), {"units": "yr"})},
        attrs={"V_ref_m3": V_ref, "t_m_s": t_m, "refreeze": refreeze})
    if x is not None:
        ds["x"] = ("basin", np.asarray(x, float), {"units": "m", "long_name": "along-flow position"})
    return ds
