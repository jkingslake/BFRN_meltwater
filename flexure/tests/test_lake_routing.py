"""Checks of the routing-flexure glue in ``lake_routing.py``.

The routing tests need the fill-spill-merge wrapper (set FILLSPILLMERGE_PATH);
they are skipped if it cannot be imported.
"""
import os
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import lake_routing as lr  # noqa: E402
from lake_routing import DAY, ShelfPlate, drain_lake, drainage_basins  # noqa: E402
from viscoelastic_beam import IceShelf  # noqa: E402

try:
    lr._import_fsm()
    HAVE_FSM = True
except ImportError:
    HAVE_FSM = False
needs_fsm = pytest.mark.skipif(not HAVE_FSM, reason="fillspillmerge wrapper not importable")

SHELF = IceShelf(H=300.0)


def grid(n=96, dx=100.0):
    x = (np.arange(n) - n / 2 + 0.5) * dx
    y = x[::-1].copy()                                  # north-up, as in a GeoTIFF
    X, Y = np.meshgrid(x, y)
    return xr.DataArray(np.zeros((n, n)), coords={"y": y, "x": x}, dims=("y", "x")), X, Y


def test_load_is_volume_conserving_and_deflection_round_trips():
    g, X, Y = grid()
    depth = g.copy(data=np.where(np.hypot(X, Y) < 2000, 3.0, 0.0))
    p = ShelfPlate(g, xr.ones_like(g, dtype=bool), SHELF, coarsen=2)
    q = p.load(depth)
    assert np.isclose(q.sum() * p.dx**2, SHELF.lake_load(depth).sum() * 100.0**2)
    smooth = np.cos(np.pi * p.xc / 9600)[None, :] * np.ones((p.yc.size, 1))
    back = p.to_grid(smooth)
    assert np.allclose(back.values, np.cos(np.pi * g.x.values / 9600)[None, :], atol=3e-3)


def test_drained_broad_lake_rebounds_towards_isostasy():
    """A lake much wider than ell rises almost isostatically at once, and the
    rest of the rebound follows within a few relaxation times."""
    g, X, Y = grid(128, 125.0)
    R = np.hypot(X, Y)
    depth = g.copy(data=5.0 * np.cos(0.5 * np.pi * np.clip(R / 5000, 0, 1)) ** 2)
    p = ShelfPlate(g, xr.ones_like(g, dtype=bool), SHELF, coarsen=2)
    ds = drain_lake(p, depth, [0.5 * DAY, 60 * DAY], probes={"centre": (0.0, 0.0)})
    f = ds.rebound_fraction
    assert 0.8 < float(f.sel(step_time_days=0.5)) < 0.99
    assert float(f[-1]) > 0.99
    assert np.all(np.diff(f.values) > -1e-9)               # monotone rebound
    iso_centre = float(ds.uplift_isostatic.max())
    assert abs(float(ds.uplift_probe[-1, 0]) / iso_centre - 1) < 0.01
    # the step-by-step series agree with the stored maps
    for t in ds.time.values:
        i = int(np.argmin(abs(ds.step_time.values - t)))
        assert np.isclose(float(ds.uplift.sel(time=t).max()), float(ds.uplift_max[i]))


def test_restore_continues_exactly():
    g, X, Y = grid(64, 100.0)
    depth = g.copy(data=np.where(np.hypot(X, Y) < 1500, 2.0, 0.0))
    p = ShelfPlate(g, xr.ones_like(g, dtype=bool), SHELF, coarsen=2)
    ds = drain_lake(p, depth, [2 * DAY, 3 * DAY], drain_time=0.5 * DAY)
    end = p.w.copy()
    p.restore(ds.sel(time=2 * DAY))
    for _ in range(4):
        p.advance(p.q, 0.25 * DAY)
    assert np.allclose(p.w, end, rtol=1e-10, atol=1e-12)


def test_drainage_basins_follow_fill_and_spill():
    """Two bowls split by a ridge.  A small unlabelled pit inside the left
    bowl spills into it, so it belongs to the left lake."""
    g, X, Y = grid(60, 100.0)
    z = 10 + 0.5 * np.minimum(np.hypot(X + 1500, Y), np.hypot(X - 1500, Y)) / 100
    z[30, 10] -= 3.0                                   # small pit in the left bowl
    surf = g.copy(data=z)
    lakes = np.zeros(z.shape, int)
    lakes[np.hypot(X + 1500, Y) < 200] = 1
    lakes[np.hypot(X - 1500, Y) < 200] = 2
    b = drainage_basins(surf, g.copy(data=lakes))
    inner = np.hypot(X, Y) < 2500
    left, right = (X < -150) & inner, (X > 150) & inner
    assert np.all(b.values[left] == 1) and np.all(b.values[right] == 2)
    assert b.values[30, 10] == 1
    assert np.all(b.values[0, :] == -1)                # edges are outlets
    seeds = np.where(b.values == -1, -1, lakes).astype(np.int64)
    py = lr._flood_python(z, seeds)
    if lr.HAVE_NUMBA:
        nb = lr._flood_numba(z, seeds)
        assert np.array_equal(py[0], nb[0]) and np.array_equal(py[1], nb[1])


def test_overflow_route_leaves_a_full_bowl_over_its_spill_point():
    """A bowl on a slope that falls to the east: once full, it overflows at
    its lowest rim point (east side) and the water runs on downhill to the edge."""
    g, X, Y = grid(60, 100.0)
    z = 20 - 0.002 * X + 3 * np.exp(-((np.hypot(X, Y) - 1200) / 300) ** 2)   # ring-shaped rim
    z[np.hypot(X, Y) < 900] -= 4.0                                             # deep floor
    surf = g.copy(data=z)
    b, rec = drainage_basins(surf, xr.zeros_like(g, dtype=int), return_receivers=True)
    r = lr.flow_route(rec, 0.0, 0.0)
    assert np.all(b.values == -1)                      # only the edge drains anything
    assert float(r.x[-1]) == float(g.x.max())          # it leaves through the low (east) edge
    out = np.flatnonzero(np.hypot(r.x, r.y) > 1500)[0]  # (inside the full bowl the water is flat,
    assert float(r.x[out]) > 0                         #  so the path there is arbitrary)
    assert np.all(np.diff(r.x[out:]) >= 0)             # over the east rim, then straight downslope


@needs_fsm
def test_route_fills_a_closed_bowl_and_conserves_water():
    g, X, Y = grid(80, 50.0)
    z = 5 + (np.hypot(X, Y) / 400) ** 2
    z[[0, -1], :] = z[:, [0, -1]] = 0.0                # ocean ring
    dem = g.copy(data=z)
    melt = xr.zeros_like(dem)
    melt[30:50, 30:50] = 0.5
    d = lr.route(dem, melt)
    assert np.isclose(float(d.sum()), float(melt.sum()), rtol=1e-9)
    wet = d.values > 0
    level = (z + d.values)[wet]
    assert np.ptp(level) < 1e-6                        # one flat lake surface


@needs_fsm
def test_coupled_loop_conserves_water_and_rigid_loop_matches_static_routing():
    g, X, Y = grid(80, 50.0)
    rng = np.random.default_rng(1)
    z = 20 + 0.01 * X + (np.hypot(X - 500, Y) / 300) ** 2 * 0.3 + rng.normal(0, 0.3, X.shape)
    z[[0, -1], :] = z[:, [0, -1]] = 0.0
    dem = g.copy(data=z)
    melt = lambda i: xr.full_like(dem, 0.05)
    t = np.arange(6) * DAY
    rigid = lr.route_with_flexure(dem, melt, t)
    assert np.allclose(rigid.water_depth[-1], lr.route(dem, 0.25), atol=1e-9)
    p = ShelfPlate(dem, xr.ones_like(dem, dtype=bool), SHELF, coarsen=2).start()
    c = lr.route_with_flexure(dem, melt, t, plate=p)
    assert np.allclose(c.water_stored + c.water_lost, c.melt_added, rtol=1e-9)
    assert float(c.max_deflection[-1]) > 0
