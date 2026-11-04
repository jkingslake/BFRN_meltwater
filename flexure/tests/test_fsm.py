"""Tests of the fill-spill-merge implementation (flexure/fsm.py)."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
import fsm  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))


def bowl(n=81, depth=5.0, width=0.35):
    x = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, x)
    return 10 - depth * np.exp(-(X**2 + Y**2) / width**2)


def level_for_volume(z, vol):
    """Brute force: flat level whose volume over z equals vol (single basin)."""
    lo, hi = z.min(), z.max()
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        v = np.clip(mid - z, 0, None).sum()
        lo, hi = (mid, hi) if v < vol else (lo, mid)
    return 0.5 * (lo + hi)


def two_basins():
    """Two overlapping bowls (the left one deeper); the saddle between them lies
    on the line joining their centres (row y = 0)."""
    n = 101
    x = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, x)
    z = 10 - 6 * np.exp(-((X + 0.4) ** 2 + Y**2) / 0.05) - 4 * np.exp(-((X - 0.4) ** 2 + Y**2) / 0.05)
    row = z[n // 2]
    i_l, i_r = np.argmin(row[: n // 2]), n // 2 + np.argmin(row[n // 2:])
    saddle = row[i_l:i_r + 1].max()
    return X, Y, z, saddle


def test_single_bowl_level_and_conservation():
    z = bowl()
    ds = fsm.route(z, 0.05)
    d = ds.water_depth.values
    assert np.isclose(d.sum(), 0.05 * z.size)              # closed grid: nothing lost
    ws = (z + d)[d > 0]
    assert ws.max() - ws.min() < 1e-9                       # flat lake surface
    assert np.isclose(ws[0], level_for_volume(z, 0.05 * z.size), atol=1e-6)


def test_water_spills_into_the_neighbour_then_merges():
    X, Y, z, saddle = two_basins()
    h = fsm.build_hierarchy(z)
    left_basin = (X < 0) & (z < saddle)
    cap_left = (saddle - z[left_basin]).sum()
    source = X < -0.2                                   # well inside the left catchment
    # (a) a little water on the left only: it stays there
    w = np.where(source, 0.2 * cap_left / source.sum(), 0.0)
    d = fsm.route(z, w, hierarchy=h).water_depth.values
    assert d[X > 0].sum() == 0 and np.isclose(d.sum(), w.sum())
    # (b) more than the left basin holds: it fills to the saddle, the rest goes right
    vol = 1.5 * cap_left
    w = np.where(source, vol / source.sum(), 0.0)
    d = fsm.route(z, w, hierarchy=h).water_depth.values
    ws_left = (z + d)[(X < 0) & (d > 0)]
    assert np.allclose(ws_left, saddle, atol=1e-9)
    # (the test's count of "left basin" cells differs from the model's by a
    # few cells beside the saddle, hence the tolerance; total water is exact)
    assert np.isclose(d[X > 0].sum(), vol - cap_left, rtol=1e-3)
    assert np.isclose(d.sum(), vol)
    # (c) enough to fill both: one merged lake above the saddle
    ds = fsm.route(z, 2.0, hierarchy=h)
    d = ds.water_depth.values
    ws = (z + d)[d > 0]
    assert ws.min() > saddle and np.ptp(ws) < 1e-9
    assert len(np.unique(ds.lake.values[d > 0])) == 1


def test_edges_and_sinks_drain():
    z = bowl()
    ds = fsm.route(z, 0.05, edges="drain")
    assert ds.attrs["volume_lost_m3"] > 0
    assert np.isclose(ds.attrs["volume_in_m3"], ds.attrs["volume_stored_m3"] + ds.attrs["volume_lost_m3"])
    sink = np.zeros(z.shape, bool)
    sink[z.shape[0] // 2, z.shape[1] // 2] = True            # a moulin at the bottom of the bowl
    ds = fsm.route(z, 0.05, sinks=sink)
    assert ds.water_depth.values.sum() < 1e-9


def test_random_terrain_invariants():
    rng = np.random.default_rng(1)
    from scipy.ndimage import gaussian_filter
    for seed in range(3):
        z = gaussian_filter(rng.normal(size=(120, 140)), 3) * 20 + 50
        w = rng.uniform(0, 0.3, z.shape)
        ds = fsm.route(z, w, edges="drain")
        d = ds.water_depth.values
        a = ds.attrs
        assert np.isclose(a["volume_in_m3"], a["volume_stored_m3"] + a["volume_lost_m3"], rtol=1e-9)
        # every lake is flat and cannot spill: no dry neighbour lies below its surface
        lab = ds.lake.values
        ws = z + d
        for L in np.unique(lab[lab >= 0]):
            m = lab == L
            assert np.ptp(ws[m]) < 1e-7
            level = ws[m][0]
            ii, jj = np.nonzero(m)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    a_, b_ = np.clip(ii + di, 0, z.shape[0] - 1), np.clip(jj + dj, 0, z.shape[1] - 1)
                    dry = d[a_, b_] == 0
                    assert np.all(z[a_, b_][dry] >= level - 1e-7)


def test_hierarchy_reuse_gives_same_answer():
    z = bowl()
    h = fsm.build_hierarchy(z)
    a = fsm.route(z, 0.1, hierarchy=h).water_depth.values
    b = fsm.route(z, 0.1).water_depth.values
    assert np.array_equal(a, b)


REF_DEM = os.path.join(REPO, "python/notebooks/rema_subsets/dem_small_2.tif")
REF_WATER = os.path.join(REPO, "python/notebooks/rema_subsets/water_input_file_2.tif")
REF_OUT = os.path.join(REPO, "python/notebooks/fsm_results/rema_tests/test-wtd.tif")


@pytest.mark.skipif(not all(os.path.exists(p) for p in (REF_DEM, REF_WATER, REF_OUT)),
                    reason="reference RichDEM run not available")
def test_matches_richdem_fill_spill_merge():
    """RichDEM's own fill-spill-merge output on a REMA subset (32 m grid)."""
    tifffile = pytest.importorskip("tifffile")
    z = tifffile.imread(REF_DEM).astype(float)
    w = tifffile.imread(REF_WATER).astype(float)
    ref = tifffile.imread(REF_OUT)
    d = fsm.route(z, w, dx=32.0).water_depth.values
    assert np.isclose(d.sum(), ref.sum(), rtol=1e-6)                  # same stored volume
    wet = (d > 1e-6) | (ref > 1e-6)
    assert np.mean(np.abs(d - ref)[wet] > 1e-3) < 0.02                 # >98% of wet cells agree
    assert np.abs(d - ref).sum() / ref.sum() < 2e-3                    # volume misplaced < 0.2%
