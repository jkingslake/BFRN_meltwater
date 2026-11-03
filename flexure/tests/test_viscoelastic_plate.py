"""Verification of the 2-D plate solvers and the fast 1-D loops.

Run from the ``flexure`` folder with:   python -m pytest -q
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from viscoelastic_beam import (  # noqa: E402
    SECONDS_PER_DAY, IceShelf, ViscoelasticBeam, fill_hold_drain, parabolic_lake,
    spectral_step_response)
from viscoelastic_plate import FDPlate, SpectralPlate, point_load_plate  # noqa: E402
import fast_beam  # noqa: E402

SHELF = IceShelf()
DAY = SECONDS_PER_DAY


def circular_lake(n, dx, radius=1500.0, depth=4.0, smooth=False):
    x = (np.arange(n) - n / 2) * dx
    X, Y = np.meshgrid(x, x)
    R = np.hypot(X, Y)
    if smooth:
        d = depth * np.cos(0.5 * np.pi * np.clip(R / radius, 0, 1)) ** 2
    else:
        d = depth * np.clip(1 - (R / radius) ** 2, 0, None)
    return R, SHELF.lake_load(d)


def test_point_load_matches_kelvin_function_solution():
    """Elastic plate on a Winkler foundation: w = -P ell^2/(2 pi D) kei(r/ell)."""
    n, dx, P = 512, 40.0, 1.0e7
    q = np.zeros((n, n))
    q[n // 2, n // 2] = P / dx**2
    w = SpectralPlate(n, n, dx, SHELF).elastic(q)
    x = (np.arange(n) - n // 2) * dx
    R = np.hypot(*np.meshgrid(x, x))
    exact = point_load_plate(R, P, SHELF)
    assert abs(w[n // 2, n // 2] / exact[n // 2, n // 2] - 1) < 1e-3
    assert np.max(np.abs(w - exact)[R > 2 * dx]) / exact.max() < 1e-4


def test_plate_with_uniform_load_along_y_is_the_beam():
    """A lake that does not vary in y gives exactly the 1-D (plane-strain) beam."""
    x = np.linspace(-25e3, 25e3, 1251)[:-1]
    s = np.clip(x / 1500, -1, 1)
    q = SHELF.lake_load(4 * np.cos(0.5 * np.pi * s) ** 2)
    t = np.linspace(0, 10 * SHELF.tau, 21)
    exact = spectral_step_response(x, q, t, SHELF)
    res = SpectralPlate(x.size, 4, x[1] - x[0], SHELF).run(t, (np.tile(q, (4, 1)), np.ones(t.size)))
    assert np.max(np.abs(res.w[:, 2, :] - exact)) / np.abs(exact).max() < 1e-12
    fd = FDPlate(x.size, 4, x[1] - x[0], SHELF, edges="periodic").run(t, lambda _: np.tile(q, (4, 1)))
    assert np.max(np.abs(fd.w[:, 2, :] - exact)) / np.abs(exact).max() < 2e-3


def test_spectral_time_integration_is_exact():
    """One huge step gives the same answer as many small ones."""
    n, dx = 128, 50.0
    _, Q = circular_lake(n, dx)
    p = SpectralPlate(n, n, dx, SHELF)
    tf = 25 * DAY
    f = lambda tt: min(tt / tf, 1.0)
    fine = p.run(np.linspace(0, tf, 201), (Q, f), store=[200]).w[-1]
    coarse = p.run([0.0, tf], (Q, f), store=[1]).w[-1]
    assert np.max(np.abs(fine - coarse)) / np.abs(fine).max() < 1e-9


def test_fd_matches_spectral_for_a_circular_lake():
    n, dx = 128, 60.0
    _, Q = circular_lake(n, dx)
    t = np.linspace(0, 6 * SHELF.tau, 41)
    fd = FDPlate(n, n, dx, SHELF, edges="periodic")
    rf = fd.run(t, lambda _: Q)
    sp_ = SpectralPlate(n, n, dx, SHELF)
    rs = sp_.run(t, (Q, np.ones(t.size)))
    assert np.max(np.abs(rf.w - rs.w)) / np.abs(rs.w).max() < 2e-3
    a = sp_.surface_stress(rs.w[0], rs.q[0])[0]
    b = fd.surface_stress(rf.mu[0])[0]
    assert np.max(np.abs(a - b)) / np.abs(a).max() < 5e-3


def test_fd_force_balance_holds_exactly():
    n, dx = 96, 60.0
    _, Q = circular_lake(n, dx)
    fixed = np.zeros((n, n), dtype=bool)
    fixed[:, :20] = True                       # a straight grounding line
    fd = FDPlate(n, n, dx, SHELF, fixed=fixed)
    t = np.linspace(0, 15 * DAY, 31)
    res = fd.run(t, lambda tt: Q * fill_hold_drain(tt, 5 * DAY, 5 * DAY, 0.5 * DAY))
    for i in range(t.size):
        r = fd.force_residual(res.w[i], res.mu[i], res.q[i])
        assert np.max(np.abs(r)) < 1e-8 * Q.max()
    assert np.all(res.w[:, fixed] == 0)


def test_cg_and_lu_agree():
    n, dx = 80, 60.0
    _, Q = circular_lake(n, dx)
    t = np.linspace(0, 3 * SHELF.tau, 9)
    a = FDPlate(n, n, dx, SHELF).run(t, lambda _: Q).w
    b = FDPlate(n, n, dx, SHELF, solver="cg").run(t, lambda _: Q).w
    assert np.max(np.abs(a - b)) / np.abs(a).max() < 1e-8


def test_plate_long_time_limit_is_isostatic():
    """A broad, smooth lake ends up floating isostatically (narrow features
    would take far longer, as in 1-D)."""
    n, dx = 128, 250.0
    _, Q = circular_lake(n, dx, radius=8000.0, smooth=True)
    w = SpectralPlate(n, n, dx, SHELF).run([0, 30 * SHELF.tau], (Q, [1, 1])).w[-1]
    assert np.allclose(w, Q / SHELF.k, atol=2e-3 * Q.max() / SHELF.k)


@pytest.mark.parametrize("left,right", [("free", "free"), ("clamped", "free")])
def test_fast_beam_loops_match_reference(left, right):
    x = np.linspace(0, 20e3, 801)
    beam = ViscoelasticBeam(x, SHELF, left, right)
    Q = SHELF.lake_load(parabolic_lake(x, 8e3, 1500, 4))
    t = np.linspace(0, 20 * DAY, 201)
    f = fill_hold_drain(t, 5 * DAY, 5 * DAY, 0.5 * DAY)
    ref = beam.run(t, lambda tt: Q * fill_hold_drain(tt, 5 * DAY, 5 * DAY, 0.5 * DAY))
    Wb, _ = fast_beam.run_banded(beam, t, Q, f)
    assert np.max(np.abs(Wb - ref.w)) / np.abs(ref.w).max() < 1e-7
    if fast_beam.HAVE_NUMBA:
        Wn, Mn = fast_beam.run_numba(beam, t, Q, f)
        assert np.max(np.abs(Wn - ref.w)) / np.abs(ref.w).max() < 1e-7
        assert np.max(np.abs(Mn - ref.M)) / np.abs(ref.M).max() < 1e-7
