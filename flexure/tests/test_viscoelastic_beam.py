"""Verification of the viscoelastic beam model against exact solutions.

Run from the ``flexure`` folder with:   python -m pytest -q
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from viscoelastic_beam import (  # noqa: E402
    SECONDS_PER_DAY, IceShelf, ViscoelasticBeam, grounding_line_profile,
    hetenyi_strip, parabolic_lake, relaxation_time, relaxation_times, spectral_step_response)

SHELF = IceShelf()
DAY = SECONDS_PER_DAY


def smooth_lake(x, centre=0.0, half_width=1500.0, depth=3.0):
    """A smooth (cos^2) lake so spectral comparisons converge cleanly."""
    s = np.clip((x - centre) / half_width, -1, 1)
    return depth * np.cos(0.5 * np.pi * s) ** 2


def test_parameters_are_sensible():
    assert 1e14 < SHELF.D < 1e16
    assert 400 < SHELF.ell < 1000
    assert 1 * DAY < SHELF.tau < 10 * DAY
    # mean time tau = 4 eta (1 - nu^2) / E; element times for incompressible creep
    E, eta, nu = SHELF.E, SHELF.eta, SHELF.nu
    assert np.isclose(SHELF.tau, 4 * eta * (1 - nu**2) / E)
    assert np.allclose(SHELF.taus, [6 * eta * (1 - nu) / E, 2 * eta * (1 + nu) / E])
    # rigidity-weighted mean of the two element times is tau
    assert np.isclose((SHELF.weights * SHELF.taus).sum(), SHELF.tau)


def test_elastic_strip_load_matches_hetenyi():
    """t = 0+: uniform strip load on a (very long) free-free beam."""
    x = np.linspace(-20e3, 20e3, 8001)
    a, q0 = 1000.0, SHELF.lake_load(2.0)
    beam = ViscoelasticBeam(x, SHELF)
    w, _ = beam.elastic(np.where(np.abs(x) <= a, q0, 0.0))
    exact = hetenyi_strip(x, a, q0, SHELF)
    assert np.max(np.abs(w - exact)) / np.max(exact) < 2e-3


def test_clamped_grounding_line_matches_analytic():
    """Uniform load on a beam clamped at x = 0: the tidal-flexure profile."""
    x = np.linspace(0, 20e3, 4001)
    q0 = SHELF.lake_load(1.0)
    beam = ViscoelasticBeam(x, SHELF, left="clamped", right="free")
    w, _ = beam.elastic(np.full_like(x, q0))
    exact = grounding_line_profile(x, q0, SHELF)
    assert np.max(np.abs(w - exact)) / np.max(exact) < 1e-3


def test_free_edge_line_load_matches_analytic():
    """Line load P at the free end of a semi-infinite beam:
    w = (2 P beta / k) exp(-beta x) cos(beta x)  (Hetenyi 1946)."""
    x = np.linspace(0, 20e3, 4001)
    dx = x[1] - x[0]
    P = 1.0e5  # N per metre of shelf front
    q = np.zeros_like(x)
    q[0] = P / (0.5 * dx)  # end node carries half a cell
    beam = ViscoelasticBeam(x, SHELF, left="free", right="free")
    w, _ = beam.elastic(q)
    beta = (SHELF.k / (4 * SHELF.D)) ** 0.25
    exact = 2 * P * beta / SHELF.k * np.exp(-beta * x) * np.cos(beta * x)
    assert np.max(np.abs(w - exact)) / np.max(exact) < 2e-3


def test_viscoelastic_relaxation_matches_spectral_solution():
    """Step load: finite differences + Crank-Nicolson vs exact Fourier modes."""
    x = np.linspace(-25e3, 25e3, 5001)[:-1]  # periodic grid for the FFT
    q = SHELF.lake_load(smooth_lake(x))
    t = np.linspace(0, 20, 801) * DAY
    beam = ViscoelasticBeam(x, SHELF)
    res = beam.run(t, lambda _: q)
    exact = spectral_step_response(x, q, t, SHELF)
    err = np.max(np.abs(res.w - exact)) / np.max(np.abs(exact))
    assert err < 2e-3


def test_long_time_limit_is_local_isostasy():
    """A broad lake (half-width >> ell) relaxes to w = q/k within ~10 tau.
    (Narrow features relax far more slowly: tau_k = tau (1 + (k ell)^4).)"""
    x = np.linspace(-40e3, 40e3, 1601)
    q = SHELF.lake_load(smooth_lake(x, half_width=8e3))
    t = np.linspace(0, 120, 481) * DAY
    res = ViscoelasticBeam(x, SHELF).run(t, lambda _: q)
    assert np.allclose(res.w[-1], q / SHELF.k, atol=2e-3 * q.max() / SHELF.k)
    assert np.max(np.abs(res.M[-1])) < 1e-2 * np.max(np.abs(res.M[0]))


def test_uniform_load_just_floats_down():
    """A load covering the whole free-free beam causes no bending at all."""
    x = np.linspace(0, 10e3, 501)
    q = np.full_like(x, SHELF.lake_load(1.0))
    res = ViscoelasticBeam(x, SHELF).run(np.linspace(0, 5 * DAY, 11), lambda _: q)
    assert np.allclose(res.w, q[0] / SHELF.k, rtol=1e-10)
    moment_scale = q[0] * SHELF.ell**2
    assert np.max(np.abs(res.M)) < 1e-9 * moment_scale


@pytest.mark.parametrize("left,right", [("free", "free"), ("clamped", "free"),
                                        ("clamped", "clamped")])
def test_force_balance_holds_exactly_every_step(left, right):
    """Discrete equilibrium  Lstar M + k w = q  at every step (to round-off)."""
    x = np.linspace(0, 20e3, 801)
    beam = ViscoelasticBeam(x, SHELF, left, right)
    depth = parabolic_lake(x, 6e3, 1.5e3, 4.0)
    t = np.linspace(0, 15, 151) * DAY
    res = beam.run(t, lambda tt: SHELF.lake_load(depth) * min(tt / (5 * DAY), 1.0))
    act = beam.active
    for i in range(t.size):
        r = beam.Lstar @ res.M[i] + SHELF.k * res.w[i] - res.q[i]
        assert np.max(np.abs(r[act])) < 1e-8 * SHELF.lake_load(4.0)


def test_time_stepping_is_second_order():
    """Halving dt cuts the time-stepping error by ~4 (reference: tiny dt)."""
    x = np.linspace(-20e3, 20e3, 801)
    beam = ViscoelasticBeam(x, SHELF)
    depth = smooth_lake(x)
    T = 12 * DAY
    load = lambda tt: SHELF.lake_load(depth) * np.sin(np.pi * tt / T) ** 2
    ref = beam.run(np.linspace(0, T, 4097), load).w[-1]
    errs = [np.max(np.abs(beam.run(np.linspace(0, T, nt + 1), load).w[-1] - ref))
            for nt in (8, 16, 32, 64)]
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(orders > 1.9), orders


def test_relaxation_time_formula():
    single = IceShelf(nu_v=SHELF.nu)            # equal Poisson ratios: one Maxwell time
    assert np.isclose(relaxation_time(0.0, single), single.tau)
    assert np.isclose(relaxation_time(1 / single.ell, single), 2 * single.tau)
    # incompressible creep: broad features relax with the two element times
    assert np.allclose(relaxation_times(0.0, SHELF), SHELF.taus)


def test_equal_poisson_ratios_reduce_to_single_maxwell_element():
    """nu_v = nu must reproduce the one-relaxation-time model exactly:
    each mode relaxes as exp(-t / (tau (1 + (k ell)^4)))."""
    single = IceShelf(nu_v=SHELF.nu)
    x = np.linspace(-25e3, 25e3, 2001)[:-1]
    q = single.lake_load(smooth_lake(x))
    t = np.linspace(0, 15, 61) * DAY
    kx = 2 * np.pi * np.fft.rfftfreq(x.size, x[1] - x[0])
    qh = np.fft.rfft(q)
    A = single.D * kx**4 + single.k
    tau_k = single.tau * (1 + (kx * single.ell) ** 4)
    closed = np.fft.irfft(qh / single.k + (qh / A - qh / single.k) * np.exp(-t[:, None] / tau_k),
                          n=x.size, axis=1)
    exact = spectral_step_response(x, q, t, single)
    assert np.max(np.abs(exact - closed)) / np.abs(closed).max() < 1e-12
    res = ViscoelasticBeam(x, single).run(t, lambda _: q)
    assert np.max(np.abs(res.w - closed)) / np.abs(closed).max() < 2e-3


def test_two_poisson_ratios_change_stress_more_than_deflection():
    """Incompressible creep barely changes w but does change the moments."""
    x = np.linspace(-15e3, 15e3, 1501)
    q = SHELF.lake_load(parabolic_lake(x, 0, 1500, 4.0))
    t = np.linspace(0, 10, 201) * DAY
    a = ViscoelasticBeam(x, SHELF).run(t, lambda _: q)
    b = ViscoelasticBeam(x, IceShelf(nu_v=SHELF.nu)).run(t, lambda _: q)
    dw = np.abs(a.w - b.w).max() / np.abs(b.w).max()
    dM = np.abs(a.M - b.M).max() / np.abs(b.M).max()
    assert dw < 0.01 and dM > dw
