"""Checks of the idealised lake-basin model in ``basins/``.

Run from the flexure folder:   python -m pytest -q tests/test_basins.py
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from basins import physics as ph  # noqa: E402
from basins.basin import (R_IW, Basin, State, advance, critical_Pi, fill_season, gap_analytic,  # noqa: E402
                          overtop_time_analytic, run_years, season_boundary)
from basins.cascade import forcing_cycles, make_chain, run_cascade  # noqa: E402
from viscoelastic_beam import IceShelf, modal_step_response, relaxation_time  # noqa: E402

YEAR, DAY = ph.YEAR, ph.DAY


# ---------------------------------------------------------------------------
# physics
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kH", [0.03, 0.1])
def test_layer_recovers_thin_plate_limit(kH):
    H, eta = 600.0, 1e15
    tau, alpha = ph.layer_modes(kH / H, H, eta)
    assert abs(tau[0] / ph.tau_thin_plate(kH / H, H, eta) - 1) < 3e-3
    assert abs(tau[1] / ph.tau_thickness_mode(H, eta) - 1) < 1e-4
    assert abs(alpha[0] - ph.RHO_W / ph.RHO_SW) < 1e-4
    assert abs(alpha[1] - (ph.RHO_W / ph.RHO_I - ph.RHO_W / ph.RHO_SW)) < 1e-4


@pytest.mark.parametrize("kH", [8.0, 30.0])
def test_layer_recovers_half_space_limit(kH):
    H, eta = 600.0, 1e15
    tau, alpha = ph.layer_modes(kH / H, H, eta)
    assert abs(tau[0] / ph.tau_half_space(kH / H, eta) - 1) < 1e-4
    assert abs(alpha[0] - ph.RHO_W / ph.RHO_I) < 1e-3


def test_layer_final_state_is_fluid_equilibrium():
    kH = np.geomspace(0.05, 20, 15)
    _, alpha = ph.layer_modes(kH / 500.0, 500.0, 1e14)
    assert np.allclose(alpha.sum(-1), ph.RHO_W / ph.RHO_I, rtol=1e-9)


def test_plate_formulae_match_the_flexure_model():
    """tau_M (1 + (k ell)^4) and the elastic share are exactly the single-mode
    behaviour of viscoelastic_beam.py (single relaxation time, nu_v = nu)."""
    H, eta, E = 500.0, 3e14, 2e9
    shelf = IceShelf(H=H, E=E, eta=eta, nu=0.3, nu_v=0.3)
    k = np.array([2e-4, 1e-3, 4e-3])
    assert np.allclose(ph.tau_viscoelastic_plate(k, H, eta, E), relaxation_time(k, shelf), rtol=1e-10)
    elastic = modal_step_response(k, shelf, [0.0])[0]                 # k w / q at t = 0+
    assert np.allclose(ph.alpha_elastic(k, H, E) * ph.RHO_SW / ph.RHO_W, elastic, rtol=1e-10)


def test_glen_viscosity_matches_the_brief():
    eta = ph.glen_viscosity(np.array([1e-3, 1e-2]) / YEAR)
    assert 8e14 < eta[0] < 1.0e15 and 1.5e14 < eta[1] < 2.5e14


# ---------------------------------------------------------------------------
# stage 1: one season
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("alpha,Pi", [(1.0, 1.5), (1.0, 0.7), (1.09, 1.5), (0.98, 2.0)])
def test_gap_and_overtopping_time_match_analytic(alpha, Pi):
    b = Basin(A=1e6, D0=2.0, tau=0.5 * YEAR, alpha=alpha, alpha_e=0.0)
    Q = Pi * b.A * b.D0 / b.tau
    ds = fill_season(b, Q, 4 * b.tau, n_out=101)
    t = ds.time.values
    before = t < min(ds.attrs["t_star"], 4 * b.tau)
    assert np.allclose(ds.capacity.values[before], gap_analytic(t[before], Pi, b.tau, b.D0, alpha), atol=1e-8)
    ta = overtop_time_analytic(Pi, b.tau, alpha)
    if np.isfinite(ta) and ta < 4 * b.tau:
        assert abs(ds.attrs["t_star"] / ta - 1) < 1e-8
    else:
        assert not ds.attrs["overtopped"]
    if alpha == 1.0 and Pi > 1:
        assert np.isclose(ta, b.tau * np.log(Pi / (Pi - 1)))


def test_critical_Pi():
    assert critical_Pi(1.0) == 1.0
    assert abs(critical_Pi(1.09) - 1.2894) < 1e-3
    b = Basin(A=1.0, D0=1.0, tau=1.0, alpha=1.09)
    Pc = critical_Pi(1.09)
    assert fill_season(b, Pc * 1.01, 50.0, n_out=2).attrs["overtopped"]
    assert not fill_season(b, Pc * 0.99, 50.0, n_out=2).attrs["overtopped"]
    assert np.isclose(season_boundary(1e3, 1.09), Pc)
    assert np.isclose(season_boundary(1e-4, 1.0), 1e4, rtol=1e-3)   # tau >> t_m: Q t_m/(A D0) = 1


def test_stepper_matches_solve_ivp_over_a_season():
    b = Basin(A=4e6, D0=1.5, tau=0.2 * YEAR, alpha=1.02, alpha_e=0.3)
    Q, t_m = 2.5 * b.A * b.D0 / (60 * DAY), 60 * DAY
    ref = fill_season(b, Q, t_m, n_out=2)
    s, spilled = State(), 0.0
    for _ in range(240):
        s, sp = advance(b, s, Q, t_m / 240)
        spilled += sp
    assert ref.attrs["overtopped"]
    assert abs(s.h - float(ref.h[-1])) < 1e-6
    assert abs(spilled / ref.attrs["spilled_volume"] - 1) < 1e-6


# ---------------------------------------------------------------------------
# stage 2: many years
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("refreeze", [1.0, 0.6])
def test_mass_is_conserved_through_spill_and_refreeze(refreeze):
    b = Basin.from_width(800.0, 700.0, 3e14, D0=2.0)
    ds = run_years(b, forcing_cycles(3, 4, 2) * 3.0 * b.A, 45 * DAY, refreeze=refreeze)
    stored = b.A * (ds.h + R_IW * ds.p)
    err = np.abs(ds.cum_inflow - stored - ds.cum_spill).max() / ds.cum_inflow[-1]
    assert float(err) < 1e-10
    assert ds.spill.sum() > 0


@pytest.mark.parametrize("alpha", [ph.ALPHA_THIN_PLATE, 1.0, ph.ALPHA_HALF_SPACE])
def test_fully_relaxed_capacity_is_the_ratchet_formula(alpha):
    """After a long rest: C = D0 - (rho_w/rho_i - alpha) M, M the retained water."""
    b = Basin(A=1e6, D0=2.0, tau=0.1 * YEAR, alpha=alpha)
    ds = run_years(b, np.r_[[3.0e6] * 3, [0.0] * 5], 45 * DAY)
    M = float((ds.inflow - ds.spill).sum()) / b.A
    assert abs(float(ds.capacity[-1]) - (b.D0 - (ph.RHO_W / ph.RHO_I - alpha) * M)) < 1e-6


def test_cascade_conserves_water_and_routes_downstream():
    basins, x = make_chain([600.0, 1500.0, 900.0, 2500.0], [2.0, 1.0, 3.0, 2.0], 700.0, 3e14)
    V = 3 * basins[0].A * 2.0
    ds = run_cascade(basins, forcing_cycles(2, 2, 2), V, x=x)
    assert np.allclose(ds.inflow.isel(basin=slice(1, None)).values, ds.spill.isel(basin=slice(None, -1)).values,
                       rtol=1e-12, atol=1e-6)
    assert ds.extent.max() >= 1 and ds.extent.min() >= 0
