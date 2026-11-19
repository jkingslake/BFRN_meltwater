"""Cross-verification against two independent codes: GMT grdflexure (FFT,
elastic and Maxwell viscoelastic) and gFlex (finite differences, elastic).
Skipped when GMT or gFlex is not installed.  The full study, with
convergence plots, is verification/cross_verification.ipynb.
"""
import os
import shutil
import sys

import numpy as np
import pytest
import xarray as xr

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "verification"))
from viscoelastic_beam import SECONDS_PER_DAY as DAY, IceShelf, ViscoelasticBeam  # noqa: E402
from viscoelastic_plate import FDPlate, SpectralPlate  # noqa: E402

try:
    import gflex
except ImportError:  # pragma: no cover
    gflex = None
needs_gmt = pytest.mark.skipif(shutil.which("gmt") is None, reason="GMT not installed")
needs_gflex = pytest.mark.skipif(gflex is None, reason="gFlex not installed")

H, E, NU = 400.0, 1e9, 0.3


def lake(n, dx):
    x = (np.arange(n) - n / 2) * dx
    X, Y = np.meshgrid(x, x)
    d = 4.0 * np.exp(-(X**2 / 1500.0**2 + Y**2 / 900.0**2) / 2)
    return xr.DataArray(d, coords={"y": x, "x": x}, dims=("y", "x"))


def gflex_w(q, dx, bc, Te=H, bc_ns=None):
    f = gflex.F2D(); f.Quiet = True; f.Method = "FD"; f.PlateSolutionType = "vWC1994"; f.Solver = "direct"
    f.g = 9.81; f.E = E; f.nu = NU; f.rho_m = 1028.0; f.rho_fill = 0.0
    f.Te = Te; f.qs = np.array(q, float); f.dx = f.dy = dx
    f.BC_W = f.BC_E = bc; f.BC_N = f.BC_S = bc_ns or bc
    f.initialize(); f.run(); f.finalize()
    return -f.w


@needs_gmt
def test_grdflexure_elastic_matches_spectral_plate():
    from gmt_flexure import GMT_G, grdflexure
    n, dx = 128, 100.0
    d = lake(n, dx)
    wg = grdflexure([(0.0, d)], [0.0], H=H, E=E, nu=NU)[0].values
    s = IceShelf(H=H, E=E, nu=NU, nu_v=NU, g=GMT_G)
    w = SpectralPlate(n, n, dx, s).elastic(s.lake_load(d.values))
    assert np.abs(wg - w).max() / np.abs(w).max() < 1e-5


@needs_gmt
def test_grdflexure_viscoelastic_history_matches_spectral_plate():
    """Fill in 3 steps, drain at day 8.5: Maxwell plate, t_m = tau, nu_v = nu.
    Steps fall between output times: at an output time equal to a step time,
    grdflexure includes or excludes the step depending on round-off in the age."""
    from gmt_flexure import GMT_G, grdflexure
    n, dx = 128, 100.0
    d = lake(n, dx)
    s = IceShelf(H=H, E=E, nu=NU, nu_v=NU, eta=1e14, g=GMT_G)
    steps = [(0.5 * DAY, 1 / 3), (1.5 * DAY, 1 / 3), (2.5 * DAY, 1 / 3), (8.5 * DAY, -1.0)]
    out = np.arange(1, 13) * DAY
    wg = grdflexure([(t, f * d) for t, f in steps], out, H=H, E=E, nu=NU, tm=s.tau).values
    tt, ff, cum = [0.0], [0.0], 0.0
    for t in sorted(set(out) | {t for t, _ in steps}):
        f = sum(f for ts, f in steps if ts == t)
        tt.append(t); ff.append(cum)
        if f:
            cum += f; tt.append(t); ff.append(cum)
    r = SpectralPlate(n, n, dx, s).run(np.array(tt), (s.lake_load(d.values), np.array(ff)), store_mu=False,
                                       stresses=False)
    w = np.array([r.w.values[int(np.flatnonzero(np.isclose(r.time.values, t)).max())] for t in out])
    assert np.abs(wg - w).max() / np.abs(w).max() < 1e-5


@needs_gflex
def test_gflex_periodic_equals_fdplate():
    n, dx = 96, 100.0
    s = IceShelf(H=H, E=E, nu=NU, nu_v=NU)
    q = s.lake_load(lake(n, dx).values)
    w, _ = FDPlate(n, n, dx, s, edges="periodic").elastic(q)
    assert np.abs(gflex_w(q, dx, "Periodic") - w).max() / np.abs(w).max() < 1e-8


@needs_gflex
def test_gflex_variable_thickness_matches_beam():
    x = np.linspace(0, 30e3, 301)
    Hx = 200.0 + 400.0 * x / 30e3
    s = IceShelf(H=Hx, E=E, nu=NU, nu_v=NU)
    q = s.lake_load(4.0 * np.exp(-((x - 12e3) / 1500.0) ** 2))
    wb, _ = ViscoelasticBeam(x, s, "clamped", "clamped").elastic(q)
    wg = gflex_w(np.tile(q, (8, 1)), 100.0, "0Displacement0Slope", Te=np.tile(Hx, (8, 1)), bc_ns="Periodic")[4]
    assert np.abs(wg - wb).max() / wb.max() < 2e-5


def test_rotated_grounding_line_is_first_order():
    """A stair-stepped clamp: error ~ dx, a few per cent at dx ~ ell/9."""
    s = IceShelf(H=H, E=E, nu=NU, nu_v=NU)
    qs = lambda sn: s.lake_load(4.0 * np.exp(-((sn - 2500.0) / 800.0) ** 2))
    xb = np.linspace(0, 30e3, 30001)
    wb, _ = ViscoelasticBeam(xb, s, "clamped", "free").elastic(qs(xb))
    th, errs = np.radians(30.0), []
    for dx in (200.0, 100.0):
        n = int(20e3 / dx); x = (np.arange(n) - n / 2) * dx; X, Y = np.meshgrid(x, x)
        sn = X * np.cos(th) + Y * np.sin(th) + 4000.0
        fixed = sn < 0
        w, _ = FDPlate(n, n, dx, s, fixed=fixed).elastic(np.where(fixed, 0.0, qs(sn)))
        m = (~fixed) & (np.abs(-X * np.sin(th) + Y * np.cos(th)) < 3000) & (sn < 8000)
        errs.append(np.abs(w[m] - np.interp(sn[m], xb, wb)).max() / wb.max())
    assert errs[1] < 0.03 and 1.6 < errs[0] / errs[1] < 2.6
