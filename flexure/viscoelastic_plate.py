"""
viscoelastic_plate.py
=====================

The map-plane (x, y) extension of ``viscoelastic_beam.py``: a thin Maxwell
viscoelastic *plate* floating on a hydrostatic foundation.  The beam's
d^4/dx^4 becomes the biharmonic operator lap^2 = (d_xx + d_yy)^2.

Ice is elastic with Poisson's ratio nu (~0.3) but creeps incompressibly
(viscous Poisson's ratio nu_v = 1/2).  The plate's moment tensor therefore
splits into an isotropic part and a deviatoric part, each a Maxwell element
with its own relaxation time (tau_1, tau_2; see ``IceShelf.taus``).  Each part
is carried by a scalar "moment potential" (units N m^2):

    dmu_i/dt = D_i dw/dt - mu_i / tau_i      (local updates, no derivatives)
    lap^2 (mu_1 + mu_2) + k w = q            (force balance)
    M = lap(mu_1) I + 2 grad grad mu_2 - lap(mu_2) I

with D_1 = D (1 + nu)/2 and D_2 = D (1 - nu)/2.  Elastically mu_i = D_i w,
which gives the textbook plate moments D[(1 - nu) w_ij + nu lap(w) d_ij]; in
the viscous limit the moments become Lambda[(1 - nu_v) k_ij + nu_v tr(k) d_ij]
with k the curvature *rate* - for nu_v = 1/2 exactly the viscous plate of
MacAyeal et al. (2021) and Bezu et al. (2025).  With nu_v = nu the two
relaxation times coincide and the model has a single relaxation time.

Two solvers
-----------
SpectralPlate   periodic domain, FFTs.  Each Fourier mode is integrated
                *exactly* in time (a 2x2 matrix exponential) for loads that
                vary linearly between the given times, so time steps are
                needed only where the load history has corners.
FDPlate         finite differences (13-point biharmonic) on a rectangle with
                clamped or periodic edges, plus an optional mask of fixed
                (grounded) nodes for irregular grounding lines.  Crank-Nicolson
                in time, exactly as in the beam model.  Linear solves by sparse
                LU (any geometry) or by conjugate gradients preconditioned with
                a fast sine transform (rectangles).

Conventions as in the beam: w positive down, q positive down, sigma > 0 is
tension at the upper surface.  Arrays are indexed [y, x]; the two moment
potentials are stacked along a leading axis of length 2.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import scipy.fft as sfft
import scipy.sparse as sp
import xarray as xr
from scipy.sparse.linalg import LinearOperator, cg, splu

from viscoelastic_beam import (IceShelf, SECONDS_PER_DAY, expm_2x2, maxwell_modes,  # noqa: F401
                               shelf_attrs)

PARTS = ["isotropic", "deviatoric"]


def plate_dataset(plate, t, w, q, mu=None, stresses=True, **attrs) -> xr.Dataset:
    """Package a 2-D run as an ``xarray.Dataset`` with dims (time, y, x).

    Variables: ``w`` (deflection, + down), ``q`` (load), ``mu`` (the two moment
    potentials, dim ``part``) and, with ``stresses``, the upper-surface stresses
    ``sxx``, ``syy``, ``sxy`` and the largest principal stress ``sigma1``."""
    t = np.asarray(t, dtype=float)
    dims = ("time", "y", "x")
    ds = xr.Dataset(
        {"w": (dims, np.asarray(w), {"units": "m", "long_name": "deflection (positive downwards)"}),
         "q": (dims, np.asarray(q), {"units": "Pa", "long_name": "surface load (positive downwards)"})},
        coords={"time": ("time", t, {"units": "s"}),
                "time_days": ("time", t / SECONDS_PER_DAY, {"units": "days"}),
                "x": ("x", plate.x, {"units": "m"}), "y": ("y", plate.y, {"units": "m"})},
        attrs={**shelf_attrs(plate.shelf), **attrs},
    )
    if mu is not None:
        ds["mu"] = (("time", "part", "y", "x"), np.asarray(mu),
                    {"units": "N m^2", "long_name": "moment potentials (isotropic, deviatoric)"})
        ds = ds.assign_coords(part=("part", PARTS))
        if stresses:
            st = [plate.surface_stress(m) for m in np.asarray(mu)]
            names = ("sxx", "syy", "sxy")
            for k, name in enumerate(names):
                ds[name] = (dims, np.stack([s_[k] for s_ in st]),
                            {"units": "Pa", "long_name": f"upper-surface stress {name[1:]} (+ = tension)"})
            s1, s2 = principal_stresses(ds.sxx.values, ds.syy.values, ds.sxy.values)
            ds["sigma1"] = (dims, s1, {"units": "Pa", "long_name": "largest principal surface stress (+ = tension)"})
    return ds


def _grid_coords(n, d, given):
    if given is None:
        return np.arange(n) * d
    given = np.asarray(given, dtype=float)
    if given.size != n:
        raise ValueError("coordinate length does not match the grid")
    return given


def moments_from_potentials(m1_xx, m1_yy, m2_xx, m2_yy, m2_xy):
    """M = lap(mu_1) I + 2 grad grad mu_2 - lap(mu_2) I, from second derivatives."""
    lap1 = m1_xx + m1_yy
    return lap1 + m2_xx - m2_yy, lap1 + m2_yy - m2_xx, 2.0 * m2_xy


def principal_stresses(sxx, syy, sxy):
    """Largest and smallest principal stress (2-D), elementwise."""
    c = 0.5 * (sxx + syy)
    r = np.sqrt((0.5 * (sxx - syy)) ** 2 + sxy**2)
    return c + r, c - r


# ===========================================================================
# Spectral (periodic) solver
# ===========================================================================
class SpectralPlate:
    """Maxwell plate on a periodic nx-by-ny grid, solved mode by mode.

    In each Fourier mode the two moment potentials obey the linear system
    dmu/dt = (d/A) dq/dt - G mu  (see ``viscoelastic_beam.maxwell_modes``), and
    the deflection follows from force balance, w = (q - K^4 (mu_1 + mu_2))/k.
    For a load that varies linearly over a step (slope s) the exact update is

        mu1 = E mu0 + B s,    E = exp(-G dt),   B = [int_0^dt exp(-G t') dt'] d / A,

    so the scheme is exact and unconditionally stable for any dt.
    """

    def __init__(self, nx: int, ny: int, dx: float, shelf: IceShelf | None = None,
                 dy: float | None = None, dtype=np.float64, workers: int = -1, x=None, y=None):
        self.shelf = shelf if shelf is not None else IceShelf()
        if np.ndim(self.shelf.H) != 0:
            raise ValueError("SpectralPlate needs a uniform thickness")
        self.nx, self.ny = nx, ny
        self.dx, self.dy = dx, (dy if dy is not None else dx)
        self.x, self.y = _grid_coords(nx, self.dx, x), _grid_coords(ny, self.dy, y)
        self.dtype = np.dtype(dtype)
        self.workers = workers
        kx = 2 * np.pi * sfft.rfftfreq(nx, self.dx)
        ky = 2 * np.pi * sfft.fftfreq(ny, self.dy)
        self.KX, self.KY = np.meshgrid(kx, ky)               # shape (ny, nx//2 + 1)
        self.K2 = self.KX**2 + self.KY**2
        self.K4 = (self.K2**2).astype(self.dtype)
        self.D, self.k = float(self.shelf.D), self.shelf.k
        self.d, A64, self._G = maxwell_modes(self.K2**2, self.shelf)
        self.A = A64.astype(self.dtype)
        self._coef: dict[float, tuple] = {}

    # -- transforms --------------------------------------------------------
    def fft(self, a):
        return sfft.rfft2(np.asarray(a, dtype=self.dtype), workers=self.workers)

    def ifft(self, a_hat):
        return sfft.irfft2(a_hat, s=(self.ny, self.nx), workers=self.workers)

    # -- limits --------------------------------------------------------------
    def elastic(self, q):
        """Instantaneous response  (D lap^2 + k) w = q."""
        return self.ifft(self.fft(q) / self.A)

    def isostatic(self, q):
        return np.asarray(q) / self.k

    # -- exact modal step ----------------------------------------------------
    def step_coefficients(self, dt):
        """Real arrays (E11, E12, E21, E22, B1, B2) with
        mu_i(t + dt) = sum_j E_ij mu_j(t) + B_i s  for a load slope s.
        Cached per dt; built in float64, then cast to the working precision."""
        key = float(dt)
        if key not in self._coef:
            E, F = expm_2x2(self._G, dt)
            dA = self.d[None, None, :] / (self.D * self.K2[..., None] ** 2 + self.k)
            B = np.einsum("...ij,...j->...i", F, dA)
            arrs = (E[..., 0, 0], E[..., 0, 1], E[..., 1, 0], E[..., 1, 1], B[..., 0], B[..., 1])
            self._coef[key] = tuple(np.ascontiguousarray(a, dtype=self.dtype) for a in arrs)
        return self._coef[key]

    def elastic_potentials(self, q_hat):
        """Moment potentials of the elastic response, mu_i = D_i q_hat / A."""
        return [q_hat * (di / self.A) for di in self.d]

    def deflection_hat(self, mu_hat, q_hat):
        """Force balance:  w_hat = (q_hat - K^4 (mu1 + mu2)) / k."""
        return (q_hat - self.K4 * (mu_hat[0] + mu_hat[1])) / self.k

    def step_hat(self, mu_hat, q0_hat, q1_hat, dt):
        """Advance the two potentials' Fourier coefficients by dt (load linear
        in between).  ``mu_hat`` is a pair of arrays; returns the new pair."""
        m1, m2 = mu_hat
        if dt == 0:  # instantaneous change of load: elastic jump
            j1, j2 = self.elastic_potentials(q1_hat - q0_hat)
            return [m1 + j1, m2 + j2]
        E11, E12, E21, E22, B1, B2 = self.step_coefficients(dt)
        s = (q1_hat - q0_hat) / dt
        return [E11 * m1 + E12 * m2 + B1 * s, E21 * m1 + E22 * m2 + B2 * s]

    def run(self, t: Sequence[float], load, store: Sequence[int] | None = None, store_mu=True,
            stresses=True) -> xr.Dataset:
        """Integrate through the times ``t`` (the load is linear in between).

        ``load`` may be
          * a function q(t) -> (ny, nx) array,
          * an array of shape (nt, ny, nx), or
          * a tuple (Q, f) for a separable load q = Q(x, y) f(t) with f a
            function or an array of length nt - the fastest option, since Q
            is transformed only once.
        ``store`` lists the time indices to return (default: all).  Only those
        need inverse FFTs.  The plate starts from rest with an elastic response
        to q(t[0]).  Returns an ``xarray.Dataset`` (see :func:`plate_dataset`);
        ``store_mu`` keeps the moment potentials and ``stresses`` adds the
        surface stresses computed from them.
        """
        t = np.asarray(t, dtype=float)
        store = list(range(t.size)) if store is None else list(store)
        if isinstance(load, tuple):
            Q, f = load
            fv = np.array([f(tt) for tt in t]) if callable(f) else np.asarray(f, float)
            Q = np.asarray(Q, dtype=self.dtype)
            Q_hat = self.fft(Q)
            q_hat_at = lambda i: Q_hat * fv[i]
            q_at = lambda i: Q * fv[i]
        elif callable(load):
            q_at = lambda i: np.asarray(load(t[i]), dtype=self.dtype)
            q_hat_at = lambda i: self.fft(q_at(i))
        else:
            load = np.asarray(load)
            q_at = lambda i: load[i]
            q_hat_at = lambda i: self.fft(load[i])

        W = np.empty((len(store), self.ny, self.nx), dtype=self.dtype)
        Qs = np.empty_like(W)
        MU = np.empty((len(store), 2, self.ny, self.nx), dtype=self.dtype) if store_mu else None
        q_hat = q_hat_at(0)
        mu_hat = self.elastic_potentials(q_hat)
        j = 0
        for i in range(t.size):
            if i > 0:
                q1_hat = q_hat_at(i)
                mu_hat = self.step_hat(mu_hat, q_hat, q1_hat, t[i] - t[i - 1])
                q_hat = q1_hat
            if j < len(store) and store[j] == i:
                W[j] = self.ifft(self.deflection_hat(mu_hat, q_hat))
                Qs[j] = q_at(i)
                if store_mu:
                    MU[j, 0], MU[j, 1] = self.ifft(mu_hat[0]), self.ifft(mu_hat[1])
                j += 1
        return plate_dataset(self, t[store], W, Qs, MU, stresses=stresses and store_mu,
                             model="SpectralPlate", dx=self.dx, dy=self.dy)

    # -- step by step (for coupling to other models) -------------------------
    def start(self, q0, t0=0.0):
        """Elastic response to q0 from rest at time t0."""
        q_hat = self.fft(q0)
        self.state = {"t": float(t0), "q_hat": q_hat, "mu_hat": self.elastic_potentials(q_hat)}
        return self

    def advance(self, q1, dt):
        """Advance the current state by dt; the load changes linearly to q1."""
        st = self.state
        q1_hat = self.fft(q1)
        st["mu_hat"] = self.step_hat(st["mu_hat"], st["q_hat"], q1_hat, dt)
        st["q_hat"], st["t"] = q1_hat, st["t"] + dt
        return self

    @property
    def w(self):
        """Deflection of the current state (+ down) [m]."""
        st = self.state
        return self.ifft(self.deflection_hat(st["mu_hat"], st["q_hat"]))

    @property
    def mu(self):
        """Moment potentials of the current state, shape (2, ny, nx)."""
        return np.stack([self.ifft(m) for m in self.state["mu_hat"]])

    def snapshot(self, stresses=True) -> xr.Dataset:
        """The current state as a one-time Dataset."""
        st = self.state
        return plate_dataset(self, [st["t"]], self.w[None], self.ifft(st["q_hat"])[None],
                             self.mu[None], stresses=stresses, model="SpectralPlate")

    # -- stresses ------------------------------------------------------------
    def surface_stress(self, mu):
        """Upper-surface stresses (sxx, syy, sxy) [Pa] from the two moment
        potentials ``mu`` (shape (2, ny, nx), e.g. ``result.mu[i]``)."""
        mu = np.asarray(mu)
        m1, m2 = self.fft(mu[0]), self.fft(mu[1])
        kxx, kyy, kxy = -self.KX**2, -self.KY**2, -self.KX * self.KY
        mxx, myy, mxy = moments_from_potentials(kxx * m1, kyy * m1, kxx * m2, kyy * m2, kxy * m2)
        f = 6.0 / float(self.shelf.H) ** 2
        return f * self.ifft(mxx), f * self.ifft(myy), f * self.ifft(mxy)


# ===========================================================================
# Finite-difference solver
# ===========================================================================
def _stencil(n, coeffs, kind):
    """Banded (or, for 'periodic', circulant) matrix from {offset: value}."""
    i = np.arange(n)
    rows, cols, vals = [], [], []
    for off, v in coeffs.items():
        j = i + off
        keep = slice(None) if kind == "periodic" else (j >= 0) & (j < n)
        rows.append(i[keep])
        cols.append(j[keep] % n)
        vals.append(np.full(rows[-1].size, float(v)))
    # duplicates (tiny periodic grids) are summed, as they should be
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(n, n))


def _second_difference(n, h, kind):
    """1-D second difference on n nodes.  'clamped': edge nodes are held at
    w = 0 (their rows are never used); 'periodic': wraps around."""
    return _stencil(n, {-1: 1, 0: -2, 1: 1}, kind) / h**2


def _fourth_difference(n, h, kind):
    """1-D fourth difference.  'clamped' uses the ghost node w[-1] = w[1]
    (zero slope), which turns the 6 into a 7 next to each edge."""
    B = _stencil(n, {-2: 1, -1: -4, 0: 6, 1: -4, 2: 1}, kind).tolil()
    if kind != "periodic":
        B[1, 1] = B[n - 2, n - 2] = 7.0
    return B.tocsr() / h**4


class FDPlate:
    """Maxwell plate by finite differences on an (ny, nx) grid.

    Parameters
    ----------
    nx, ny, dx : grid size and spacing [m] (dy = dx)
    shelf : IceShelf with a uniform thickness
    edges : 'clamped' (w = dw/dn = 0 on the outer edge) or 'periodic',
            or a pair (x_edges, y_edges)
    fixed : optional boolean (ny, nx) mask of nodes held at w = 0, e.g.
            grounded ice.  Because the stencil reaches two nodes, a fixed
            region acts as a clamped boundary (first-order at irregular edges).
    solver : 'lu' (sparse LU, factorised once; any geometry) or 'cg'
            (conjugate gradients with a fast sine/Fourier-transform
            preconditioner; rectangle without a mask)
    """

    def __init__(self, nx, ny, dx, shelf: IceShelf | None = None, edges="clamped",
                 fixed=None, solver="lu", cg_rtol=1e-10, x=None, y=None):
        self.shelf = shelf if shelf is not None else IceShelf()
        if np.ndim(self.shelf.H) != 0:
            raise ValueError("FDPlate needs a uniform thickness")
        self.nx, self.ny, self.h = nx, ny, dx
        self.x, self.y = _grid_coords(nx, dx, x), _grid_coords(ny, dx, y)
        ex, ey = (edges, edges) if isinstance(edges, str) else edges
        self.edges = (ex, ey)
        Tx, Ty = _second_difference(nx, dx, ex), _second_difference(ny, dx, ey)
        Bx, By = _fourth_difference(nx, dx, ex), _fourth_difference(ny, dx, ey)
        Ix, Iy = sp.identity(nx, format="csr"), sp.identity(ny, format="csr")
        # flattening is row-major over [y, x], so x varies fastest: kron(Iy, Bx)
        self.Bih = (sp.kron(Iy, Bx) + 2 * sp.kron(Ty, Tx) + sp.kron(By, Ix)).tocsr()

        fix = np.zeros((ny, nx), dtype=bool)
        if ex == "clamped":
            fix[:, [0, -1]] = True
        if ey == "clamped":
            fix[[0, -1], :] = True
        if fixed is not None:
            fix |= np.asarray(fixed, dtype=bool)
        self.fixed = fix
        self.free = np.flatnonzero(~fix.ravel())
        self.D, self.k = float(self.shelf.D), self.shelf.k
        self.Dp = self.shelf.weights * self.D                 # (D1, D2)
        self.taup = np.asarray(self.shelf.taus, dtype=float)  # (tau_1, tau_2)
        self.solver = solver
        self.cg_rtol = cg_rtol
        self.cg_iterations: list[int] = []
        if solver == "cg" and (fixed is not None or ex != ey):
            raise ValueError("solver='cg' needs a plain rectangle (no mask, same edges)")
        self._cache: dict[float, tuple] = {}

    # -- linear algebra ------------------------------------------------------
    def _system(self, dt):
        """Factorise (sum_i c_i D_i lap^2 + k) on the free nodes (cached per dt)."""
        key = float(dt)
        if key not in self._cache:
            a = dt / self.taup
            c = 1.0 / (1.0 + 0.5 * a)
            Dc = float((c * self.Dp).sum())
            Af = (Dc * self.Bih + self.k * sp.identity(self.nx * self.ny))
            Aff = Af.tocsr()[self.free][:, self.free].tocsc()
            if self.solver == "lu":
                lu = splu(Aff, permc_spec="MMD_AT_PLUS_A")
                solve = lu.solve
            else:
                solve = self._cg_solver(Aff.tocsr(), Dc)
            self._cache[key] = (solve, c, a, Dc)
        return self._cache[key]

    def _cg_solver(self, Aff, Dc):
        """CG preconditioned by the same operator with simply-supported
        (clamped) or exact (periodic) symbols, applied with fast transforms."""
        ex, _ = self.edges
        h = self.h
        if ex == "clamped":
            mx, my = self.nx - 2, self.ny - 2
            lx = -(2 - 2 * np.cos(np.pi * np.arange(1, mx + 1) / (mx + 1))) / h**2
            ly = -(2 - 2 * np.cos(np.pi * np.arange(1, my + 1) / (my + 1))) / h**2
            sym = Dc * (ly[:, None] + lx[None, :]) ** 2 + self.k
            shape = (my, mx)

            def prec(r):
                r = r.reshape(shape)
                return sfft.idstn(sfft.dstn(r, type=1) / sym, type=1).ravel()
        else:
            lx = -(2 - 2 * np.cos(2 * np.pi * np.arange(self.nx) / self.nx)) / h**2
            ly = -(2 - 2 * np.cos(2 * np.pi * np.arange(self.ny) / self.ny)) / h**2
            sym = Dc * (ly[:, None] + lx[None, :]) ** 2 + self.k
            shape = (self.ny, self.nx)

            def prec(r):
                r = r.reshape(shape)
                return np.real(sfft.ifft2(sfft.fft2(r) / sym)).ravel()

        n = Aff.shape[0]
        M = LinearOperator((n, n), matvec=prec, dtype=float)

        def solve(b, x0=None):
            it = [0]

            def cb(_):
                it[0] += 1
            x, info = cg(Aff, b, x0=x0, rtol=self.cg_rtol, M=M, callback=cb, maxiter=500)
            if info != 0:
                raise RuntimeError(f"CG did not converge (info={info})")
            self.cg_iterations.append(it[0])
            return x
        return solve

    def _solve(self, solve, rhs, guess=None):
        w = np.zeros(self.nx * self.ny)
        if self.solver == "cg" and guess is not None:
            w[self.free] = solve(rhs[self.free], guess[self.free])
        else:
            w[self.free] = solve(rhs[self.free])
        return w

    # -- limits and stepping -------------------------------------------------
    def elastic(self, q):
        """Elastic response.  Returns w (ny, nx) and potentials (2, ny, nx)."""
        solve = self._system(0.0)[0]
        q = np.asarray(q, dtype=float).ravel()
        w = self._solve(solve, q)
        mu = self.Dp[:, None] * w[None, :]
        return w.reshape(self.ny, self.nx), mu.reshape(2, self.ny, self.nx)

    def step(self, w0, mu0, q1, dt):
        """Crank-Nicolson step, exactly as in the beam model (L* D L -> D lap^2):
            (sum ci Di lap^2 + k) w1 = q1 + sum ci Di lap^2 w0 - lap^2 sum ci (1 - ai/2) mu_i0
            mu_i1 = ci [(1 - ai/2) mu_i0 + Di (w1 - w0)]
        w and q are flattened (ny * nx); mu has shape (2, ny * nx)."""
        solve, c, a, Dc = self._system(dt)
        h = 1.0 - 0.5 * a
        rhs = q1 + Dc * (self.Bih @ w0) - self.Bih @ (c[0] * h[0] * mu0[0] + c[1] * h[1] * mu0[1])
        w1 = self._solve(solve, rhs, guess=w0)
        dw = w1 - w0
        mu1 = np.stack([c[i] * (h[i] * mu0[i] + self.Dp[i] * dw) for i in range(2)])
        mu1[:, self.fixed.ravel()] = 0.0
        return w1, mu1

    def run(self, t, load: Callable | np.ndarray, store=None, stresses=True) -> xr.Dataset:
        """Integrate through the times ``t`` (Crank-Nicolson; keep dt << tau).
        ``load`` is a function q(t) or an array (nt, ny, nx).  Returns an
        ``xarray.Dataset`` (see :func:`plate_dataset`)."""
        t = np.asarray(t, dtype=float)
        store = list(range(t.size)) if store is None else list(store)
        q_at = (lambda i: np.asarray(load(t[i]), float).ravel()) if callable(load) else \
            (lambda i: np.asarray(load[i], float).ravel())
        N = self.nx * self.ny
        W, Qs = np.empty((len(store), self.ny, self.nx)), np.empty((len(store), self.ny, self.nx))
        MU = np.empty((len(store), 2, self.ny, self.nx))
        q = q_at(0)
        w, mu = self.elastic(q.reshape(self.ny, self.nx))
        w, mu = w.ravel(), mu.reshape(2, N)
        j = 0
        for i in range(t.size):
            if i > 0:
                q = q_at(i)
                w, mu = self.step(w, mu, q, t[i] - t[i - 1])
            if j < len(store) and store[j] == i:
                W[j], Qs[j] = w.reshape(self.ny, self.nx), q.reshape(self.ny, self.nx)
                MU[j] = mu.reshape(2, self.ny, self.nx)
                j += 1
        ds = plate_dataset(self, t[store], W, Qs, MU, stresses=stresses, model="FDPlate",
                           dx=self.h, edges=str(self.edges), solver=self.solver)
        ds["fixed"] = (("y", "x"), self.fixed, {"long_name": "nodes held at w = 0 (grounded or edge)"})
        return ds

    # -- step by step (for coupling to other models) -------------------------
    def start(self, q0, t0=0.0):
        """Elastic response to q0 from rest at time t0."""
        q0 = np.asarray(q0, dtype=float)
        w, mu = self.elastic(q0.reshape(self.ny, self.nx))
        self.state = {"t": float(t0), "q": q0.ravel(), "w": w.ravel(),
                      "mu": mu.reshape(2, self.nx * self.ny)}
        return self

    def advance(self, q1, dt):
        """Advance the current state by dt to the load q1 (Crank-Nicolson)."""
        st = self.state
        q1 = np.asarray(q1, dtype=float).ravel()
        st["w"], st["mu"] = self.step(st["w"], st["mu"], q1, dt)
        st["q"], st["t"] = q1, st["t"] + dt
        return self

    @property
    def w(self):
        return self.state["w"].reshape(self.ny, self.nx)

    @property
    def mu(self):
        return self.state["mu"].reshape(2, self.ny, self.nx)

    def snapshot(self, stresses=True) -> xr.Dataset:
        """The current state as a one-time Dataset."""
        st = self.state
        return plate_dataset(self, [st["t"]], self.w[None], st["q"].reshape(1, self.ny, self.nx),
                             self.mu[None], stresses=stresses, model="FDPlate")

    def force_residual(self, w, mu, q):
        """lap^2 (mu_1 + mu_2) + k w - q on the free nodes (should be round-off)."""
        r = self.Bih @ np.asarray(mu).reshape(2, -1).sum(axis=0) + self.k * np.ravel(w) - np.ravel(q)
        return r[self.free]

    def _second_derivatives(self, f):
        h = self.h
        ex, ey = self.edges
        p = np.pad(f, ((1, 1), (0, 0)), mode="wrap" if ey == "periodic" else "reflect")
        p = np.pad(p, ((0, 0), (1, 1)), mode="wrap" if ex == "periodic" else "reflect")
        fxx = (p[1:-1, 2:] - 2 * p[1:-1, 1:-1] + p[1:-1, :-2]) / h**2
        fyy = (p[2:, 1:-1] - 2 * p[1:-1, 1:-1] + p[:-2, 1:-1]) / h**2
        fxy = (p[2:, 2:] - p[2:, :-2] - p[:-2, 2:] + p[:-2, :-2]) / (4 * h**2)
        return fxx, fyy, fxy

    def surface_stress(self, mu):
        """Upper-surface stresses (sxx, syy, sxy) [Pa] from the two moment
        potentials ``mu`` (shape (2, ny, nx), e.g. ``result.mu[i]``)."""
        mu = np.asarray(mu)
        a_xx, a_yy, _ = self._second_derivatives(mu[0])
        b_xx, b_yy, b_xy = self._second_derivatives(mu[1])
        mxx, myy, mxy = moments_from_potentials(a_xx, a_yy, b_xx, b_yy, b_xy)
        f = 6.0 / float(self.shelf.H) ** 2
        return f * mxx, f * myy, f * mxy


# ===========================================================================
# Exact solution for verification
# ===========================================================================
def point_load_plate(r, P, shelf: IceShelf):
    """Elastic deflection of an infinite plate on a Winkler foundation under a
    point load P [N] (Hertz):  w = -P ell^2 / (2 pi D) kei(r / ell),
    so w(0) = P / (8 sqrt(k D))."""
    from scipy.special import kei
    ell = float(shelf.ell)
    return -P * ell**2 / (2 * np.pi * float(shelf.D)) * kei(np.asarray(r) / ell)
