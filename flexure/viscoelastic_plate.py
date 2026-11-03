"""
viscoelastic_plate.py
=====================

The map-plane (x, y) extension of ``viscoelastic_beam.py``: a thin Maxwell
viscoelastic *plate* floating on a hydrostatic foundation.  The beam's
d^4/dx^4 becomes the biharmonic operator and everything else carries over:

    (D lap^2 + k) dw/dt = dq/dt + (q - k w) / tau,       lap^2 = (d_xx + d_yy)^2

Moments come from a scalar "moment potential" mu (units N m^2), which obeys
the same Maxwell law as the beam's moment and makes the bookkeeping trivial:

    dmu/dt = D dw/dt - mu / tau          (local update, no derivatives)
    lap^2 mu + k w = q                   (force balance)
    M_xx = mu_xx + nu mu_yy,  M_yy = mu_yy + nu mu_xx,  M_xy = (1 - nu) mu_xy

Elastically mu = D w, which gives the usual plate moments.  (Assumption, as in
the beam model: the viscous part of the Maxwell law has the same tensor
structure as the elastic part, so the plate keeps a single relaxation time.)

Two solvers
-----------
SpectralPlate   periodic domain, FFTs.  Each Fourier mode is integrated
                *exactly* in time for loads that vary linearly between the
                given times, so time steps are needed only where the load
                history has corners.  Fastest by far.
FDPlate         finite differences (13-point biharmonic) on a rectangle with
                clamped or periodic edges, plus an optional mask of fixed
                (grounded) nodes for irregular grounding lines.  Crank-Nicolson
                in time, exactly as in the beam model.  Linear solves by sparse
                LU (any geometry) or by conjugate gradients preconditioned with
                a fast sine transform (rectangles).

Conventions as in the beam: w positive down, q positive down, sigma > 0 is
tension at the upper surface.  Arrays are indexed [y, x].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import scipy.fft as sfft
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg, splu

from viscoelastic_beam import IceShelf, SECONDS_PER_DAY  # noqa: F401  (re-exported)


@dataclass
class Result2D:
    t: np.ndarray            # output times [s]
    w: np.ndarray            # deflection, shape (nt, ny, nx) [m]
    q: np.ndarray            # load at the output times [Pa]
    mu: np.ndarray | None    # moment potential (FD solver only) [N m^2]


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

    For each Fourier mode (wavenumber magnitude K) the master equation is the
    scalar ODE  A dW/dt = dQ/dt + (Q - k W)/tau  with  A = D K^4 + k.  If Q
    varies linearly over a step (slope s) the exact update is

        W1 = c0 + s dt / k + (W0 - c0) exp(-lam dt),
        lam = k / (tau A),   c0 = (Q0 + s tau)/k - tau A s / k^2,

    so the scheme is exact and unconditionally stable for any dt.
    """

    def __init__(self, nx: int, ny: int, dx: float, shelf: IceShelf | None = None,
                 dy: float | None = None, dtype=np.float64, workers: int = -1):
        self.shelf = shelf if shelf is not None else IceShelf()
        if np.ndim(self.shelf.H) != 0:
            raise ValueError("SpectralPlate needs a uniform thickness")
        self.nx, self.ny = nx, ny
        self.dx, self.dy = dx, (dy if dy is not None else dx)
        self.dtype = np.dtype(dtype)
        self.cdtype = np.result_type(self.dtype, np.complex64)
        self.workers = workers
        kx = 2 * np.pi * sfft.rfftfreq(nx, self.dx)
        ky = 2 * np.pi * sfft.fftfreq(ny, self.dy)
        self.KX, self.KY = np.meshgrid(kx, ky)               # shape (ny, nx//2 + 1)
        K2 = self.KX**2 + self.KY**2
        self.K4 = (K2**2).astype(self.dtype)
        self.D, self.k, self.tau = float(self.shelf.D), self.shelf.k, float(self.shelf.tau)
        self.A = (self.D * self.K4 + self.k).astype(self.dtype)
        self.lam = (self.k / (self.tau * self.A)).astype(self.dtype)

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
        """Real arrays (E, b0, b1) with  W1 = E W0 + b0 Q0 + b1 Q1  (exact for a
        load linear in time over the step).  Cached per dt; built in float64
        with cancellation-free formulas, then cast to the working precision."""
        key = float(dt)
        cache = self.__dict__.setdefault("_coef", {})
        if key not in cache:
            A = self.D * self.K4.astype(np.float64) + self.k
            x = self.k * dt / (self.tau * A)              # lam * dt
            E = np.exp(-x)
            phi = -np.expm1(-x) / x                       # (1 - E) / x
            one_m_phi = np.where(x < 1e-4, x / 2 - x**2 / 6 + x**3 / 24, 1 - phi)
            b1 = (one_m_phi + phi * self.k / A) / self.k  # weight of the new load
            b0 = (1 - E) / self.k - b1                    # weight of the old load
            cache[key] = tuple(a.astype(self.dtype) for a in (E, b0, b1))
        return cache[key]

    def step_hat(self, w_hat, q0_hat, q1_hat, dt):
        """Advance the Fourier coefficients by dt (load linear in between).

        Solving  A W' = Q' + (Q - k W)/tau  exactly for linear Q(t) gives
        W1 = E W0 + b0 Q0 + b1 Q1  with  E = exp(-dt k / (tau A))."""
        if dt == 0:  # instantaneous change of load: elastic jump
            return w_hat + (q1_hat - q0_hat) / self.A
        E, b0, b1 = self.step_coefficients(dt)
        return E * w_hat + b0 * q0_hat + b1 * q1_hat

    def run(self, t: Sequence[float], load, w0=None, store: Sequence[int] | None = None):
        """Integrate through the times ``t`` (the load is linear in between).

        ``load`` may be
          * a function q(t) -> (ny, nx) array,
          * an array of shape (nt, ny, nx), or
          * a tuple (Q, f) for a separable load q = Q(x, y) f(t) with f a
            function or an array of length nt - the fastest option, since Q
            is transformed only once.
        ``store`` lists the time indices to return (default: all).  Only those
        need an inverse FFT.  Starts from rest with an elastic response to
        q(t[0]) unless w0 is given.
        """
        t = np.asarray(t, dtype=float)
        store = list(range(t.size)) if store is None else list(store)
        if isinstance(load, tuple):
            return self._run_separable(t, *load, w0=w0, store=store)
        elif callable(load):
            q_at = lambda i: load(t[i])
            q_hat_at = lambda i: self.fft(q_at(i))
        else:
            load = np.asarray(load)
            q_at = lambda i: load[i]
            q_hat_at = lambda i: self.fft(load[i])

        W = np.empty((len(store), self.ny, self.nx), dtype=self.dtype)
        Qs = np.empty_like(W)
        q_hat = q_hat_at(0)
        w_hat = q_hat / self.A if w0 is None else self.fft(w0)
        j = 0
        for i in range(t.size):
            if i > 0:
                q1_hat = q_hat_at(i)
                w_hat = self.step_hat(w_hat, q_hat, q1_hat, t[i] - t[i - 1])
                q_hat = q1_hat
            if j < len(store) and store[j] == i:
                W[j] = self.ifft(w_hat)
                Qs[j] = q_at(i)
                j += 1
        return Result2D(t[store], W, Qs, None)

    def _run_separable(self, t, Q, f, w0=None, store=None):
        """q = Q(x, y) f(t): transform Q once; each step is then
        W <- E W + (b0 f0 + b1 f1) Q_hat, two operations on the spectrum."""
        fv = np.array([f(tt) for tt in t]) if callable(f) else np.asarray(f, float)
        Q = np.asarray(Q, dtype=self.dtype)
        Q_hat = self.fft(Q)
        W = np.empty((len(store), self.ny, self.nx), dtype=self.dtype)
        Qs = np.empty_like(W)
        w_hat = Q_hat * (fv[0] / self.A) if w0 is None else self.fft(w0)
        j = 0
        for i in range(t.size):
            if i > 0:
                dt = t[i] - t[i - 1]
                if dt == 0:
                    w_hat = w_hat + Q_hat * ((fv[i] - fv[i - 1]) / self.A)
                else:
                    E, b0, b1 = self.step_coefficients(dt)
                    w_hat = E * w_hat + (b0 * fv[i - 1] + b1 * fv[i]) * Q_hat
            if j < len(store) and store[j] == i:
                W[j] = self.ifft(w_hat)
                Qs[j] = Q * fv[i]
                j += 1
        return Result2D(t[store], W, Qs, None)

    # -- stresses ------------------------------------------------------------
    def surface_stress(self, w, q):
        """Upper-surface stresses (sxx, syy, sxy) [Pa] from w and q.

        Force balance gives the moment potential directly:
        mu_hat = (q_hat - k w_hat) / K^4  (the uniform mode carries no moment).
        """
        nu, H = self.shelf.nu, float(self.shelf.H)
        mu_hat = (self.fft(q) - self.k * self.fft(w))
        with np.errstate(divide="ignore", invalid="ignore"):
            mu_hat = np.where(self.K4 > 0, mu_hat / self.K4, 0.0)
        mxx = -self.KX**2 * mu_hat
        myy = -self.KY**2 * mu_hat
        mxy = -self.KX * self.KY * mu_hat
        f = 6.0 / H**2
        sxx = f * self.ifft(mxx + nu * myy)
        syy = f * self.ifft(myy + nu * mxx)
        sxy = f * (1 - nu) * self.ifft(mxy)
        return sxx, syy, sxy


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
                 fixed=None, solver="lu", cg_rtol=1e-10):
        self.shelf = shelf if shelf is not None else IceShelf()
        if np.ndim(self.shelf.H) != 0:
            raise ValueError("FDPlate needs a uniform thickness")
        self.nx, self.ny, self.h = nx, ny, dx
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
        self.D, self.k, self.tau = float(self.shelf.D), self.shelf.k, float(self.shelf.tau)
        self.solver = solver
        self.cg_rtol = cg_rtol
        self.cg_iterations: list[int] = []
        if solver == "cg" and (fixed is not None or ex != ey):
            raise ValueError("solver='cg' needs a plain rectangle (no mask, same edges)")
        self._cache: dict[float, tuple] = {}

    # -- linear algebra ------------------------------------------------------
    def _system(self, dt):
        key = float(dt)
        if key not in self._cache:
            a = dt / self.tau
            c = 1.0 / (1.0 + 0.5 * a)
            Af = (c * self.D * self.Bih + self.k * sp.identity(self.nx * self.ny))
            Aff = Af.tocsr()[self.free][:, self.free].tocsc()
            if self.solver == "lu":
                lu = splu(Aff, permc_spec="MMD_AT_PLUS_A")
                solve = lu.solve
            else:
                solve = self._cg_solver(Aff.tocsr(), c)
            self._cache[key] = (solve, c, a)
        return self._cache[key]

    def _cg_solver(self, Aff, c):
        """CG preconditioned by the same operator with simply-supported
        (clamped) or exact (periodic) symbols, applied with fast transforms."""
        ex, _ = self.edges
        h = self.h
        if ex == "clamped":
            mx, my = self.nx - 2, self.ny - 2
            lx = -(2 - 2 * np.cos(np.pi * np.arange(1, mx + 1) / (mx + 1))) / h**2
            ly = -(2 - 2 * np.cos(np.pi * np.arange(1, my + 1) / (my + 1))) / h**2
            sym = c * self.D * (ly[:, None] + lx[None, :]) ** 2 + self.k
            shape = (my, mx)

            def prec(r):
                r = r.reshape(shape)
                return sfft.idstn(sfft.dstn(r, type=1) / sym, type=1).ravel()
        else:
            lx = -(2 - 2 * np.cos(2 * np.pi * np.arange(self.nx) / self.nx)) / h**2
            ly = -(2 - 2 * np.cos(2 * np.pi * np.arange(self.ny) / self.ny)) / h**2
            sym = c * self.D * (ly[:, None] + lx[None, :]) ** 2 + self.k
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
        solve, _, _ = self._system(0.0)
        q = np.asarray(q, dtype=float).ravel()
        w = self._solve(solve, q)
        return w.reshape(self.ny, self.nx), (self.D * w).reshape(self.ny, self.nx)

    def step(self, w0, mu0, q1, dt):
        """Crank-Nicolson step, exactly as in the beam model (L* D L -> D lap^2):
            (c D lap^2 + k) w1 = q1 + c D lap^2 w0 - lap^2 [c (1 - a/2) mu0]
            mu1 = c [(1 - a/2) mu0 + D (w1 - w0)]
        Arrays are flattened (ny * nx)."""
        solve, c, a = self._system(dt)
        rhs = q1 + c * self.D * (self.Bih @ w0) - self.Bih @ (c * (1 - 0.5 * a) * mu0)
        w1 = self._solve(solve, rhs, guess=w0)
        mu1 = c * ((1 - 0.5 * a) * mu0 + self.D * (w1 - w0))
        mu1[self.fixed.ravel()] = 0.0
        return w1, mu1

    def run(self, t, load: Callable | np.ndarray, store=None):
        t = np.asarray(t, dtype=float)
        store = list(range(t.size)) if store is None else list(store)
        q_at = (lambda i: np.asarray(load(t[i]), float).ravel()) if callable(load) else \
            (lambda i: np.asarray(load[i], float).ravel())
        shape = (len(store), self.ny, self.nx)
        W, MU, Qs = np.empty(shape), np.empty(shape), np.empty(shape)
        q = q_at(0)
        w, mu = (a.ravel() for a in self.elastic(q.reshape(self.ny, self.nx)))
        j = 0
        for i in range(t.size):
            if i > 0:
                q = q_at(i)
                w, mu = self.step(w, mu, q, t[i] - t[i - 1])
            if j < len(store) and store[j] == i:
                W[j], MU[j], Qs[j] = (a.reshape(self.ny, self.nx) for a in (w, mu, q))
                j += 1
        return Result2D(t[store], W, Qs, MU)

    def force_residual(self, w, mu, q):
        """lap^2 mu + k w - q on the free nodes (should be round-off)."""
        r = self.Bih @ mu.ravel() + self.k * w.ravel() - q.ravel()
        return r[self.free]

    def surface_stress(self, mu):
        """Upper-surface stresses (sxx, syy, sxy) [Pa] from the moment potential."""
        nu, H, h = self.shelf.nu, float(self.shelf.H), self.h
        ex, ey = self.edges
        p = np.pad(mu, ((1, 1), (0, 0)), mode="wrap" if ey == "periodic" else "reflect")
        p = np.pad(p, ((0, 0), (1, 1)), mode="wrap" if ex == "periodic" else "reflect")
        mxx = (p[1:-1, 2:] - 2 * p[1:-1, 1:-1] + p[1:-1, :-2]) / h**2
        myy = (p[2:, 1:-1] - 2 * p[1:-1, 1:-1] + p[:-2, 1:-1]) / h**2
        mxy = (p[2:, 2:] - p[2:, :-2] - p[:-2, 2:] + p[:-2, :-2]) / (4 * h**2)
        f = 6.0 / H**2
        return f * (mxx + nu * myy), f * (myy + nu * mxx), f * (1 - nu) * mxy


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
