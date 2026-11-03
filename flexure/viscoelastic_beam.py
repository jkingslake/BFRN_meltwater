"""
viscoelastic_beam.py
====================

A small, fast, readable model of a floating ice shelf bending under the weight
of surface meltwater.  The shelf is treated as a thin 2-D (x-z, plane-strain)
Maxwell viscoelastic beam resting on a hydrostatic (Winkler) foundation - the
ocean.  See ``derivation.md`` for the physics and the numerics, step by step.

Sign conventions (used everywhere in this folder)
-------------------------------------------------
* ``x``  horizontal distance [m]
* ``w``  vertical deflection, **positive downwards** [m]
         (so the change in surface elevation is ``-w``)
* ``q``  load per unit area, positive downwards [Pa]; for a lake of water
         depth ``d`` this is ``q = rho_w * g * d``
* ``M``  bending moment per unit width [N]; elastically ``M = D * w''``
* ``sigma_top = 6 M / H**2`` is the bending stress at the upper surface,
         positive = tension (this is the stress that opens surface crevasses)

The governing equations
-----------------------
    equilibrium  :   M'' + k w = q                    (k = rho_sw * g)
    Maxwell law  :   dM/dt = D * d(w'')/dt - M / tau  (tau = Lambda / D)

which combine into the single "master equation"

    (D d^4/dx^4 + k) dw/dt = dq/dt + (q - k w) / tau.

At t = 0+ the shelf responds elastically, (D d^4/dx^4 + k) w = q, and as
t -> infinity it relaxes to local isostasy, w = q / k.

Numerics in one paragraph
-------------------------
Second-order finite differences in x.  Two sparse matrices do all the work:
``L`` turns deflection into curvature (and knows the *kinematic* boundary
conditions on w), and ``Lstar`` turns bending moment into net vertical force
(and knows the *dynamic* boundary conditions on M).  Time is advanced with the
trapezoidal rule (Crank-Nicolson), which is second-order accurate and
unconditionally stable.  The system matrix is constant, so it is factorised
once and every time step is a single banded back-substitution - O(N) work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

SECONDS_PER_DAY = 86400.0


# ---------------------------------------------------------------------------
# Material and geometric parameters
# ---------------------------------------------------------------------------
@dataclass
class IceShelf:
    """Physical parameters of the floating ice shelf (SI units).

    The defaults are illustrative values for an Antarctic ice shelf; the
    effective Young's modulus is the ~1 GPa typically inferred from tidal
    flexure, and the viscosity sets a Maxwell time of a few days.
    """

    H: float | np.ndarray = 250.0  # ice thickness [m] (scalar or array on the grid)
    E: float = 1.0e9               # effective Young's modulus [Pa]
    nu: float = 0.3                # Poisson's ratio [-]
    eta: float = 1.0e14            # ice (shear) viscosity [Pa s]
    rho_i: float = 917.0           # ice density [kg m^-3]
    rho_sw: float = 1028.0         # sea-water density [kg m^-3]
    rho_w: float = 1000.0          # melt-water density [kg m^-3]
    g: float = 9.81                # gravity [m s^-2]

    @property
    def D(self):
        """Elastic flexural rigidity  D = E H^3 / (12 (1 - nu^2))   [N m]."""
        return self.E * np.asarray(self.H) ** 3 / (12.0 * (1.0 - self.nu**2))

    @property
    def Lam(self):
        """Viscous flexural rigidity  Lambda = eta H^3 / 3   [N m s]."""
        return self.eta * np.asarray(self.H) ** 3 / 3.0

    @property
    def tau(self):
        """Maxwell relaxation time  tau = Lambda / D = 4 eta (1 - nu^2) / E  [s]."""
        return self.Lam / self.D

    @property
    def k(self) -> float:
        """Foundation (buoyancy) stiffness  k = rho_sw g   [Pa m^-1]."""
        return self.rho_sw * self.g

    @property
    def ell(self):
        """Flexural length  ell = (D / k)^(1/4)   [m]."""
        return (self.D / self.k) ** 0.25

    def lake_load(self, depth):
        """Load [Pa] exerted by a lake of water depth ``depth`` [m]."""
        return self.rho_w * self.g * np.asarray(depth, dtype=float)

    def surface_stress(self, M):
        """Bending stress at the upper surface [Pa], positive = tension."""
        return 6.0 * M / np.asarray(self.H) ** 2


# ---------------------------------------------------------------------------
# The two finite-difference operators
# ---------------------------------------------------------------------------
def curvature_operator(n: int, dx: float, left: str, right: str) -> sp.csr_matrix:
    """``L``: deflection w (nodes) -> curvature kappa = w'' (nodes).

    Interior rows are the usual [1, -2, 1] / dx^2 stencil.  End rows encode
    the *kinematic* boundary conditions on w:

    * ``'clamped'`` (w = 0, w' = 0, e.g. a grounding line): ghost node
      w[-1] = w[1]  ->  kappa[0] = (2 w[1] - 2 w[0]) / dx^2.
    * ``'free'``    (calving front): the moment there is zero, so its
      curvature is never used -> row of zeros (keeps M[0] = 0 for all time).
    """
    main = -2.0 * np.ones(n)
    upper = np.ones(n - 1)
    lower = np.ones(n - 1)
    if left == "clamped":
        upper[0] = 2.0
    else:  # free
        main[0], upper[0] = 0.0, 0.0
    if right == "clamped":
        lower[-1] = 2.0
    else:
        main[-1], lower[-1] = 0.0, 0.0
    return sp.diags([lower, main, upper], [-1, 0, 1], format="csr") / dx**2


def force_operator(n: int, dx: float, left: str, right: str) -> sp.csr_matrix:
    """``Lstar``: bending moment M (nodes) -> net vertical force M'' (nodes).

    Interior rows are again [1, -2, 1] / dx^2.  End rows encode the
    *dynamic* boundary conditions on M:

    * ``'free'``    (M = 0, shear V = M' = 0): ghost node M[-1] = M[1]
      ->  force[0] = (2 M[1] - 2 M[0]) / dx^2.
    * ``'clamped'`` the force balance at a clamped node is replaced by w = 0,
      so this row is never used.
    """
    main = -2.0 * np.ones(n)
    upper = np.ones(n - 1)
    lower = np.ones(n - 1)
    if left == "free":
        upper[0] = 2.0
    if right == "free":
        lower[-1] = 2.0
    return sp.diags([lower, main, upper], [-1, 0, 1], format="csr") / dx**2


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
@dataclass
class Result:
    """Output of :meth:`ViscoelasticBeam.run` (rows = time, columns = x)."""

    x: np.ndarray       # [m]
    t: np.ndarray       # [s]
    q: np.ndarray       # load [Pa]
    w: np.ndarray       # deflection, positive down [m]
    M: np.ndarray       # bending moment [N]
    sigma: np.ndarray   # upper-surface bending stress, + = tension [Pa]

    @property
    def t_days(self) -> np.ndarray:
        return self.t / SECONDS_PER_DAY


class ViscoelasticBeam:
    """Maxwell viscoelastic beam on a hydrostatic foundation.

    Parameters
    ----------
    x : 1-D array of equally spaced node positions [m]
    shelf : :class:`IceShelf` (``H`` may be an array the same length as x)
    left, right : ``'free'`` (calving front / far field) or ``'clamped'``
        (grounding line)

    Example
    -------
    >>> x = np.linspace(-10e3, 10e3, 2001)
    >>> beam = ViscoelasticBeam(x, IceShelf())
    >>> depth = 3.0 * np.clip(1 - (x / 1e3) ** 2, 0, None)       # 3 m deep lake
    >>> t = np.linspace(0, 20, 401) * SECONDS_PER_DAY
    >>> res = beam.run(t, lambda t: beam.shelf.lake_load(depth))  # load from t=0+
    """

    def __init__(self, x: np.ndarray, shelf: IceShelf | None = None,
                 left: str = "free", right: str = "free"):
        for bc in (left, right):
            if bc not in ("free", "clamped"):
                raise ValueError(f"boundary condition must be 'free' or 'clamped', got {bc!r}")
        self.x = np.asarray(x, dtype=float)
        dxs = np.diff(self.x)
        if not np.allclose(dxs, dxs[0], rtol=1e-9):
            raise ValueError("x must be equally spaced")
        self.dx = dxs[0]
        self.n = self.x.size
        self.shelf = shelf if shelf is not None else IceShelf()
        self.left, self.right = left, right

        self.L = curvature_operator(self.n, self.dx, left, right)
        self.Lstar = force_operator(self.n, self.dx, left, right)
        self.D = np.broadcast_to(self.shelf.D, (self.n,)).astype(float)
        self.tau = np.broadcast_to(self.shelf.tau, (self.n,)).astype(float)
        self.k = self.shelf.k

        # unknown (non-clamped) nodes; clamped nodes are held at w = 0
        self.active = np.ones(self.n, dtype=bool)
        if left == "clamped":
            self.active[0] = False
        if right == "clamped":
            self.active[-1] = False
        self._lu_cache: dict[float, tuple] = {}

    # -- building blocks ----------------------------------------------------
    def _step_matrices(self, dt: float):
        """Factorise  (Lstar C D L + k I)  for a time step dt (cached).

        ``C = 1 / (1 + dt / 2 tau)`` comes from the trapezoidal rule applied to
        the Maxwell law; ``dt = 0`` gives the purely elastic operator.
        """
        key = float(dt)
        if key not in self._lu_cache:
            a = dt / self.tau
            c = 1.0 / (1.0 + 0.5 * a)
            S = self.Lstar @ sp.diags(c * self.D) @ self.L          # stiffness over a step
            A = (S + self.k * sp.identity(self.n)).tocsc()
            idx = np.flatnonzero(self.active)
            lu = splu(A[idx][:, idx])
            self._lu_cache[key] = (lu, S.tocsr(), c, a)
        return self._lu_cache[key]

    def _solve(self, lu, rhs):
        w = np.zeros(self.n)
        w[self.active] = lu.solve(rhs[self.active])
        return w

    # -- the two limits -----------------------------------------------------
    def elastic(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Instantaneous (t = 0+) elastic response: (D w'''' + k w = q)."""
        lu, S, _, _ = self._step_matrices(0.0)
        w = self._solve(lu, np.asarray(q, dtype=float))
        return w, self.D * (self.L @ w)

    def isostatic(self, q: np.ndarray) -> np.ndarray:
        """Fully relaxed (t -> infinity) response: local isostasy w = q / k."""
        w = np.asarray(q, dtype=float) / self.k
        return np.where(self.active, w, 0.0)

    # -- one time step ------------------------------------------------------
    def step(self, w0, M0, q1, dt):
        """Advance (w0, M0) by dt to the new load q1.  Returns (w1, M1).

        Trapezoidal rule on the Maxwell law  dM/dt = D L dw/dt - M/tau:
            M1 = c [ (1 - a/2) M0 + D L (w1 - w0) ],   a = dt/tau, c = 1/(1 + a/2)
        Substituting into equilibrium  Lstar M1 + k w1 = q1  gives one linear
        solve for w1:
            (Lstar c D L + k) w1 = q1 + (Lstar c D L) w0 - Lstar c (1 - a/2) M0
        """
        lu, S, c, a = self._step_matrices(dt)
        rhs = q1 + S @ w0 - self.Lstar @ (c * (1.0 - 0.5 * a) * M0)
        w1 = self._solve(lu, rhs)
        M1 = c * ((1.0 - 0.5 * a) * M0 + self.D * (self.L @ (w1 - w0)))
        return w1, M1

    # -- a whole simulation -------------------------------------------------
    def run(self, t: Sequence[float], load: Callable[[float], np.ndarray] | np.ndarray,
            w0: np.ndarray | None = None, M0: np.ndarray | None = None) -> Result:
        """Integrate from t[0] to t[-1].

        ``load`` is either a function ``q(t)`` returning the load [Pa] on the
        grid, or an array of shape (len(t), len(x)).  The shelf starts unloaded
        and at rest (w = M = 0) unless ``w0``/``M0`` are given; the load at
        t[0] is applied elastically at the first instant.
        """
        t = np.asarray(t, dtype=float)
        qfun = load if callable(load) else (lambda _t, _i=iter(np.asarray(load)): next(_i))
        nt = t.size
        Q = np.empty((nt, self.n))
        W = np.empty((nt, self.n))
        Mm = np.empty((nt, self.n))

        Q[0] = qfun(t[0])
        if w0 is None:  # instantaneous elastic response to the initial load
            W[0], Mm[0] = self.elastic(Q[0])
        else:
            W[0] = w0
            Mm[0] = M0 if M0 is not None else self.D * (self.L @ w0)

        for i in range(1, nt):
            Q[i] = qfun(t[i])
            W[i], Mm[i] = self.step(W[i - 1], Mm[i - 1], Q[i], t[i] - t[i - 1])

        return Result(self.x, t, Q, W, Mm, self.shelf.surface_stress(Mm))


# ---------------------------------------------------------------------------
# Exact solutions (used for verification and for intuition)
# ---------------------------------------------------------------------------
def relaxation_time(wavenumber, shelf: IceShelf):
    """Relaxation time of a sinusoidal deflection of wavenumber k_x [rad/m]:

        tau_k = tau (1 + (k_x ell)^4)

    Long wavelengths relax in ~tau; short ones are held up elastically for
    much longer.
    """
    return shelf.tau * (1.0 + (np.asarray(wavenumber) * shelf.ell) ** 4)


def spectral_step_response(x, q, t, shelf: IceShelf):
    """Exact response of a *periodic* beam to a load ``q`` switched on at t = 0.

    Each Fourier mode jumps elastically to q_hat / (D k^4 + k_f) and then
    relaxes exponentially, with time constant tau_k, towards q_hat / k_f.
    Returns w with shape (len(t), len(x)).
    """
    x = np.asarray(x)
    n = x.size
    dx = x[1] - x[0]
    kx = 2 * np.pi * np.fft.rfftfreq(n, dx)
    D, kf = float(shelf.D), shelf.k
    q_hat = np.fft.rfft(q)
    w_el = q_hat / (D * kx**4 + kf)
    w_iso = q_hat / kf
    tau_k = relaxation_time(kx, shelf)
    t = np.atleast_1d(t)[:, None]
    w_hat = w_iso + (w_el - w_iso) * np.exp(-t / tau_k)
    return np.fft.irfft(w_hat, n=n, axis=1)


def hetenyi_strip(x, a, q0, shelf: IceShelf):
    """Elastic deflection of an infinite beam under a uniform strip load q0
    on |x| < a (Hetenyi 1946).  beta = (k / 4D)^(1/4)."""
    beta = (shelf.k / (4 * float(shelf.D))) ** 0.25
    x = np.abs(np.asarray(x, dtype=float))
    u, v = beta * (a + x), beta * np.abs(a - x)
    f = lambda s: np.exp(-s) * np.cos(s)
    inside = 2.0 - f(u) - f(v)
    outside = f(v) - f(u)
    return q0 / (2 * shelf.k) * np.where(x <= a, inside, outside)


def grounding_line_profile(x, q0, shelf: IceShelf):
    """Elastic deflection of a semi-infinite beam clamped at x = 0 under a
    uniform load q0 (the classic tidal-flexure shape):
        w = (q0/k) [1 - exp(-beta x)(cos beta x + sin beta x)]."""
    beta = (shelf.k / (4 * float(shelf.D))) ** 0.25
    bx = beta * np.asarray(x, dtype=float)
    return q0 / shelf.k * (1 - np.exp(-bx) * (np.cos(bx) + np.sin(bx)))


# ---------------------------------------------------------------------------
# Convenience: lake shapes and time histories
# ---------------------------------------------------------------------------
def parabolic_lake(x, centre, half_width, max_depth):
    """Water depth [m] of a bowl-shaped lake."""
    s = (np.asarray(x) - centre) / half_width
    return max_depth * np.clip(1.0 - s**2, 0.0, None)


def fill_hold_drain(t, t_fill, t_hold, t_drain):
    """Fraction of the full lake present at time t [s]: linear filling over
    t_fill, held for t_hold, then drained linearly over t_drain."""
    t = np.asarray(t, dtype=float)
    t1, t2, t3 = t_fill, t_fill + t_hold, t_fill + t_hold + t_drain
    up = np.clip(t / t_fill, 0, 1) if t_fill > 0 else (t >= 0).astype(float)
    down = np.clip((t3 - t) / t_drain, 0, 1) if t_drain > 0 else (t < t2).astype(float)
    return np.where(t <= t2, up, down) * (t >= 0)


if __name__ == "__main__":  # quick demo:  python viscoelastic_beam.py
    shelf = IceShelf()
    print(f"flexural rigidity D   = {shelf.D:.3e} N m")
    print(f"flexural length   ell = {shelf.ell:.0f} m")
    print(f"Maxwell time      tau = {shelf.tau / SECONDS_PER_DAY:.2f} days")
    x = np.linspace(-15e3, 15e3, 3001)
    beam = ViscoelasticBeam(x, shelf)
    q = shelf.lake_load(parabolic_lake(x, 0.0, 1500.0, 4.0))
    t = np.linspace(0, 30, 601) * SECONDS_PER_DAY
    import time
    tic = time.perf_counter()
    res = beam.run(t, lambda _: q)
    toc = time.perf_counter()
    i0 = np.argmin(np.abs(x))
    print(f"{x.size} nodes x {t.size} steps in {1e3 * (toc - tic):.0f} ms")
    print(f"centre deflection: elastic {res.w[0, i0]:.3f} m -> "
          f"after 30 d {res.w[-1, i0]:.3f} m (isostatic {q[i0] / shelf.k:.3f} m)")
