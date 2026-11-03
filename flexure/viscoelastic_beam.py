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
Ice is elastic with Poisson's ratio ``nu`` (~0.3) but creeps incompressibly,
i.e. its viscous part has Poisson's ratio ``nu_v = 1/2``.  A single spring and
dashpot cannot have two Poisson ratios, so the bending moment splits into two
parts (the isotropic and deviatoric parts of the plate's moment tensor), each
a Maxwell element with its own stiffness and relaxation time:

    equilibrium  :   (M1 + M2)'' + k w = q                 (k = rho_sw * g)
    Maxwell laws :   dMi/dt = Di d(w'')/dt - Mi / tau_i,   i = 1, 2

    D1 = D (1 + nu)/2,   tau_1 = 4 eta (1 + nu_v)(1 - nu) / E   (= 6 eta (1-nu)/E)
    D2 = D (1 - nu)/2,   tau_2 = 4 eta (1 - nu_v)(1 + nu) / E   (= 2 eta (1+nu)/E)

Together:   (D d^4/dx^4 + k) dw/dt = dq/dt + d^2/dx^2 (M1/tau_1 + M2/tau_2).

With nu_v = nu the two times coincide (tau = 4 eta (1 - nu^2)/E) and this is
the single-element "master equation"  (D d^4 + k) dw/dt = dq/dt + (q - k w)/tau.
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
    flexure, and the viscosity sets relaxation times of a few days.
    ``nu`` is the elastic Poisson's ratio and ``nu_v`` that of the viscous
    (creep) part: 0.5 for incompressible creep.  Setting ``nu_v = nu`` gives
    the simpler model with a single relaxation time.
    """

    H: float | np.ndarray = 250.0  # ice thickness [m] (scalar or array on the grid)
    E: float = 1.0e9               # effective Young's modulus [Pa]
    nu: float = 0.3                # elastic Poisson's ratio [-]
    eta: float = 1.0e14            # ice (shear) viscosity [Pa s]
    nu_v: float = 0.5              # Poisson's ratio of viscous creep [-] (0.5 = incompressible)
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
        """Viscous flexural rigidity (plane strain)  Lambda = eta H^3 / 3   [N m s]."""
        return self.eta * np.asarray(self.H) ** 3 / 3.0

    @property
    def tau(self):
        """Mean relaxation time  Lambda / D = 4 eta (1 - nu^2) / E  [s].

        This is the rigidity-weighted mean (D1 tau_1 + D2 tau_2) / D of the two
        element times, and the single relaxation time when nu_v = nu."""
        return self.Lam / self.D

    @property
    def weights(self):
        """Elastic weights of the two moment parts, D_i / D = ((1+nu)/2, (1-nu)/2)."""
        return np.array([(1.0 + self.nu) / 2.0, (1.0 - self.nu) / 2.0])

    @property
    def taus(self):
        """Relaxation times of the two moment parts [s]:
        tau_1 = tau (1 + nu_v)/(1 + nu)  (isotropic part),
        tau_2 = tau (1 - nu_v)/(1 - nu)  (deviatoric part)."""
        return np.array([self.tau * (1.0 + self.nu_v) / (1.0 + self.nu),
                         self.tau * (1.0 - self.nu_v) / (1.0 - self.nu)])

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
    M: np.ndarray       # total bending moment M1 + M2 [N]
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

    The state is the deflection w and the two parts of the bending moment,
    ``Mp`` of shape (2, n); the total moment is ``Mp.sum(0)``.

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
        # the two Maxwell elements: stiffnesses (2, n) and relaxation times (2, n)
        self.Dp = self.shelf.weights[:, None] * self.D[None, :]
        self.taup = (self.shelf.taus.reshape(2, -1) * np.ones((1, self.n))).astype(float)
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
        """Factorise  (Lstar [c1 D1 + c2 D2] L + k I)  for a time step dt (cached).

        ``c_i = 1 / (1 + dt / 2 tau_i)`` comes from the trapezoidal rule applied
        to each Maxwell element; ``dt = 0`` gives the purely elastic operator.
        Returns (lu, S, c, a) with c, a of shape (2, n).
        """
        key = float(dt)
        if key not in self._lu_cache:
            a = dt / self.taup
            c = 1.0 / (1.0 + 0.5 * a)
            S = self.Lstar @ sp.diags((c * self.Dp).sum(axis=0)) @ self.L   # stiffness over a step
            A = (S + self.k * sp.identity(self.n)).tocsc()
            idx = np.flatnonzero(self.active)
            lu = splu(A[idx][:, idx])
            self._lu_cache[key] = (lu, S.tocsr(), c, a)
        return self._lu_cache[key]

    def _solve(self, lu, rhs):
        w = np.zeros(self.n)
        w[self.active] = lu.solve(rhs[self.active])
        return w

    def split_moment(self, M):
        """Split a total moment that is in elastic proportion into its parts."""
        return self.shelf.weights[:, None] * np.asarray(M)[None, :]

    # -- the two limits -----------------------------------------------------
    def elastic(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Instantaneous (t = 0+) elastic response: (D w'''' + k w = q).
        Returns (w, M) with M the total moment."""
        lu, S, _, _ = self._step_matrices(0.0)
        w = self._solve(lu, np.asarray(q, dtype=float))
        return w, self.D * (self.L @ w)

    def isostatic(self, q: np.ndarray) -> np.ndarray:
        """Fully relaxed (t -> infinity) response: local isostasy w = q / k."""
        w = np.asarray(q, dtype=float) / self.k
        return np.where(self.active, w, 0.0)

    # -- one time step ------------------------------------------------------
    def step(self, w0, Mp0, q1, dt):
        """Advance (w0, Mp0) by dt to the new load q1.  Returns (w1, Mp1).

        ``Mp0`` holds the two moment parts, shape (2, n).  The trapezoidal rule
        on each Maxwell law  dMi/dt = Di L dw/dt - Mi/tau_i  gives
            Mi1 = ci [ (1 - ai/2) Mi0 + Di L (w1 - w0) ],   ai = dt/tau_i, ci = 1/(1 + ai/2)
        Substituting into equilibrium  Lstar (M11 + M21) + k w1 = q1  gives one
        linear solve for w1:
            (Lstar [sum ci Di] L + k) w1 = q1 + (Lstar [sum ci Di] L) w0
                                              - Lstar sum ci (1 - ai/2) Mi0
        """
        lu, S, c, a = self._step_matrices(dt)
        rhs = q1 + S @ w0 - self.Lstar @ (c * (1.0 - 0.5 * a) * Mp0).sum(axis=0)
        w1 = self._solve(lu, rhs)
        Mp1 = c * ((1.0 - 0.5 * a) * Mp0 + self.Dp * (self.L @ (w1 - w0))[None, :])
        return w1, Mp1

    # -- a whole simulation -------------------------------------------------
    def run(self, t: Sequence[float], load: Callable[[float], np.ndarray] | np.ndarray,
            w0: np.ndarray | None = None, Mp0: np.ndarray | None = None) -> Result:
        """Integrate from t[0] to t[-1].

        ``load`` is either a function ``q(t)`` returning the load [Pa] on the
        grid, or an array of shape (len(t), len(x)).  The shelf starts unloaded
        and at rest (w = M = 0) unless ``w0``/``Mp0`` are given; the load at
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
            W[0], M = self.elastic(Q[0])
            Mp = self.split_moment(M)
        else:
            W[0] = w0
            Mp = Mp0 if Mp0 is not None else self.Dp * (self.L @ w0)[None, :]
        Mm[0] = Mp.sum(axis=0)

        for i in range(1, nt):
            Q[i] = qfun(t[i])
            W[i], Mp = self.step(W[i - 1], Mp, Q[i], t[i] - t[i - 1])
            Mm[i] = Mp.sum(axis=0)

        return Result(self.x, t, Q, W, Mm, self.shelf.surface_stress(Mm))


# ---------------------------------------------------------------------------
# Exact Fourier-mode solutions (used for verification and for intuition)
# ---------------------------------------------------------------------------
def maxwell_modes(K4, shelf: IceShelf):
    """Per-mode matrices of the two-element Maxwell model.

    In a Fourier mode with K^4 = (k_x^2 + k_y^2)^2 the two moment potentials
    mu = (mu1, mu2) obey
        dmu/dt = (d / A) dq/dt - G mu,       w = (q - K^4 (mu1 + mu2)) / k,
    with d = (D1, D2), A = D K^4 + k and
        G = T^-1 - (K^4 / A) d (1^T T^-1),   T = diag(tau_1, tau_2).
    Returns (d, A, G) with G of shape K4.shape + (2, 2) (float64).
    """
    K4 = np.asarray(K4, dtype=float)
    D, kf = float(shelf.D), shelf.k
    d = shelf.weights * D
    ti = 1.0 / shelf.taus
    A = D * K4 + kf
    r = (K4 / A)[..., None, None]
    G = np.diag(ti) - r * (d[:, None] * ti[None, :])
    return d, A, G


def _phi1(z):
    """(1 - exp(-z)) / z, stable at z = 0."""
    z = np.asarray(z, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(z) < 1e-8, 1.0 - z / 2, -np.expm1(-z) / z)


def _psi(z):
    """(1 - exp(-z)(1 + z)) / z^2, stable at z = 0 (value 1/2)."""
    z = np.asarray(z, dtype=float)
    small = np.abs(z) < 1e-3
    zs = np.where(small, 1.0, z)
    direct = (-np.expm1(-zs) - zs * np.exp(-zs)) / zs**2
    series = 0.5 - z / 3 + z**2 / 8 - z**3 / 30
    return np.where(small, series, direct)


def expm_2x2(G, t):
    """exp(-G t) and  int_0^t exp(-G s) ds  for a batch of real 2x2 matrices
    with real eigenvalues, via the Newton (divided-difference) form
        f(G) = f(l2) I + f[l1, l2] (G - l2 I),
    which stays accurate when the two eigenvalues l1 >= l2 nearly coincide."""
    G = np.asarray(G, dtype=float)
    m = 0.5 * (G[..., 0, 0] + G[..., 1, 1])
    r = np.sqrt(np.maximum(0.25 * (G[..., 0, 0] - G[..., 1, 1]) ** 2 + G[..., 0, 1] * G[..., 1, 0], 0.0))
    l2, dl = m - r, 2 * r
    I = np.eye(2)
    Gm = G - l2[..., None, None] * I
    # f1(l) = exp(-l t)
    f1 = np.exp(-l2 * t)
    df1 = -t * f1 * _phi1(dl * t)
    E = f1[..., None, None] * I + df1[..., None, None] * Gm
    # f2(l) = (1 - exp(-l t)) / l = t phi1(l t)
    f2a, f2b = t * _phi1(l2 * t), t * _phi1((l2 + dl) * t)
    with np.errstate(divide="ignore", invalid="ignore"):
        df2 = np.where(dl * t > 1e-5, (f2b - f2a) / np.where(dl > 0, dl, 1.0),
                       -t**2 * _psi((l2 + 0.5 * dl) * t))
    F = f2a[..., None, None] * I + df2[..., None, None] * Gm
    return E, F


def relaxation_times(wavenumber, shelf: IceShelf):
    """The two relaxation times [s] of a sinusoid of wavenumber k_x [rad/m],
    returned as an array (slow, fast).

    With nu_v = nu the slow time is  tau (1 + (k_x ell)^4)  and the fast mode
    never moves the deflection.  With nu_v = 1/2 the slow time stays within
    ~15% of that and carries most of the relaxation; the fast one is ~3 days.
    """
    K4 = np.asarray(wavenumber, dtype=float) ** 4
    _, _, G = maxwell_modes(K4, shelf)
    m = 0.5 * (G[..., 0, 0] + G[..., 1, 1])
    r = np.sqrt(np.maximum(0.25 * (G[..., 0, 0] - G[..., 1, 1]) ** 2 + G[..., 0, 1] * G[..., 1, 0], 0.0))
    return np.stack([1.0 / (m - r), 1.0 / (m + r)])


def relaxation_time(wavenumber, shelf: IceShelf):
    """The slow (dominant) relaxation time of a sinusoid of wavenumber k_x:
    tau (1 + (k_x ell)^4) exactly when nu_v = nu, close to it otherwise."""
    return relaxation_times(wavenumber, shelf)[0]


def modal_step_response(wavenumber, shelf: IceShelf, t):
    """Deflection of a sinusoid of wavenumber k_x, relative to its isostatic
    value, when a load is switched on at t = 0 and held.  Shape (len(t), ...)."""
    K4 = np.asarray(wavenumber, dtype=float) ** 4
    d, A, G = maxwell_modes(K4, shelf)
    mu0 = d * (1.0 / A)[..., None]                  # elastic potentials for unit load
    out = []
    for tt in np.atleast_1d(t):
        E, _ = expm_2x2(G, tt)
        mu = np.einsum("...ij,...j->...i", E, mu0)
        out.append(1.0 - K4 * mu.sum(axis=-1))       # k w / q
    return np.array(out)


def spectral_step_response(x, q, t, shelf: IceShelf):
    """Exact response of a *periodic* beam to a load ``q`` switched on at t = 0.

    Each Fourier mode jumps elastically and then relaxes exactly (a 2x2 matrix
    exponential per mode) towards q_hat / k.  Returns w, shape (len(t), len(x)).
    """
    x = np.asarray(x)
    n = x.size
    dx = x[1] - x[0]
    kx = 2 * np.pi * np.fft.rfftfreq(n, dx)
    q_hat = np.fft.rfft(q)
    resp = modal_step_response(kx, shelf, t)        # (nt, nk), real
    return np.fft.irfft(resp * q_hat[None, :] / shelf.k, n=n, axis=1)


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
    t1, t2 = shelf.taus / SECONDS_PER_DAY
    print(f"relaxation times      = {t1:.2f} and {t2:.2f} days "
          f"(mean {shelf.tau / SECONDS_PER_DAY:.2f}; nu = {shelf.nu}, nu_v = {shelf.nu_v})")
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
