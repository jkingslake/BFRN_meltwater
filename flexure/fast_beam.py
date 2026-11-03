"""
fast_beam.py: faster drop-in time loops for ``ViscoelasticBeam``.

Same discretisation (both Maxwell elements) and the same answers as
``ViscoelasticBeam.run``, but the
per-step overhead is removed in two ways:

* ``run_banded``  keeps everything in NumPy/SciPy.  The pentadiagonal matrix is
                  factorised once with LAPACK ``dgbtrf`` and each step calls
                  ``dgbtrs``.  The sparse matrix-vector products become a few
                  array slices.
* ``run_numba``   compiles the whole time loop (banded products plus a
                  pentadiagonal LU solve) with Numba, so a step costs a few
                  passes over the grid with no Python in between.

Both take a separable load  q(x, t) = Q(x) f(t)  (f sampled at the times t).
This covers lake filling and draining histories and keeps the loop tight.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.linalg import lapack

try:
    import numba
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False


def _bands(M, kl, ku):
    """Dense diagonals of a sparse matrix: row d holds diagonal (ku - d)."""
    n = M.shape[0]
    ab = np.zeros((kl + ku + 1, n))
    Md = M.todia()
    for off, data in zip(Md.offsets, Md.data):
        if -kl <= off <= ku:
            ab[ku - off] += data
    return ab


def _band_matvec(ab, x, kl, ku):
    """y = M x for a matrix stored as LAPACK-style bands."""
    n = x.size
    y = ab[ku] * x
    for d in range(1, ku + 1):          # super-diagonals
        y[:-d] += ab[ku - d, d:] * x[d:]
    for d in range(1, kl + 1):          # sub-diagonals
        y[d:] += ab[ku + d, :-d] * x[:-d]
    return y


def _setup(beam, dt):
    """Step matrix S = L* (c1 D1 + c2 D2) L and system matrix A = S + k I, with
    clamped nodes turned into identity rows/columns (their right-hand side is
    zeroed).  c and a have shape (2, n): one row per Maxwell element."""
    a = dt / beam.taup
    c = 1.0 / (1.0 + 0.5 * a)
    S = (beam.Lstar @ sp.diags((c * beam.Dp).sum(axis=0)) @ beam.L).tocsr()
    act = beam.active.astype(float)
    P = sp.diags(act)
    A = (P @ S @ P + sp.diags(beam.k * act + (1.0 - act))).tocsr()
    return S, A, c, a


def run_banded(beam, t, Q, f):
    """Time loop with a once-factorised LAPACK band LU.  Returns (w, M)."""
    t = np.asarray(t, float)
    dt = t[1] - t[0]
    if not np.allclose(np.diff(t), dt):
        raise ValueError("run_banded needs a constant time step")
    S, A, c, a = _setup(beam, dt)
    kl = ku = 2
    ab = _bands(A, kl, ku)
    abf = np.zeros((2 * kl + ku + 1, beam.n))
    abf[kl:] = ab
    lub, piv, info = lapack.dgbtrf(abf, kl, ku)
    assert info == 0
    Sb, Lb, Lsb = _bands(S, 2, 2), _bands(beam.L, 1, 1), _bands(beam.Lstar, 1, 1)
    act = beam.active.astype(float)
    W = np.empty((t.size, beam.n))
    Mm = np.empty_like(W)
    w, M = beam.elastic(Q * f[0])
    Mp = beam.split_moment(M)                       # the two Maxwell elements
    W[0], Mm[0] = w, M
    h = 1.0 - 0.5 * a
    g = c * h
    for i in range(1, t.size):
        rhs = Q * f[i] + _band_matvec(Sb, w, 2, 2) - _band_matvec(Lsb, (g * Mp).sum(axis=0), 1, 1)
        rhs *= act
        w1, info = lapack.dgbtrs(lub, kl, ku, rhs, piv)
        Mp = c * (h * Mp + beam.Dp * _band_matvec(Lb, w1 - w, 1, 1)[None, :])
        w = w1
        W[i], Mm[i] = w, Mp.sum(axis=0)
    return W, Mm


def _rhs_at(j, n, qj, Sb, Lsb, w, gM):
    """Right-hand side at an edge node (the general, bounds-checked stencil)."""
    s = qj + Sb[2, j] * w[j] - Lsb[1, j] * gM[j]
    if j >= 1:
        s += Sb[3, j - 1] * w[j - 1] - Lsb[2, j - 1] * gM[j - 1]
    if j >= 2:
        s += Sb[4, j - 2] * w[j - 2]
    if j + 1 < n:
        s += Sb[1, j + 1] * w[j + 1] - Lsb[0, j + 1] * gM[j + 1]
    if j + 2 < n:
        s += Sb[0, j + 2] * w[j + 2]
    return s


def _loop(Q, f, w0, M10, M20, Sb, Lb, Lsb, act, lo1, lo2, u0, u1, u2, D1, D2,
          c1, a1, c2, a2, store_every):
    """The whole time loop for the two Maxwell elements M1, M2.  Interior loops
    have no branches, so the compiler can vectorise them; the four edge nodes
    use the bounds-checked stencil."""
    n = w0.size
    nt = f.size
    nout = (nt - 1) // store_every + 1
    W = np.empty((nout, n))
    Mo = np.empty((nout, n))
    w = w0.copy()
    M1 = M10.copy()
    M2 = M20.copy()
    W[0] = w
    Mo[0] = M1 + M2
    h1, h2 = 1.0 - 0.5 * a1, 1.0 - 0.5 * a2
    g1, g2 = c1 * h1, c2 * h2
    inv_u0 = 1.0 / u0
    gM = np.empty(n)
    rhs = np.empty(n)
    y = np.empty(n)
    w1 = np.empty(n)
    dw = np.empty(n)
    k = 1
    for it in range(1, nt):
        fi = f[it]
        for j in range(n):
            gM[j] = g1 * M1[j] + g2 * M2[j]
        # rhs = q + S w - L* (g M)
        for j in range(2, n - 2):
            rhs[j] = (Q[j] * fi + Sb[0, j + 2] * w[j + 2] + Sb[1, j + 1] * w[j + 1]
                      + Sb[2, j] * w[j] + Sb[3, j - 1] * w[j - 1] + Sb[4, j - 2] * w[j - 2]
                      - Lsb[0, j + 1] * gM[j + 1] - Lsb[1, j] * gM[j] - Lsb[2, j - 1] * gM[j - 1])
        for j in (0, 1, n - 2, n - 1):
            rhs[j] = _rhs_at(j, n, Q[j] * fi, Sb, Lsb, w, gM)
        for j in range(n):
            rhs[j] *= act[j]
        # forward substitution (unit lower factor) ...
        y[0] = rhs[0]
        y[1] = rhs[1] - lo1[0] * y[0]
        for j in range(2, n):
            y[j] = rhs[j] - lo1[j - 1] * y[j - 1] - lo2[j - 2] * y[j - 2]
        # ... and back substitution (upper factor)
        w1[n - 1] = y[n - 1] * inv_u0[n - 1]
        w1[n - 2] = (y[n - 2] - u1[n - 2] * w1[n - 1]) * inv_u0[n - 2]
        for j in range(n - 3, -1, -1):
            w1[j] = (y[j] - u1[j] * w1[j + 1] - u2[j] * w1[j + 2]) * inv_u0[j]
        # Maxwell updates  Mi <- ci [(1 - ai/2) Mi + Di L (w1 - w)]
        for j in range(n):
            dw[j] = w1[j] - w[j]
        dk = Lb[1, 0] * dw[0] + Lb[0, 1] * dw[1]
        M1[0] = c1 * (h1 * M1[0] + D1[0] * dk)
        M2[0] = c2 * (h2 * M2[0] + D2[0] * dk)
        for j in range(1, n - 1):
            dk = Lb[2, j - 1] * dw[j - 1] + Lb[1, j] * dw[j] + Lb[0, j + 1] * dw[j + 1]
            M1[j] = c1 * (h1 * M1[j] + D1[j] * dk)
            M2[j] = c2 * (h2 * M2[j] + D2[j] * dk)
        dk = Lb[2, n - 2] * dw[n - 2] + Lb[1, n - 1] * dw[n - 1]
        M1[n - 1] = c1 * (h1 * M1[n - 1] + D1[n - 1] * dk)
        M2[n - 1] = c2 * (h2 * M2[n - 1] + D2[n - 1] * dk)
        for j in range(n):
            w[j] = w1[j]
        if it % store_every == 0:
            W[k] = w
            for j in range(n):
                Mo[k, j] = M1[j] + M2[j]
            k += 1
    return W, Mo


if HAVE_NUMBA:
    _rhs_at = numba.njit(cache=True, inline="always")(_rhs_at)
    _loop = numba.njit(cache=True)(_loop)


def _penta_lu_core(d, sup1, sup2, sub1, sub2):
    """LU without pivoting of a pentadiagonal matrix (diagonal d, super-diagonals
    sup1/sup2, sub-diagonals sub1/sub2).  Returns the two sub-diagonals of the
    unit-lower factor and the three diagonals of the upper factor."""
    n = d.size
    u0 = d.copy()
    u1 = sup1.copy()
    u2 = sup2.copy()
    s1 = sub1.copy()            # current entries A[i+1, i]
    s2 = sub2.copy()            # current entries A[i+2, i]
    lo1 = np.zeros(n - 1)
    lo2 = np.zeros(n - 2)
    for i in range(n - 1):
        m = s1[i] / u0[i]                   # eliminate A[i+1, i]
        lo1[i] = m
        u0[i + 1] -= m * u1[i]
        if i + 2 < n:
            u1[i + 1] -= m * u2[i]
            m2 = s2[i] / u0[i]              # eliminate A[i+2, i]
            lo2[i] = m2
            s1[i + 1] -= m2 * u1[i]         # A[i+2, i+1]
            u0[i + 2] -= m2 * u2[i]
    return lo1, lo2, u0, u1, u2


if HAVE_NUMBA:
    _penta_lu_core = numba.njit(cache=True)(_penta_lu_core)


def _penta_lu(ab):
    """LU of a pentadiagonal matrix stored as 5 LAPACK-style bands."""
    return _penta_lu_core(ab[2].copy(), ab[1, 1:].copy(), ab[0, 2:].copy(),
                          ab[3, :-1].copy(), ab[4, :-2].copy())


def run_numba(beam, t, Q, f, store_every=1):
    """Compiled time loop.  Returns (w, M) at every ``store_every``-th step."""
    if not HAVE_NUMBA:
        raise ImportError("numba is not installed")
    t = np.asarray(t, float)
    dt = t[1] - t[0]
    if not np.allclose(np.diff(t), dt):
        raise ValueError("run_numba needs a constant time step")
    S, A, c, a = _setup(beam, dt)
    if not np.allclose(beam.taup, beam.taup[:, :1]):
        raise ValueError("run_numba assumes uniform relaxation times")
    lo1, lo2, u0, u1, u2 = _penta_lu(_bands(A, 2, 2))
    w0, M0 = beam.elastic(Q * f[0])
    Mp0 = beam.split_moment(M0)
    return _loop(np.asarray(Q, float), np.asarray(f, float), w0, Mp0[0].copy(), Mp0[1].copy(),
                 _bands(S, 2, 2), _bands(beam.L, 1, 1), _bands(beam.Lstar, 1, 1),
                 beam.active.astype(float), lo1, lo2, u0, u1, u2,
                 np.ascontiguousarray(beam.Dp[0]), np.ascontiguousarray(beam.Dp[1]),
                 float(c[0, 0]), float(a[0, 0]), float(c[1, 0]), float(a[1, 0]), store_every)
