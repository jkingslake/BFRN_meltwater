"""
basins/physics.py
=================

Relaxation times, compensation factors and viscosity for a lake basin of
width W on an ice shelf of thickness H.  SI units throughout.

A lake of width W is represented by one Fourier mode of wavenumber k.  The
choice of k is a placeholder: ``k_factor`` = pi (k = pi/W, the lake is half a
wavelength) and 2 pi (k = 2 pi/W, the lake is a whole wavelength) bracket
it, and both should be reported.

Three descriptions of the viscous response are provided.

* ``tau_thin_plate``: thin viscous plate floating on sea water (kH << 1),
  tau = eta H^3 k^4 / (3 rho_sw g).
* ``tau_half_space``: viscous half-space (kH >~ 2),
  tau = 2 eta k / (rho_i g).
* ``layer_modes``: the exact Fourier-mode solution for a linear viscous
  layer of thickness H with a traction-free top, gravity at both surfaces and
  hydrostatic sea water at the base.  It has two relaxation modes (bending,
  and a thickness change) and recovers both limits above.

The elastic part is that of a thin elastic plate on a Winkler foundation.
"""
from __future__ import annotations

import numpy as np

RHO_W = 1000.0     # melt water [kg m^-3]
RHO_SW = 1028.0    # sea water
RHO_I = 917.0      # ice
G = 9.81           # gravity [m s^-2]
DAY = 86400.0
YEAR = 365.25 * DAY

#: equilibrium deflection per unit water-equivalent load
ALPHA_THIN_PLATE = RHO_W / RHO_SW   # the whole column floats down (~0.97)
ALPHA_HALF_SPACE = RHO_W / RHO_I    # the ice under the load flows away (~1.09)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def wavenumber(W, k_factor=np.pi):
    """Wavenumber [rad/m] standing for a lake of width W [m].  PLACEHOLDER:
    k_factor = pi or 2 pi bracket the choice."""
    return k_factor / np.asarray(W, dtype=float)


def flexural_rigidity(H, E=1.0e9, nu=0.3):
    """D_f = E H^3 / (12 (1 - nu^2))  [N m]."""
    return E * np.asarray(H, dtype=float) ** 3 / (12.0 * (1.0 - nu**2))


def flexural_length(H, E=1.0e9, nu=0.3):
    """ell = (D_f / (rho_sw g))^(1/4)  [m]."""
    return (flexural_rigidity(H, E, nu) / (RHO_SW * G)) ** 0.25


def maxwell_time(eta, E=1.0e9, nu=0.3):
    """Maxwell time of a viscoelastic plate, 4 eta (1 - nu^2) / E  [s]."""
    return 4.0 * np.asarray(eta, dtype=float) * (1.0 - nu**2) / E


def glen_viscosity(strain_rate, A=1.7e-25, n=3.0):
    """Effective viscosity of Glen's law linearised about a background
    effective strain rate [s^-1]:  eta = B / (2 e^((n-1)/n)),  B = A^(-1/n).
    A = 1.7e-25 Pa^-3 s^-1 is about -20 C."""
    B = A ** (-1.0 / n)
    return B / (2.0 * np.asarray(strain_rate, dtype=float) ** ((n - 1.0) / n))


# ---------------------------------------------------------------------------
# Relaxation-time limits
# ---------------------------------------------------------------------------
def tau_thin_plate(k, H, eta):
    """Thin viscous plate on sea water: eta H^3 k^4 / (3 rho_sw g).  Valid for kH << 1."""
    return np.asarray(eta, float) * np.asarray(H, float) ** 3 * np.asarray(k, float) ** 4 / (3.0 * RHO_SW * G)


def tau_half_space(k, eta):
    """Viscous half-space: 2 eta k / (rho_i g).  Valid for kH >~ 2; independent of H."""
    return 2.0 * np.asarray(eta, float) * np.asarray(k, float) / (RHO_I * G)


def tau_thickness_mode(H, eta):
    """Long-wave limit of the layer's slow mode: thickness differences spread
    by extensional flow of a floating layer, 4 eta / (rho_i (1 - rho_i/rho_sw) g H).
    It does not depend on k."""
    return 4.0 * np.asarray(eta, float) / (RHO_I * (1.0 - RHO_I / RHO_SW) * G * np.asarray(H, float))


def tau_viscoelastic_plate(k, H, eta, E=1.0e9, nu=0.3):
    """Relaxation time of a mode of a Maxwell viscoelastic plate on sea water,
    tau_M (1 + (k ell)^4) = tau_M + tau_thin_plate.  This is the slow time of
    the flexure model in ``viscoelastic_beam.py`` when nu_v = nu."""
    kl4 = (np.asarray(k, float) * flexural_length(H, E, nu)) ** 4
    return maxwell_time(eta, E, nu) * (1.0 + kl4)


def alpha_elastic(k, H, E=1.0e9, nu=0.3):
    """Immediate elastic deflection per unit water-equivalent load,
    (rho_w/rho_sw) / (1 + (k ell)^4): about 0.97 for wide lakes (W >> ell),
    about 0 for narrow ones.  Exact for one Fourier mode of a thin elastic
    plate; with k = 1/W it is the interpolation suggested in the brief,
    0.97 (W/ell)^4 / (1 + (W/ell)^4).  PLACEHOLDER where thin-plate theory
    fails (kH >~ 1)."""
    kl4 = (np.asarray(k, float) * flexural_length(H, E, nu)) ** 4
    return ALPHA_THIN_PLATE / (1.0 + kl4)


# ---------------------------------------------------------------------------
# Exact Fourier-mode solution for a viscous layer floating on sea water
# ---------------------------------------------------------------------------
def _layer_matrix(kH):
    """Rates of the two interface displacements for one Fourier mode, in
    units where k = 1, eta = 1 and rho_i g = 1.

    Stream function f(z) e^{ikx}, with z measured in units of 1/k from the
    base (0) to the top (kH), and a basis that stays bounded for any kH:
        f = b1 e^{s} + b2 s e^{s} + b3 e^{-z} + b4 z e^{-z},   s = z - kH.
    Boundary conditions, linearised about the flat layer:
        top  (z = kH): shear stress 0;  normal stress = -(q + rho_i g zeta_t)
        base (z = 0):  shear stress 0;  normal stress = (rho_sw - rho_i) g zeta_b
    and the interfaces move with the vertical velocity.  Returns P with
        d/dt (zeta_t, zeta_b) = P @ (q + rho_i g zeta_t, -(rho_sw - rho_i) g zeta_b).
    """
    h = float(kH)

    def basis(z):
        s = z - h
        es, ez = np.exp(s), np.exp(-z)
        # value and derivatives 1..3 of each basis function (in units of k)
        phi = np.array([[es, es, es, es],
                        [s * es, (1 + s) * es, (2 + s) * es, (3 + s) * es],
                        [ez, -ez, ez, -ez],
                        [z * ez, (1 - z) * ez, (z - 2) * ez, (3 - z) * ez]])
        return phi                                   # [function, derivative order]

    top, base = basis(h), basis(0.0)
    # with stream-function coefficients i*b_j: sigma_xz = i eta k^2 sum b (phi'' + phi),
    # sigma_zz = eta k^2 sum b N,  N = -(phi''' - 3 phi'),  w = k sum b phi
    shear = lambda ph: ph[:, 2] + ph[:, 0]
    normal = lambda ph: -(ph[:, 3] - 3 * ph[:, 1])
    M = np.array([shear(top), shear(base), normal(top), normal(base)])
    Phi = np.array([top[:, 0], base[:, 0]])
    # sum b N(top) = -(q + rho_i g zeta_t)/(eta k^2),  sum b N(base) = (rho_sw - rho_i) g zeta_b/(eta k^2)
    return -Phi @ np.linalg.solve(M, np.eye(4))[:, 2:]


def layer_modes(k, H, eta):
    """Exact relaxation of a viscous layer (thickness H, viscosity eta)
    floating on sea water, for one Fourier mode k, after a surface load is
    switched on.

    The surface deflection (positive down), per unit water-equivalent load,
    is   sum_j alpha_j (1 - exp(-t / tau_j))   with two modes.  Returns
    (tau, alpha), each an array of shape k.shape + (2,), ordered
    (fast, slow).  The alphas add up to rho_w/rho_i, the final, fully fluid
    state in which the ice under the load has flowed away.

    * kH << 1: the fast mode is bending, tau ~ tau_thin_plate and
      alpha ~ rho_w/rho_sw.  The slow mode is the thickness change,
      tau ~ tau_thickness_mode and alpha ~ rho_w/rho_i - rho_w/rho_sw.
    * kH >> 1: one mode, tau ~ tau_half_space and alpha ~ rho_w/rho_i,
      carries all the deflection.
    """
    k, H, eta = np.broadcast_arrays(np.asarray(k, float), np.asarray(H, float), np.asarray(eta, float))
    tau = np.empty(k.shape + (2,))
    alpha = np.empty(k.shape + (2,))
    drho = (RHO_SW - RHO_I) / RHO_I
    for idx in np.ndindex(k.shape):
        P = _layer_matrix(k[idx] * H[idx])
        Msys = P @ np.diag([1.0, -drho])             # d zeta/dt = Msys zeta + P[:, 0] q   (scaled)
        lam, V = np.linalg.eig(Msys)
        lam, V = lam.real, V.real
        order = np.argsort(-np.abs(lam))             # fastest first
        lam, V = lam[order], V[:, order]
        c = np.linalg.solve(V, np.linalg.solve(Msys, P[:, 0]))   # zeta_ss = -sum c_j v_j
        # deflection down of the top, per unit load q (scaled by rho_i g):
        amp = c * V[0]                               # zeta_t(t) = -sum amp_j (1 - e^{lam_j t})
        scale = eta[idx] * k[idx] / (RHO_I * G)      # time unit
        tau[idx] = -scale / lam
        alpha[idx] = amp * RHO_W / RHO_I             # per unit water depth
    return tau, alpha


def layer_dominant(k, H, eta):
    """(tau, alpha) of the layer mode that carries most of the surface deflection."""
    tau, alpha = layer_modes(k, H, eta)
    j = np.argmax(np.abs(alpha), axis=-1)[..., None]
    return np.take_along_axis(tau, j, -1)[..., 0], np.take_along_axis(alpha, j, -1)[..., 0]


# ---------------------------------------------------------------------------
# Parameters for the single-mode basin model
# ---------------------------------------------------------------------------
def basin_response(W, H, eta, E=1.0e9, nu=0.3, k_factor=np.pi, method="layer"):
    """(alpha_e, alpha, tau) for a lake basin of width W, for the basin model
    w = alpha_e L + w_v,  dw_v/dt = ((alpha - alpha_e) L - w_v) / tau.

    method
      'thin_plate'  Maxwell viscoelastic plate mode: alpha = rho_w/rho_sw,
                    tau = tau_M + tau_thin_plate.  Exact for one mode of the
                    plate model in ``viscoelastic_beam.py`` (nu_v = nu).
      'half_space'  alpha = rho_w/rho_i, tau = tau_M + tau_half_space.
      'layer'       the dominant mode of the exact viscous layer, plus the
                    Maxwell time tau_M as in the plate (an approximation
                    away from the thin-plate limit).  At small kH this is
                    the bending mode; the slow thickness mode (decades to
                    centuries) is left out.
    alpha_e is ``alpha_elastic`` in every case (PLACEHOLDER for kH >~ 1).
    """
    k = wavenumber(W, k_factor)
    a_e = alpha_elastic(k, H, E, nu)
    tM = maxwell_time(eta, E, nu)
    if method == "thin_plate":
        return a_e, ALPHA_THIN_PLATE * np.ones_like(a_e), tM + tau_thin_plate(k, H, eta)
    if method == "half_space":
        return a_e, ALPHA_HALF_SPACE * np.ones_like(a_e), tM + tau_half_space(k, eta)
    if method == "layer":
        tau, alpha = layer_dominant(k, H, eta)
        return a_e, alpha, tM + tau
    raise ValueError(f"unknown method {method!r}")
