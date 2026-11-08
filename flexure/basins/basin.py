"""
basins/basin.py
===============

A single lake basin whose floor sinks under its load.

State (all depths in metres, measured from the original lake floor):

* ``h``  liquid water depth
* ``p``  thickness of the refrozen ice plug
* ``wv`` viscous part of the floor deflection (positive down)

Load, in water-equivalent depth:    L = h + (rho_i/rho_w) p
Floor deflection:                   w = alpha_e L + wv,   dwv/dt = ((alpha - alpha_e) L - wv) / tau
Free space below the sill:          C = D0 + w - p - h

During a melt season the inflow fills the basin until C = 0. The lake then
sits at its sill, and whatever the deepening basin cannot take spills
downstream. The basin is a vertical-sided box of area A, and the sill does
not move: alpha is the floor's deflection relative to the sill.

Two independent integrators:

* :func:`fill_season`: one season with constant inflow, using
  ``scipy.integrate.solve_ivp`` with an event at h = D (stage 1). It is
  checked against the analytic results :func:`gap_analytic`,
  :func:`overtop_time_analytic` and :func:`critical_Pi`.
* :func:`advance` / :func:`run_years`: an exact piecewise solution, phase
  by phase (filling, spilling at the sill, or over-full), for many years
  with refreezing (stage 2). It is accurate for any tau and step.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import xarray as xr
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from .physics import DAY, RHO_I, RHO_W, YEAR, basin_response

R_IW = RHO_I / RHO_W          # water-equivalent depth of 1 m of ice (0.917)


@dataclass
class Basin:
    """One lake basin.  ``A`` area [m^2], ``D0`` initial sill height above the
    floor [m], ``tau`` viscous relaxation time [s], ``alpha`` final
    deflection per unit water-equivalent load, ``alpha_e`` its immediate
    elastic part (0 <= alpha_e <= alpha, alpha_e < 1)."""
    A: float
    D0: float
    tau: float
    alpha: float = 1.0
    alpha_e: float = 0.0
    W: float = np.nan

    def __post_init__(self):
        if not (0.0 <= self.alpha_e < 1.0 and self.alpha_e <= self.alpha):
            raise ValueError("need 0 <= alpha_e < 1 and alpha_e <= alpha")
        if self.tau <= 0 or self.A <= 0:
            raise ValueError("tau and A must be positive")

    @classmethod
    def from_width(cls, W, H, eta, D0, area=None, E=1.0e9, nu=0.3, k_factor=np.pi, method="layer"):
        """A basin of width W on ice of thickness H and viscosity eta, with
        (alpha_e, alpha, tau) from :func:`physics.basin_response`.  The area
        defaults to W^2 (PLACEHOLDER shape)."""
        a_e, a, tau = basin_response(W, H, eta, E=E, nu=nu, k_factor=k_factor, method=method)
        return cls(A=float(W) ** 2 if area is None else float(area), D0=float(D0), tau=float(tau),
                   alpha=float(a), alpha_e=float(a_e), W=float(W))

    @property
    def a(self):
        """Viscous share of the final deflection, alpha - alpha_e."""
        return self.alpha - self.alpha_e


@dataclass
class State:
    h: float = 0.0
    p: float = 0.0
    wv: float = 0.0


def load(s: State) -> float:
    return s.h + R_IW * s.p


def deflection(b: Basin, s: State) -> float:
    return b.alpha_e * load(s) + s.wv


def capacity(b: Basin, s: State) -> float:
    return b.D0 + deflection(b, s) - s.p - s.h


def freeze(s: State, fraction: float = 1.0) -> State:
    """End of season: a fraction of the water refreezes into the plug, which
    expands by rho_w/rho_i.  The load (and so the deflection) is unchanged.
    fraction < 1 leaves a buried liquid lake (PLACEHOLDER switch)."""
    hf = fraction * s.h
    return replace(s, h=s.h - hf, p=s.p + hf / R_IW)


# ===========================================================================
# Stage 1: analytic results (alpha_e = 0, constant inflow, empty start)
# ===========================================================================
def fill_number(Q, A, D0, tau):
    """Pi = tau Q / (A D0): how far the lake can fill in one relaxation time,
    relative to the basin depth."""
    return tau * Q / (A * D0)


def gap_analytic(t, Pi, tau, D0, alpha=1.0):
    """g = D - h for constant inflow, alpha_e = 0, until the lake overtops:
    g/D0 = 1 - alpha Pi + (alpha - 1) Pi t/tau + alpha Pi exp(-t/tau).
    For alpha = 1: g = (D0 - Q tau/A) + (Q tau/A) exp(-t/tau)."""
    s = np.asarray(t, float) / tau
    return D0 * (1.0 - alpha * Pi + (alpha - 1.0) * Pi * s + alpha * Pi * np.exp(-s))


def _F(s, alpha):
    return alpha * (1.0 - np.exp(-s)) - (alpha - 1.0) * s


def critical_Pi(alpha=1.0):
    """Smallest Pi for which the lake ever overtops (given unlimited time).
    1 for alpha = 1; 1 / [1 - (alpha - 1) ln(alpha/(alpha - 1))] for alpha > 1
    (about 1.29 for alpha = 1.09); 0 for alpha < 1 (it always overtops in the
    end, because water arrives faster than the floor can sink)."""
    if alpha < 1.0:
        return 0.0
    if alpha == 1.0:
        return 1.0
    return 1.0 / (1.0 - (alpha - 1.0) * np.log(alpha / (alpha - 1.0)))


def season_boundary(s_m, alpha=1.0):
    """The Pi above which the lake overtops within a season of length
    t_m = s_m tau (alpha_e = 0, constant inflow):  Pi_b = 1 / max F(s), s <= s_m,
    with F(s) = alpha (1 - e^-s) - (alpha - 1) s.  For tau >> t_m this is
    Q t_m / (A D0) > 1."""
    s_m = np.asarray(s_m, float)
    if alpha > 1.0:
        s_m = np.minimum(s_m, np.log(alpha / (alpha - 1.0)))
    return 1.0 / _F(s_m, alpha)


def overtop_time_analytic(Pi, tau, alpha=1.0):
    """Time at which the lake first reaches its sill (alpha_e = 0, constant
    inflow, empty start), or inf.  For alpha = 1: tau ln(Pi / (Pi - 1))."""
    if alpha == 1.0:
        return tau * np.log(Pi / (Pi - 1.0)) if Pi > 1.0 else np.inf
    G = lambda s: 1.0 - Pi * _F(s, alpha)
    if alpha > 1.0:
        s_min = np.log(alpha / (alpha - 1.0))
        return tau * brentq(G, 0.0, s_min) if G(s_min) <= 0 else np.inf
    hi = 1.0
    while G(hi) > 0:
        hi *= 2.0
    return tau * brentq(G, 0.0, hi)


# ===========================================================================
# Stage 1: one season with solve_ivp and an event at the sill
# ===========================================================================
def _h_sill(b: Basin, p: float, wv: float) -> float:
    """Water depth with the lake at its sill (C = 0), given p and wv."""
    return (b.D0 - p * (1.0 - b.alpha_e * R_IW) + wv) / (1.0 - b.alpha_e)


def fill_season(b: Basin, Q: float, t_m: float, state0: State | None = None,
                rtol: float = 1e-10, atol: float = 1e-12, n_out: int = 201) -> xr.Dataset:
    """One melt season of constant inflow Q [m^3/s] lasting t_m [s], integrated
    with ``solve_ivp``: filling (dh/dt = Q/A) until the event C = 0, then at
    the sill (h tied to the floor, the rest spills) until the season ends or
    the floor sinks faster than the lake fills.  Returns a Dataset over time
    with h, the basin depth D = D0 + w, wv, the free space C, the spill rate
    and the spilled volume; attrs hold ``overtopped`` and ``t_star``."""
    s0 = State() if state0 is None else state0
    p, u = s0.p, Q / b.A
    t_out = np.linspace(0.0, t_m, n_out)
    rec_t, rec_h, rec_wv, rec_S = [], [], [], []
    t, h, wv, spilled, t_star = 0.0, s0.h, s0.wv, 0.0, np.inf

    def rhs_fill(_, y):
        L = y[0] + R_IW * p
        return [u, (b.a * L - y[1]) / b.tau]

    def ev_full(_, y):
        return b.D0 + b.alpha_e * (y[0] + R_IW * p) + y[1] - p - y[0]
    ev_full.terminal, ev_full.direction = True, -1

    def dwv_sill(wv_):
        return (b.a * (_h_sill(b, p, wv_) + R_IW * p) - wv_) / b.tau

    def rhs_sill(_, y):
        return [dwv_sill(y[0])]

    def ev_detach(_, y):                                 # spill rate falls to zero
        return Q - b.A * dwv_sill(y[0]) / (1.0 - b.alpha_e)
    ev_detach.terminal, ev_detach.direction = True, -1

    at_sill = ev_full(0, [h, wv]) <= 1e-12 * b.D0 and Q - b.A * dwv_sill(wv) / (1 - b.alpha_e) >= 0
    for _ in range(50):
        if t >= t_m * (1 - 1e-12):
            break
        te = np.unique(np.r_[t_out[(t_out > t) & (t_out < t_m)], t_m])
        if not at_sill:
            sol = solve_ivp(rhs_fill, (t, t_m), [h, wv], events=ev_full, rtol=rtol, atol=atol,
                            t_eval=te, method="LSODA")
            y = np.asarray(sol.y, float).reshape(2, -1)        # (empty if the event came first)
            rec_t += list(sol.t); rec_h += list(y[0]); rec_wv += list(y[1]); rec_S += [0.0] * len(sol.t)
            if sol.status == 1:                          # reached the sill
                t, (h, wv) = sol.t_events[0][0], sol.y_events[0][0]
                t_star = min(t_star, t)
                at_sill = True
            else:
                break
        else:
            sol = solve_ivp(rhs_sill, (t, t_m), [wv], events=ev_detach, rtol=rtol, atol=atol,
                            t_eval=te, method="LSODA")
            y = np.asarray(sol.y, float).reshape(1, -1)
            hs = np.array([_h_sill(b, p, v) for v in y[0]])
            Ss = np.array([Q - b.A * dwv_sill(v) / (1 - b.alpha_e) for v in y[0]])
            rec_t += list(sol.t); rec_h += list(hs); rec_wv += list(y[0]); rec_S += list(Ss)
            t1 = sol.t_events[0][0] if sol.status == 1 else t_m
            wv1 = sol.y_events[0][0][0] if sol.status == 1 else y[0][-1]
            spilled += Q * (t1 - t) - b.A * (_h_sill(b, p, wv1) - _h_sill(b, p, wv))
            t, wv, h = t1, wv1, _h_sill(b, p, wv1)
            at_sill = False
            if sol.status != 1:
                break
    rec_t, rec_h, rec_wv = np.array(rec_t), np.array(rec_h), np.array(rec_wv)
    L = rec_h + R_IW * p
    w = b.alpha_e * L + rec_wv
    ds = xr.Dataset(
        {"h": ("time", rec_h, {"units": "m", "long_name": "water depth"}),
         "D": ("time", b.D0 + w - p, {"units": "m", "long_name": "basin depth above the plug (sill - floor - plug)"}),
         "wv": ("time", rec_wv, {"units": "m"}),
         "capacity": ("time", b.D0 + w - p - rec_h, {"units": "m"}),
         "spill_rate": ("time", np.array(rec_S), {"units": "m3 s-1"})},
        coords={"time": ("time", rec_t, {"units": "s"})},
        attrs={"overtopped": int(np.isfinite(t_star)), "t_star": float(t_star),
               "spilled_volume": float(spilled), "Pi": fill_number(Q, b.A, b.D0, b.tau),
               "alpha": b.alpha, "alpha_e": b.alpha_e, "tau": b.tau, "t_m": t_m})
    return ds


# ===========================================================================
# Stage 2: exact piecewise stepper
# ===========================================================================
def _fill_solution(b, s, u, t):
    """Exact state after time t of filling at rate u = q/A (no spill)."""
    L0 = load(s)
    wv = b.a * (L0 + u * (t - b.tau)) + (s.wv - b.a * (L0 - u * b.tau)) * np.exp(-t / b.tau)
    return replace(s, h=s.h + u * t, wv=wv)


def _sill_coeffs(b, p):
    c1 = (b.alpha - 1.0) / (1.0 - b.alpha_e)
    c0 = b.a * ((b.D0 - p * (1.0 - b.alpha_e * R_IW)) / (1.0 - b.alpha_e) + R_IW * p)
    return c0, c1


def _sill_wv(b, p, wv0, t):
    """Exact wv after time t with the lake held at its sill."""
    c0, c1 = _sill_coeffs(b, p)
    if abs(c1) < 1e-14:
        return wv0 + c0 * t / b.tau
    winf = -c0 / c1
    return winf + (wv0 - winf) * np.exp(c1 * t / b.tau)


def _sill_spill_rate(b, p, wv, q):
    c0, c1 = _sill_coeffs(b, p)
    return q - b.A * (c0 + c1 * wv) / b.tau / (1.0 - b.alpha_e)


def _relax(b, s, t):
    """Exact wv after time t with a constant load (no inflow, or over-full)."""
    target = b.a * load(s)
    return replace(s, wv=target + (s.wv - target) * np.exp(-t / b.tau))


def _first_drop_below(f, t0, t1, n=8):
    """First t in (t0, t1] where f changes from >= 0 to < 0 (f smooth with at
    most a couple of turning points), or None."""
    ts = np.linspace(t0, t1, n + 1)
    fs = [f(x) for x in ts]
    for i in range(n):
        if fs[i] >= 0 > fs[i + 1]:
            return brentq(f, ts[i], ts[i + 1], xtol=1e-9 * max(1.0, t1), rtol=1e-12)
    return None


def advance(b: Basin, s: State, q: float, dt: float):
    """Advance one basin by dt with a constant inflow rate q [m^3/s].
    Returns (new state, spilled volume [m^3]).  Solved exactly, phase by
    phase, with root finding at the phase changes."""
    tol = 1e-10 * max(1.0, b.D0)
    t, spilled = 0.0, 0.0
    u = q / b.A
    for _ in range(12):
        rem = dt - t
        if rem <= 1e-12 * dt:
            break
        C = capacity(b, s)
        if C > tol:                                           # filling
            f = lambda x: capacity(b, _fill_solution(b, s, u, x))
            tc = _first_drop_below(f, 0.0, rem) if f(rem) < 0 else None
            if tc is None:
                s = _fill_solution(b, s, u, rem)
                t = dt
            else:
                s = _fill_solution(b, s, u, tc)
                t += tc
            continue
        hs = _h_sill(b, s.p, s.wv)
        if hs <= 0.0:                                         # over-full: the plug reaches the sill
            spilled += b.A * s.h                              # any water left on top runs off
            s = replace(s, h=0.0)
            Cempty = lambda x: capacity(b, _relax(b, s, x))
            tc = None
            if Cempty(rem) > 0:
                tc = brentq(Cempty, 0.0, rem) if Cempty(0.0) < 0 else 0.0
            span = rem if tc is None else max(tc, 1e-9 * dt)
            s = _relax(b, s, span)
            spilled += q * span
            t += span
            continue
        spilled += b.A * (s.h - hs)                           # (round-off) put the lake exactly at its sill
        s = replace(s, h=hs)
        if _sill_spill_rate(b, s.p, s.wv, q) < 0:             # floor sinks faster than inflow: detach
            f = lambda x: capacity(b, _fill_solution(b, s, u, x))
            tc = _first_drop_below(f, 1e-9 * rem, rem) if f(rem) < 0 else None
            span = rem if tc is None else tc
            s = _fill_solution(b, s, u, span)
            t += span
            continue
        # at the sill, spilling
        S = lambda x: _sill_spill_rate(b, s.p, _sill_wv(b, s.p, s.wv, x), q)
        hsf = lambda x: _h_sill(b, s.p, _sill_wv(b, s.p, s.wv, x))
        ends = [rem]
        if S(rem) < 0:
            ends.append(brentq(S, 0.0, rem))
        if hsf(rem) < 0:
            ends.append(brentq(hsf, 0.0, rem))
        span = max(min(ends), 1e-9 * dt)
        wv1 = _sill_wv(b, s.p, s.wv, span)
        h1 = max(_h_sill(b, s.p, wv1), 0.0)
        spilled += q * span - b.A * (h1 - hs)
        s = replace(s, h=h1, wv=wv1)
        t += span
    return s, spilled


def run_years(b: Basin, inflow, t_m: float, dt: float = 0.25 * DAY, refreeze: float = 1.0,
              state0: State | None = None, winter_points: int = 12) -> xr.Dataset:
    """Run a basin for len(inflow) years.  ``inflow[y]`` is the volume [m^3]
    delivered in year y, at a constant rate over the first t_m [s] of the
    year.  At the end of each season a fraction ``refreeze`` of the water
    freezes into the plug, then the floor keeps relaxing through the winter
    (and through cool years).

    Returns an ``xarray.Dataset`` with time series (dim ``time``, in years)
    of h, p, w, wv and the free space C, cumulative inflow and spill, and
    per-year values (dim ``year``): inflow, spill, whether the lake
    overtopped, and the free space at the start and end of the season."""
    s = State() if state0 is None else replace(state0)
    n_season = max(1, int(round(t_m / dt)))
    dts = t_m / n_season
    T, rows = 0.0, []
    cum_in = cum_sp = 0.0
    yr = {k: [] for k in ("inflow", "spill", "C_start", "C_end", "h_end", "p_end", "w_end")}

    def rec():
        rows.append((T / YEAR, s.h, s.p, deflection(b, s), s.wv, capacity(b, s), cum_in, cum_sp))

    rec()
    for V in np.asarray(inflow, float):
        q = V / t_m
        yr["C_start"].append(capacity(b, s))
        sp_year = 0.0
        for _ in range(n_season):
            s, sp = advance(b, s, q, dts)
            T += dts
            cum_in += q * dts
            cum_sp += sp
            sp_year += sp
            rec()
        yr["inflow"].append(V); yr["spill"].append(sp_year)
        yr["C_end"].append(capacity(b, s)); yr["h_end"].append(s.h)
        s = freeze(s, refreeze)
        yr["p_end"].append(s.p); yr["w_end"].append(deflection(b, s))
        winter = YEAR - t_m
        for _ in range(winter_points):
            s, sp = advance(b, s, 0.0, winter / winter_points)
            T += winter / winter_points
            cum_sp += sp
            rec()
    r = np.array(rows)
    ny = len(yr["inflow"])
    ds = xr.Dataset(
        {"h": ("time", r[:, 1], {"units": "m", "long_name": "liquid water depth"}),
         "p": ("time", r[:, 2], {"units": "m", "long_name": "ice-plug thickness"}),
         "w": ("time", r[:, 3], {"units": "m", "long_name": "floor deflection (+ down)"}),
         "wv": ("time", r[:, 4], {"units": "m", "long_name": "viscous part of the deflection"}),
         "capacity": ("time", r[:, 5], {"units": "m", "long_name": "free space below the sill"}),
         "cum_inflow": ("time", r[:, 6], {"units": "m3"}),
         "cum_spill": ("time", r[:, 7], {"units": "m3"}),
         "inflow": ("year", np.array(yr["inflow"]), {"units": "m3"}),
         "spill": ("year", np.array(yr["spill"]), {"units": "m3", "long_name": "volume spilled downstream"}),
         "overtopped": ("year", np.array(yr["spill"]) > 1e-9 * max(1.0, b.A)),
         "C_start": ("year", np.array(yr["C_start"]), {"units": "m", "long_name": "free space at the start of the season"}),
         "C_end": ("year", np.array(yr["C_end"]), {"units": "m"}),
         "h_end": ("year", np.array(yr["h_end"]), {"units": "m"}),
         "p_end": ("year", np.array(yr["p_end"]), {"units": "m"}),
         "w_end": ("year", np.array(yr["w_end"]), {"units": "m"})},
        coords={"time": ("time", r[:, 0], {"units": "yr"}), "year": np.arange(ny)},
        attrs={"A": b.A, "D0": b.D0, "tau_s": b.tau, "tau_yr": b.tau / YEAR, "alpha": b.alpha,
               "alpha_e": b.alpha_e, "W": b.W, "t_m_s": t_m, "refreeze": refreeze})
    return ds
