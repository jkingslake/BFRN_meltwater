"""
Worked examples for the viscoelastic ice-shelf flexure model.

    python run_examples.py        # writes figures/*.png (takes a few seconds)

1. step_response     a lake appears suddenly and is left in place
2. fill_drain        a lake fills, sits, then drains rapidly (stress reversal)
3. relaxation        how fast each wavelength relaxes:  tau_k = tau (1 + (k ell)^4)
4. grounding_line    a lake near a clamped grounding line
5. verification      comparison with exact solutions + convergence
"""
import os
import time

import matplotlib.pyplot as plt
import numpy as np

import plotstyle as ps
from viscoelastic_beam import (
    SECONDS_PER_DAY as DAY, IceShelf, ViscoelasticBeam, fill_hold_drain,
    hetenyi_strip, modal_step_response, parabolic_lake, relaxation_time, relaxation_times,
    spectral_step_response)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
ps.apply()

SHELF = IceShelf()             # H = 250 m, E = 1 GPa, eta = 1e14 Pa s, nu = 0.3, nu_v = 0.5
LAKE_HALF_WIDTH = 1500.0       # m
LAKE_DEPTH = 4.0               # m


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path)
    plt.close(fig)
    print("  wrote", os.path.relpath(path, HERE))


def shade_lake(ax, x_km, depth, scale=1.0, base=0.0):
    """Light blue band showing where the lake sits."""
    inside = depth > 0
    if inside.any():
        ax.axvspan(x_km[inside][0], x_km[inside][-1], color=ps.OCEAN_LIGHT, alpha=0.35, lw=0,
                   zorder=0)


# ---------------------------------------------------------------------------
def example_step_response():
    x = np.linspace(-15e3, 15e3, 3001)
    depth = parabolic_lake(x, 0.0, LAKE_HALF_WIDTH, LAKE_DEPTH)
    q = SHELF.lake_load(depth)
    beam = ViscoelasticBeam(x, SHELF)
    tau = SHELF.tau
    t = np.linspace(0, 30 * tau, 1201)
    tic = time.perf_counter()
    res = beam.run(t, lambda _: q)
    print(f"  step response: {x.size} nodes x {t.size} steps in "
          f"{time.perf_counter() - tic:.2f} s")

    show = [0.0, 0.5, 1, 3, 10, 30]
    cols = [ps.ELASTIC] + ps.SEQ[2::2][: len(show) - 1]
    xk = x / 1e3
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.2, 6.6), sharex=True)
    for s, c in zip(show, cols):
        i = np.argmin(np.abs(t - s * tau))
        lab = r"$t = 0^+$ (elastic)" if s == 0 else rf"$t = {s:g}\,\tau$"
        a1.plot(xk, -res.w[i], color=c, label=lab)
        a2.plot(xk, res.sigma[i] / 1e3, color=c)
    a1.plot(xk, -beam.isostatic(q), color=ps.ISO, ls="--", label="isostatic limit")
    for a in (a1, a2):
        shade_lake(a, xk, depth)
    a1.set_ylabel("surface elevation change, $-w$ (m)")
    a1.set_title("A 4 m deep, 3 km wide lake appears at t = 0 and stays")
    a1.legend(loc="lower right", ncol=1, fontsize=9)
    a2.axhline(0, color=ps.MUTED, lw=0.8)
    a2.set_ylabel("upper-surface stress (kPa)\n+ = tension")
    a2.set_xlabel("distance, x (km)")
    a2.set_title("Bending stresses relax away as the ice creeps")
    a1.set_xlim(-6, 6)
    a1.text(0.01, 0.04, rf"$\ell$ = {SHELF.ell:.0f} m,  $\tau$ = {tau / DAY:.1f} days",
            transform=a1.transAxes, color=ps.INK2)
    save(fig, "step_response.png")
    return res


# ---------------------------------------------------------------------------
def example_fill_drain():
    x = np.linspace(-15e3, 15e3, 3001)
    depth = parabolic_lake(x, 0.0, LAKE_HALF_WIDTH, LAKE_DEPTH)
    q_full = SHELF.lake_load(depth)
    t_fill, t_hold, t_drain = 10 * DAY, 15 * DAY, 0.25 * DAY
    frac = lambda tt: fill_hold_drain(tt, t_fill, t_hold, t_drain)
    t = np.unique(np.concatenate([np.linspace(0, 50 * DAY, 2001),
                                  np.linspace(25 * DAY, 25.5 * DAY, 41)]))
    beam = ViscoelasticBeam(x, SHELF)
    res = beam.run(t, lambda tt: frac(tt) * q_full)
    f = frac(t)
    i0 = np.argmin(np.abs(x))
    w_el_full, M_el_full = beam.elastic(q_full)
    td = res.time_days.values

    fig = plt.figure(figsize=(9.5, 8.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.65, 1, 1], hspace=0.55, wspace=0.28)
    a0 = fig.add_subplot(gs[0, :])
    a0.fill_between(td, LAKE_DEPTH * f, color=ps.WATER, alpha=0.35, lw=0)
    a0.plot(td, LAKE_DEPTH * f, color=ps.WATER)
    a0.set_ylabel("lake depth\nat centre (m)")
    a0.set_title("Fill for 10 days, hold for 15 days, drain in 6 hours")
    for tt, lab in [(5, "filling"), (17.5, "full"), (37, "drained")]:
        a0.text(tt, LAKE_DEPTH * 0.45, lab, ha="center", color=ps.INK2)

    a1 = fig.add_subplot(gs[1, :], sharex=a0)
    a1.plot(td, -f * w_el_full[i0], color=ps.ELASTIC, label="purely elastic")
    a1.plot(td, -f * q_full[i0] / SHELF.k, color=ps.ISO, ls="--", label="isostatic")
    a1.plot(td, -res.w[:, i0], color=ps.VISCO, label="viscoelastic (Maxwell)")
    a1.set_ylabel("deflection at\nlake centre (m)")
    a1.legend(loc="center right")
    a1.set_title("The ice sags further as it creeps; after drainage a dip lingers for weeks")

    a2 = fig.add_subplot(gs[2, :], sharex=a0)
    s_el = SHELF.surface_stress(M_el_full) / 1e3
    a2.plot(td, f * s_el[i0], color=ps.ELASTIC, label="purely elastic")
    a2.plot(td, res.sigma[:, i0] / 1e3, color=ps.VISCO, label="viscoelastic, lake centre")
    a2.plot(td, res.sigma.max(axis=1) / 1e3, color=ps.TENSION, lw=1.4, ls=":",
            label="viscoelastic, max tension anywhere")
    a2.axhline(0, color=ps.MUTED, lw=0.8)
    a2.set_ylabel("upper-surface\nstress (kPa)")
    a2.set_xlabel("time (days)")
    a2.legend(loc="lower right", fontsize=9)
    a2.set_title("Drainage flips the stress: compression under the lake becomes tension")
    a2.set_xlim(0, 50)
    save(fig, "fill_drain_timeseries.png")

    # profiles at key moments
    keys = [(10, "end of filling (day 10)"), (25, "just before drainage (day 25)"),
            (25.25, "just after drainage"), (35, "10 days later")]
    cols = [ps.SEQ[3], ps.SEQ[7], ps.TENSION, ps.ACCENT]
    xk = x / 1e3
    fig, (b1, b2) = plt.subplots(2, 1, figsize=(8.2, 6.6), sharex=True)
    for (tt, lab), c in zip(keys, cols):
        i = np.argmin(np.abs(td - tt))
        b1.plot(xk, -res.w[i], color=c, label=lab)
        b2.plot(xk, res.sigma[i] / 1e3, color=c)
    for b in (b1, b2):
        shade_lake(b, xk, depth)
        b.axhline(0, color=ps.MUTED, lw=0.8)
    b1.set_ylabel("surface elevation change (m)")
    b1.set_title("Deflection through a fill-drain cycle")
    b1.legend(loc="lower right", fontsize=9)
    b2.set_ylabel("upper-surface stress (kPa)\n+ = tension")
    b2.set_xlabel("distance, x (km)")
    b2.set_title("After drainage the ice is left bent upwards, in tension at the centre")
    b1.set_xlim(-6, 6)
    save(fig, "fill_drain_profiles.png")
    return res


# ---------------------------------------------------------------------------
def example_relaxation():
    ell, tau = SHELF.ell, SHELF.tau
    single = IceShelf(nu_v=SHELF.nu)            # equal Poisson ratios: one Maxwell time
    lam = np.logspace(-0.3, 2.3, 400) * ell
    kx = 2 * np.pi / lam
    slow, fast = relaxation_times(kx, SHELF)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.3))
    a1.loglog(lam / ell, relaxation_time(kx, single) / tau, color=ps.MUTED, ls=":", lw=2,
              label=r"one Maxwell time: $\tau\,[1 + (2\pi\ell/\lambda)^4]$")
    a1.loglog(lam / ell, slow / tau, color=ps.VISCO, label="slow mode (most of the creep)")
    a1.loglog(lam / ell, fast / tau, color=ps.ELASTIC, label="fast mode")
    a1.set_xlabel(r"wavelength / flexural length,  $\lambda/\ell$")
    a1.set_ylabel(r"relaxation time / mean Maxwell time,  $\tau_\lambda/\tau$")
    a1.set_title("Relaxation times of a wavelength $\\lambda$")
    a1.legend(loc="upper right", fontsize=8.5)
    a1.annotate("narrow features:\nheld up elastically\nfor a long time", (0.62, 4),
                color=ps.INK2, fontsize=9)
    a1.annotate("broad features: relax\nin a few days", (12, 1.6), color=ps.INK2, fontsize=9)

    t = np.linspace(0, 12, 300)
    for r, c in zip([20, 6, 3, 1.5], [ps.SEQ[1], ps.SEQ[4], ps.SEQ[7], ps.SEQ[9]]):
        k = 2 * np.pi / (r * ell)
        a2.plot(t, modal_step_response(k, SHELF, t * tau), color=c, label=rf"$\lambda = {r:g}\,\ell$")
        a2.plot(t, modal_step_response(k, single, t * tau), color=c, ls=":", lw=1.2)
    a2.set_xlabel(r"time / mean Maxwell time,  $t/\tau$")
    a2.set_ylabel("amplitude / isostatic amplitude")
    a2.set_title("Each wavelength jumps elastically, then creeps\n(dotted: one Maxwell time)",
                 fontsize=11)
    a2.set_ylim(0, 1.05)
    a2.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "relaxation_spectrum.png")


# ---------------------------------------------------------------------------
def example_grounding_line():
    x = np.linspace(0, 30e3, 3001)
    beam = ViscoelasticBeam(x, SHELF, left="clamped", right="free")
    tau = SHELF.tau
    t = np.linspace(0, 20 * tau, 801)
    xk = x / 1e3
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.2, 6.4), sharex=True)
    for x0, c in zip([1.0e3, 2.5e3, 5e3], [ps.SEQ[9], ps.SEQ[6], ps.SEQ[2]]):
        depth = parabolic_lake(x, x0 + LAKE_HALF_WIDTH, LAKE_HALF_WIDTH, LAKE_DEPTH)
        res = beam.run(t, lambda _: SHELF.lake_load(depth))
        lab = f"lake edge {x0 / 1e3:g} km from grounding line"
        a1.plot(xk, -res.w[0], color=c, label=lab)
        a1.plot(xk, -res.w[-1], color=c, ls="--", lw=1.4)
        a2.plot(xk, res.sigma[0] / 1e3, color=c)
        a2.plot(xk, res.sigma[-1] / 1e3, color=c, ls="--", lw=1.4)
    a1.axvline(0, color=ps.INK, lw=2)
    a1.set_ylabel("surface elevation change (m)")
    a1.set_title(r"Lakes near a grounding line (clamped, $w = w' = 0$):  solid $t=0^+$,"
                 r" dashed $t = 20\tau$", fontsize=11)
    a1.legend(loc="lower right", fontsize=9)
    a2.axhline(0, color=ps.MUTED, lw=0.8)
    a2.set_ylabel("upper-surface stress (kPa)\n+ = tension")
    a2.set_xlabel("distance from grounding line (km)")
    a2.set_title("The grounding line feels the lake elastically, then forgets it")
    a1.set_xlim(0, 12)
    save(fig, "grounding_line.png")


# ---------------------------------------------------------------------------
def example_verification():
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.0))
    # (a) elastic strip load vs Hetenyi
    x = np.linspace(-20e3, 20e3, 4001)
    a, q0 = 1000.0, SHELF.lake_load(3.0)
    beam = ViscoelasticBeam(x, SHELF)
    w, _ = beam.elastic(np.where(np.abs(x) <= a, q0, 0.0))
    ex = hetenyi_strip(x, a, q0, SHELF)
    xk = x / 1e3
    ax[0].plot(xk, -ex, color=ps.MUTED, lw=5, alpha=0.5, label="Hetényi (exact)")
    ax[0].plot(xk, -w, color=ps.ELASTIC, lw=1.6, label="finite differences")
    ax[0].set_xlim(-5, 5)
    ax[0].set_title("Elastic limit: strip load")
    ax[0].set_xlabel("x (km)")
    ax[0].set_ylabel("surface elevation change (m)")
    ax[0].legend(loc="lower right")

    # (b) viscoelastic vs exact Fourier solution (periodic)
    x = np.linspace(-25e3, 25e3, 2501)[:-1]
    s = np.clip(x / LAKE_HALF_WIDTH, -1, 1)
    q = SHELF.lake_load(LAKE_DEPTH * np.cos(0.5 * np.pi * s) ** 2)
    tau = SHELF.tau
    t = np.linspace(0, 10 * tau, 401)
    res = ViscoelasticBeam(x, SHELF).run(t, lambda _: q)
    exact = spectral_step_response(x, q, t, SHELF)
    for tt, c in zip([0, 1, 3, 10], [ps.ELASTIC, ps.SEQ[3], ps.SEQ[6], ps.SEQ[9]]):
        i = np.argmin(np.abs(t - tt * tau))
        ax[1].plot(x / 1e3, -exact[i], color=ps.MUTED, lw=5, alpha=0.5)
        ax[1].plot(x / 1e3, -res.w[i], color=c, lw=1.6,
                   label=r"$t=0^+$" if tt == 0 else rf"$t={tt}\tau$")
    ax[1].set_xlim(-5, 5)
    ax[1].set_title("Viscoelastic: grey = exact Fourier modes")
    ax[1].set_xlabel("x (km)")
    ax[1].legend(loc="lower right")

    # (c) convergence in dt and dx, both made dimensionless
    T = 6 * tau
    load = lambda tt: SHELF.lake_load(LAKE_DEPTH * np.cos(0.5 * np.pi * s) ** 2) * \
        np.sin(np.pi * tt / T) ** 2
    bm = ViscoelasticBeam(x, SHELF)
    ref = bm.run(np.linspace(0, T, 8193), load).w[-1].values
    scale = np.max(np.abs(ref))
    nts = np.array([8, 16, 32, 64, 128])
    et = np.array([np.max(np.abs(bm.run(np.linspace(0, T, n + 1), load).w[-1] - ref))
                   for n in nts]) / scale
    dts = T / nts / tau
    exs, ns = [], np.array([250, 500, 1000, 2000])
    for n in ns:
        xx = np.linspace(-25e3, 25e3, n + 1)[:-1]
        ss = np.clip(xx / LAKE_HALF_WIDTH, -1, 1)
        qq = SHELF.lake_load(LAKE_DEPTH * np.cos(0.5 * np.pi * ss) ** 2)
        ww, _ = ViscoelasticBeam(xx, SHELF).elastic(qq)
        ex0 = spectral_step_response(xx, qq, [0.0], SHELF)[0]
        exs.append(np.max(np.abs(ww - ex0)) / np.max(np.abs(ex0)))
    dxs = 50e3 / ns / SHELF.ell
    ax[2].loglog(dts, et, "o-", color=ps.VISCO, ms=6, label=r"vs time step  $\Delta t/\tau$")
    ax[2].loglog(dxs, exs, "s-", color=ps.ELASTIC, ms=6, label=r"vs grid spacing  $\Delta x/\ell$")
    for xs, es in ((dts, et), (dxs, exs)):
        ax[2].loglog(xs, es[0] * (xs / xs[0]) ** 2, color=ps.MUTED, ls="--", lw=1, zorder=0)
    ax[2].text(0.04, 0.95, "dashed: slope 2\n(second-order accurate)", transform=ax[2].transAxes,
               color=ps.INK2, fontsize=9, va="top")
    ax[2].minorticks_off()
    ax[2].set_title("Convergence")
    ax[2].set_xlabel("dimensionless step size")
    ax[2].set_ylabel("max error / max deflection")
    ax[2].legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    save(fig, "verification.png")


if __name__ == "__main__":
    t1, t2 = SHELF.taus / DAY
    print(f"D = {SHELF.D:.3e} N m, ell = {SHELF.ell:.0f} m, tau = {SHELF.tau / DAY:.2f} days "
          f"(elements {t1:.2f} and {t2:.2f} days)")
    example_step_response()
    example_fill_drain()
    example_relaxation()
    example_grounding_line()
    example_verification()
