"""Experiment 4: one basin through cycles of 3 high-melt years and 8 cool years.

(a) A narrow and a wide basin with physically derived (alpha_e, alpha, tau):
    how much free space each has at the start of every season, and how much
    it spills downstream.
(b) The "ratchet": an idealised basin that relaxes fully during the cool
    spell (tau = 0.3 yr), run with alpha = rho_w/rho_sw (thin plate), 1 and
    rho_w/rho_i (half-space).  Fully relaxed, the free space is
        C = D0 - (rho_w/rho_i - alpha) M,
    where M is the water-equivalent depth retained in the basin so far.  So
    the thin-plate basin loses about 0.12 M per cycle, and the half-space
    basin resets completely.

    python -m basins.experiments.exp4_multiyear
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from ..basin import Basin, run_years
from ..cascade import forcing_cycles
from ..physics import ALPHA_HALF_SPACE, ALPHA_THIN_PLATE, DAY, RHO_I, RHO_W, YEAR
from .common import ps, save, savefig


def run(H=700.0, eta=1e15, D0=2.0, t_m=45 * DAY, widths=(500.0, 3000.0), fill_high=3.0, cool=0.1,
        n_high=3, n_cool=8, n_cycles=2, refreeze=1.0, ratchet_tau_yr=0.3) -> xr.Dataset:
    melt = forcing_cycles(n_high, n_cool, n_cycles, high=1.0, cool=cool)
    phys = []
    for W in widths:
        b = Basin.from_width(W, H, eta, D0)
        r = run_years(b, melt * fill_high * b.A, t_m, refreeze=refreeze)
        phys.append(r.assign_attrs({}))
    phys = xr.concat(phys, dim="width", combine_attrs="drop").assign_coords(width=("width", list(widths), {"units": "m"}))
    phys["tau_yr"] = ("width", [b_.tau / YEAR for b_ in [Basin.from_width(W, H, eta, D0) for W in widths]])
    phys["alpha_b"] = ("width", [Basin.from_width(W, H, eta, D0).alpha for W in widths])
    phys["alpha_e_b"] = ("width", [Basin.from_width(W, H, eta, D0).alpha_e for W in widths])

    melt0 = forcing_cycles(n_high, n_cool, n_cycles, high=1.0, cool=0.0)
    alphas = (ALPHA_THIN_PLATE, 1.0, ALPHA_HALF_SPACE)
    rat = []
    for al in alphas:
        b = Basin(A=1e6, D0=D0, tau=ratchet_tau_yr * YEAR, alpha=al, alpha_e=0.0)
        rat.append(run_years(b, melt0 * fill_high * b.A, t_m, refreeze=refreeze))
    rat = xr.concat(rat, dim="alpha", combine_attrs="drop").assign_coords(alpha=list(alphas))
    retained = ((rat.inflow - rat.spill) / 1e6).cumsum("year")       # water-equivalent depth kept [m]
    rat["retained"] = retained.assign_attrs(units="m", long_name="water-equivalent depth retained so far")
    starts = np.arange(n_cycles) * (n_high + n_cool)
    pred = D0 - (RHO_W / RHO_I - rat.alpha) * retained.shift(year=1, fill_value=0.0)
    rat["C_relaxed_analytic"] = pred.assign_attrs(units="m", long_name="fully relaxed free space, analytic")

    ds = xr.merge([phys.rename({v: f"phys_{v}" for v in phys.data_vars}).rename({"time": "time"}),
                   rat.rename({v: f"ratchet_{v}" for v in rat.data_vars})], compat="override", combine_attrs="drop")
    ds["melt"] = ("year", melt)
    ds["cycle_start"] = ("cycle", starts)
    ds.attrs.update(H=H, eta=eta, D0=D0, t_m_days=t_m / DAY, fill_high=fill_high, cool=cool,
                    n_high=n_high, n_cool=n_cool, refreeze=refreeze, ratchet_tau_yr=ratchet_tau_yr)
    return ds


def plot_basins(ds):
    import matplotlib.pyplot as plt
    nW = ds.sizes["width"]
    fig, axs = plt.subplots(3, nW, figsize=(15, 9), sharex=True, constrained_layout=True)
    t = ds.time
    for j, W in enumerate(ds.width.values):
        d = ds.sel(width=W)
        a = axs[0, j]
        a.plot(t, d.phys_capacity, color=ps.VISCO, lw=1.6)
        a.axhline(ds.D0, color=ps.MUTED, ls=":", lw=1)
        a.set(ylabel="free space C (m)",
              title=rf"W = {W / 1e3:g} km:  $\tau$ = {float(d.phys_tau_yr):.2g} yr, $\alpha$ = {float(d.phys_alpha_b):.3f},"
                    rf" $\alpha_e$ = {float(d.phys_alpha_e_b):.2f}")
        a = axs[1, j]
        a.plot(t, d.phys_p, color=ps.ACCENT, label="ice plug p")
        a.plot(t, d.phys_w, color=ps.ELASTIC, label="floor deflection w")
        a.plot(t, d.phys_h, color=ps.WATER, label="water h")
        a.set(ylabel="m")
        a.legend(fontsize=8.5, loc="upper left")
        a = axs[2, j]
        yr = ds.year.values
        a.bar(yr + 0.5, d.phys_spill / float(d.phys_inflow.max()), width=0.8, color=ps.ELASTIC, label="spilled")
        a.step(np.r_[yr, yr[-1] + 1], np.r_[ds.melt.values, ds.melt.values[-1]], where="post", color=ps.INK,
               lw=1, label="melt (relative)")
        a.set(ylabel="per year / high-year inflow", xlabel="years")
        a.legend(fontsize=8.5, loc="upper right")
    for a in axs.flat:
        for y0 in range(0, ds.sizes["year"], ds.n_high + ds.n_cool):
            a.axvspan(y0, y0 + ds.n_high, color=ps.GRID, zorder=0)
    return fig


def plot_ratchet(ds):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.4), constrained_layout=True)
    cols = [ps.ELASTIC, ps.INK, ps.ISO]
    for c, al in zip(cols, ds.alpha.values):
        d = ds.sel(alpha=al)
        ax[0].plot(ds.time, d.ratchet_capacity, color=c, label=rf"$\alpha$ = {al:.3f}")
        ax[1].plot(ds.year, d.ratchet_C_start, "o", color=c, ms=5, label=rf"model, $\alpha$ = {al:.3f}")
        ax[1].plot(ds.year, d.ratchet_C_relaxed_analytic, "-", color=c, lw=1, alpha=0.7)
    for a in ax:
        for y0 in range(0, ds.sizes["year"], ds.n_high + ds.n_cool):
            a.axvspan(y0, y0 + ds.n_high, color=ps.GRID, zorder=0)
        a.axhline(ds.D0, color=ps.MUTED, ls=":", lw=1)
    ax[0].set(xlabel="years", ylabel="free space C (m)",
              title=rf"idealised basin, $\tau$ = {ds.ratchet_tau_yr:g} yr, no melt in cool years")
    ax[0].legend(fontsize=9)
    ax[1].set(xlabel="year", ylabel="free space at season start (m)",
              title=r"dots: model; lines: fully relaxed $D_0 - (\rho_w/\rho_i - \alpha)M$")
    return fig


def main():
    ds = run()
    for al in ds.alpha.values:
        d = ds.sel(alpha=al)
        s = ds.cycle_start.values[1:]
        print(f"alpha {al:.3f}: free space at the start of each later cycle {d.ratchet_C_start.values[s].round(3)} m, "
              f"analytic {d.ratchet_C_relaxed_analytic.values[s].round(3)} m")
    print(save(ds, "exp4_multiyear"), savefig(plot_basins(ds), "exp4_basins"), savefig(plot_ratchet(ds), "exp4_ratchet"))


if __name__ == "__main__":
    main()
