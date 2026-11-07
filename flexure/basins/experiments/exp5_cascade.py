"""Experiment 5: a chain of basins along a flow line.

Baseline: 15 basins with log-normal widths (median 1 km) and sill heights
between 1 and 4 m, on 700 m of ice with eta = 3e14 Pa s.  The head inflow in a
reference (high) year, V_ref, would just fill the first five empty basins.

* Extent through time under three forcings with the same mean melt: 3 high
  years then 8 cool years (repeated), constant melt, and random melt.
* Sensitivity of growth and memory to eta, H and the width distribution
  (one at a time, cycles forcing).  Growth: extent in the 3rd high year minus
  the 1st.  Memory: extent in the 1st high year of the 2nd cycle minus the
  1st cycle's (0 = full reset).
* Reset time: 3 high years, then n cool years, then one high year.  How
  many cool years until that high year reaches no further than a fresh
  system's first high year?

    python -m basins.experiments.exp5_cascade
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from ..cascade import (forcing_constant, forcing_cycles, forcing_stochastic, make_chain, random_widths,
                       run_cascade)
from ..physics import DAY
from .common import ps, save, savefig

BASE = dict(N=15, median=1000.0, spread=0.6, H=700.0, eta=3e14, D0_range=(1.0, 4.0), n_fill=5,
            t_m=45 * DAY, cool=0.1, seed=1)


def chain(N=15, median=1000.0, spread=0.6, H=700.0, eta=3e14, D0_range=(1.0, 4.0), n_fill=5, seed=1, **_):
    W = random_widths(N, median=median, spread=spread, seed=seed)
    rng = np.random.default_rng(seed + 1)
    D0 = np.exp(rng.uniform(*np.log(D0_range), N))
    basins, x = make_chain(W, D0, H, eta)
    V_ref = float(sum(b.A * b.D0 for b in basins[:n_fill]))
    return basins, x, V_ref


def run(base=BASE, n_years=33, sens=None, n_cool_max=15) -> xr.Dataset:
    p = dict(base)
    basins, x, V_ref = chain(**p)
    cyc = forcing_cycles(3, 8, n_years // 11, high=1.0, cool=p["cool"])
    mean = float(cyc.mean())
    forcings = {"cycles": cyc, "constant": forcing_constant(cyc.size, mean),
                "stochastic": forcing_stochastic(cyc.size, mean, 0.6, seed=3)}
    runs = [run_cascade(basins, f, V_ref, t_m=p["t_m"], x=x) for f in forcings.values()]
    ds = xr.concat(runs, dim="forcing", combine_attrs="drop_conflicts").assign_coords(forcing=list(forcings))
    ds.attrs.update({k: (str(v) if isinstance(v, tuple) else v) for k, v in p.items()})
    ds.attrs["V_ref_m3"] = V_ref

    # sensitivity, one parameter at a time
    sens = sens or {"eta": [1e14, 3e14, 1e15], "H": [300.0, 700.0, 1000.0], "median": [500.0, 1000.0, 2000.0]}
    labels, growth, memory, emax, tau_med = [], [], [], [], []
    for key, vals in sens.items():
        for v in vals:
            q = dict(p, **{key: v})
            b_, x_, V_ = chain(**q)
            r = run_cascade(b_, forcing_cycles(3, 8, 2, cool=q["cool"]), V_, t_m=q["t_m"])
            e = r.extent.values
            labels.append(f"{key}={v:g}")
            growth.append(e[2] - e[0]); memory.append(e[11] - e[0]); emax.append(e.max())
            tau_med.append(float(np.median(r.tau)))
    ds["sens_growth"] = ("case", growth, {"long_name": "extent, 3rd minus 1st high year"})
    ds["sens_memory"] = ("case", memory, {"long_name": "extent, 1st high year of cycle 2 minus cycle 1"})
    ds["sens_max_extent"] = ("case", emax)
    ds["sens_tau_median_yr"] = ("case", tau_med)
    ds = ds.assign_coords(case=labels)

    # reset time: with no melt in the cool years (pure relaxation) and with the baseline cool melt
    etas = sens["eta"]
    n_cool = np.arange(0, n_cool_max + 1)
    cools = (0.0, p["cool"])
    excess = np.zeros((len(cools), len(etas), n_cool.size))
    for c, cool in enumerate(cools):
        for a, eta in enumerate(etas):
            b_, x_, V_ = chain(**dict(p, eta=eta))
            for j, n in enumerate(n_cool):
                m = np.r_[[1.0] * 3, [cool] * n, [1.0]]
                e = run_cascade(b_, m, V_, t_m=p["t_m"]).extent.values
                excess[c, a, j] = e[-1] - e[0]
    ds["reset_excess"] = (("reset_cool", "reset_eta", "n_cool"), excess,
                          {"long_name": "extent of the high year after n cool years, minus a fresh system's"})
    ds = ds.assign_coords(reset_cool=list(cools), reset_eta=etas, n_cool=n_cool)
    return ds


def plot_extent(ds):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(ds.sizes["forcing"], 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    N = ds.sizes["basin"]
    for ax, f in zip(axs, ds.forcing.values):
        d = ds.sel(forcing=f)
        yr = ds.year.values
        ax2 = ax.twinx()
        ax2.bar(yr + 0.5, d.melt, width=0.85, color=ps.WATER, alpha=0.25)
        ax2.set_ylabel("melt (x reference)", color=ps.WATER)
        ax2.set_ylim(0, 3)
        ax2.grid(False)
        ax.step(np.r_[yr, yr[-1] + 1], np.r_[d.extent, d.extent[-1]], where="post", color=ps.VISCO, lw=2.2)
        ax.axhline(N, color=ps.TENSION, ls=":", lw=1)
        ax.set(ylabel="extent (basin index)", ylim=(-0.5, N + 0.8), title=f"forcing: {f}")
    axs[-1].set_xlabel("years")
    return fig


def plot_capacity(ds, forcing="cycles"):
    import matplotlib.pyplot as plt
    d = ds.sel(forcing=forcing)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw=dict(width_ratios=[3, 1]), constrained_layout=True)
    rel = (d.capacity / d.D0).clip(-0.2, 1.0)
    im = ax[0].pcolormesh(d.time, d.basin, rel.T, cmap="viridis", vmin=-0.2, vmax=1.0, shading="nearest")
    ax[0].step(np.r_[d.year, d.year[-1] + 1], np.r_[d.extent, d.extent[-1]], where="post", color="w", lw=1.5,
               label="extent")
    fig.colorbar(im, ax=ax[0], label=r"free space / $D_0$")
    ax[0].set(xlabel="years", ylabel="basin (downstream)", title=f"free space in each basin ({forcing} forcing)")
    ax[0].legend(loc="upper right")
    ax[1].barh(d.basin, d.tau, color=ps.ACCENT)
    ax[1].set(xscale="log", xlabel=r"$\tau$ (yr)", title="relaxation time")
    ax[1].axvline(1.0, color=ps.MUTED, ls=":")
    for i, (w, t) in enumerate(zip(d.W.values, d.tau.values)):
        ax[1].text(t * 1.1, i, f"{w / 1e3:.1f} km", va="center", fontsize=7.5)
    ax[1].set_ylim(ax[0].get_ylim())
    return fig


def plot_sensitivity(ds):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.6), constrained_layout=True)
    yy = np.arange(ds.sizes["case"])
    ax[0].barh(yy - 0.2, ds.sens_growth, height=0.4, color=ps.ELASTIC, label="growth over 3 high years")
    ax[0].barh(yy + 0.2, ds.sens_memory, height=0.4, color=ps.VISCO, label="memory after 8 cool years")
    labs = [f"{c}  (median tau {t:.2g} yr)" for c, t in zip(ds.case.values, ds.sens_tau_median_yr.values)]
    ax[0].set_yticks(yy, labs, fontsize=8.5)
    ax[0].invert_yaxis()
    ax[0].set(xlabel="change in extent (basins)", title="sensitivity (cycles forcing, one change at a time)")
    ax[0].legend(fontsize=8.5)
    for eta, c in zip(ds.reset_eta.values, [ps.VISCO, ps.ACCENT, ps.ELASTIC]):
        for cool, ls, mk in zip(ds.reset_cool.values, ["--", "-"], ["s", "o"]):
            ax[1].plot(ds.n_cool, ds.reset_excess.sel(reset_eta=eta, reset_cool=cool), ls=ls, marker=mk, ms=4,
                       color=c, label=rf"$\eta$ = {eta:.0e}, cool-year melt {cool:g}")
    ax[1].axhline(0, color=ps.MUTED, lw=1)
    ax[1].set(xlabel="cool years between high spells", ylabel="extra extent of the next high year",
              title="does the system reset?")
    ax[1].legend(fontsize=8.5)
    return fig


def main():
    ds = run()
    print("extent (cycles):", ds.extent.sel(forcing="cycles").values)
    print(ds[["sens_growth", "sens_memory", "sens_max_extent", "sens_tau_median_yr"]].to_dataframe())
    print(save(ds, "exp5_cascade"), savefig(plot_extent(ds), "exp5_extent"),
          savefig(plot_capacity(ds), "exp5_capacity"), savefig(plot_sensitivity(ds), "exp5_sensitivity"))


if __name__ == "__main__":
    main()
