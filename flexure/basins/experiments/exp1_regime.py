"""Experiment 1: regime diagram.  Does a lake overtop its sill within one melt
season?  Shown in (Pi, t_m/tau) space for alpha = 0.98, 1 and 1.09, with the
analytic boundaries (constant inflow, alpha_e = 0, empty start).

    python -m basins.experiments.exp1_regime        (from the flexure folder)
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from ..basin import Basin, critical_Pi, fill_season, season_boundary
from ..physics import ALPHA_HALF_SPACE, ALPHA_THIN_PLATE
from .common import ps, save, savefig

ALPHAS = (round(ALPHA_THIN_PLATE, 3), 1.0, round(ALPHA_HALF_SPACE, 3))


def run(alphas=ALPHAS, Pi=np.geomspace(0.1, 20.0, 49), s_m=np.geomspace(0.01, 100.0, 41)) -> xr.Dataset:
    """Classify every (alpha, t_m/tau, Pi) with the solve_ivp model, in units
    where tau = A = D0 = 1 (so Q = Pi), and add the analytic boundary."""
    over = np.zeros((len(alphas), len(s_m), len(Pi)), bool)
    tstar = np.full(over.shape, np.nan)
    for a, alpha in enumerate(alphas):
        b = Basin(A=1.0, D0=1.0, tau=1.0, alpha=alpha, alpha_e=0.0)
        for i, sm in enumerate(s_m):
            for j, P in enumerate(Pi):
                ds = fill_season(b, P, sm, n_out=2)
                over[a, i, j] = bool(ds.attrs["overtopped"])
                tstar[a, i, j] = ds.attrs["t_star"]
    bound = np.array([season_boundary(s_m, al) for al in alphas])
    ds = xr.Dataset(
        {"overtopped": (("alpha", "s_m", "Pi"), over),
         "t_star": (("alpha", "s_m", "Pi"), tstar, {"long_name": "overtopping time / tau"}),
         "Pi_boundary": (("alpha", "s_m"), bound, {"long_name": "analytic boundary Pi_b(t_m/tau)"}),
         "Pi_critical": ("alpha", [critical_Pi(al) for al in alphas])},
        coords={"alpha": list(alphas), "s_m": ("s_m", s_m, {"long_name": "t_m / tau"}),
                "Pi": ("Pi", Pi, {"long_name": "tau Q / (A D0)"})},
        attrs={"experiment": "regime diagram", "alpha_e": 0.0})
    analytic = ds.Pi >= ds.Pi_boundary
    ds["agreement"] = (ds.overtopped == analytic).mean(("s_m", "Pi"))
    return ds


def plot(ds):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    fig, axs = plt.subplots(1, ds.sizes["alpha"], figsize=(15, 4.6), sharey=True, constrained_layout=True)
    cmap = ListedColormap([ps.SEQ[1], ps.ELASTIC])
    for ax, al in zip(axs, ds.alpha.values):
        d = ds.sel(alpha=al)
        ax.pcolormesh(d.s_m, d.Pi, d.overtopped.T.astype(float), cmap=cmap, vmin=0, vmax=1, shading="nearest")
        ax.plot(d.s_m, d.Pi_boundary, color=ps.INK, lw=2.2, label="analytic boundary")
        ax.plot(d.s_m, 1.0 / d.s_m, color=ps.INK, ls=":", lw=1.3, label=r"$Q t_m/(A D_0) = 1$ (rigid)")
        if float(d.Pi_critical) > 0:
            ax.axhline(float(d.Pi_critical), color=ps.ACCENT, ls="--", lw=1.3,
                       label=rf"$\Pi_c$ = {float(d.Pi_critical):.2f}")
        ax.set(xscale="log", yscale="log", xlabel=r"$t_m/\tau$", ylim=(d.Pi.min(), d.Pi.max()),
               title=rf"$\alpha$ = {al:g}   (agreement {100 * float(d.agreement):.1f}%)")
        ax.text(0.03, 0.95, "overtops", transform=ax.transAxes, color="w", weight="bold", va="top")
        ax.text(0.97, 0.05, "pinned", transform=ax.transAxes, color=ps.INK, weight="bold", ha="right")
        ax.grid(False)
    axs[0].set_ylabel(r"$\Pi = \tau Q/(A D_0)$")
    axs[-1].legend(loc="lower left", fontsize=8.5, framealpha=0.85, frameon=True)
    return fig


def main():
    ds = run()
    print(f"agreement with the analytic boundary: {ds.agreement.values}")
    print(save(ds, "exp1_regime"), savefig(plot(ds), "exp1_regime"))


if __name__ == "__main__":
    main()
