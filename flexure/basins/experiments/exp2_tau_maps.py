"""Experiment 2: relaxation time tau(W, H) of a lake basin.

Both limit formulae (thin viscous plate, viscous half-space) and the exact
viscous-layer solution, for W = 0.2-5 km and H = 300-1000 m, with k = pi/W
and 2 pi/W as bounds.  tau scales linearly with eta, so the maps are computed
for eta = 1e15 Pa s and the 1-year contour is also drawn for 1e14 Pa s.

    python -m basins.experiments.exp2_tau_maps
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from ..physics import (YEAR, flexural_length, layer_modes, tau_half_space, tau_thickness_mode,
                       tau_thin_plate, wavenumber)
from .common import km_axis, ps, save, savefig


def run(W=np.geomspace(200.0, 5000.0, 61), H=np.linspace(300.0, 1000.0, 36),
        k_factors=(np.pi, 2 * np.pi), eta=1e15, E=(1e9, 9e9)) -> xr.Dataset:
    kf = np.asarray(k_factors, float)
    k = wavenumber(W[None, None, :], kf[:, None, None]) * np.ones((1, H.size, 1))
    HH = H[None, :, None] * np.ones_like(k)
    tau, alpha = layer_modes(k, HH, eta)
    dims = ("k_factor", "H", "W")
    ds = xr.Dataset(
        {"kH": (dims, k * HH),
         "tau_thin_plate": (dims, tau_thin_plate(k, HH, eta) / YEAR, {"units": "yr"}),
         "tau_half_space": (dims, tau_half_space(k, eta) / YEAR, {"units": "yr"}),
         "tau_layer_fast": (dims, tau[..., 0] / YEAR, {"units": "yr"}),
         "tau_layer_slow": (dims, tau[..., 1] / YEAR, {"units": "yr"}),
         "alpha_fast": (dims, alpha[..., 0]),
         "alpha_slow": (dims, alpha[..., 1]),
         "ell": (("E", "H"), np.array([flexural_length(H, e) for e in E]), {"units": "m"})},
        coords={"k_factor": ("k_factor", kf, {"long_name": "k W"}), "H": ("H", H, {"units": "m"}),
                "W": ("W", W, {"units": "m"}), "E": ("E", list(E), {"units": "Pa"})},
        attrs={"eta": eta, "note": "tau is proportional to eta"})
    # tau of the mode that carries most of the surface deflection
    dom = np.abs(ds.alpha_fast) >= np.abs(ds.alpha_slow)
    ds["tau_layer"] = xr.where(dom, ds.tau_layer_fast, ds.tau_layer_slow).assign_attrs(units="yr")
    # a 1-D sweep in kH for the curves
    kH = np.geomspace(0.05, 30.0, 200)
    H0 = 700.0
    t1, a1 = layer_modes(kH / H0, H0, eta)
    ds["curve_tau_fast"] = ("kH_curve", t1[:, 0] / YEAR, {"units": "yr"})
    ds["curve_tau_slow"] = ("kH_curve", t1[:, 1] / YEAR, {"units": "yr"})
    ds["curve_alpha_fast"] = ("kH_curve", a1[:, 0])
    ds["curve_alpha_slow"] = ("kH_curve", a1[:, 1])
    ds["curve_tau_tp"] = ("kH_curve", tau_thin_plate(kH / H0, H0, eta) / YEAR)
    ds["curve_tau_hs"] = ("kH_curve", tau_half_space(kH / H0, eta) / YEAR)
    ds["curve_tau_th"] = ((), float(tau_thickness_mode(H0, eta) / YEAR))
    ds = ds.assign_coords(kH_curve=kH)
    ds.attrs["curve_H"] = H0
    return ds


def plot_maps(ds, k_factor=np.pi):
    import matplotlib.pyplot as plt
    d = ds.sel(k_factor=k_factor, method="nearest")
    Wk = d.W / 1e3
    fig, axs = plt.subplots(1, 3, figsize=(16, 5), sharey=True, constrained_layout=True)
    levels = np.arange(-3, 4.01, 0.25)
    panels = [("tau_thin_plate", "thin viscous plate", d.kH > 1, r"$kH>1$"),
              ("tau_half_space", "viscous half-space", d.kH < 2, r"$kH<2$"),
              ("tau_layer", "exact viscous layer (dominant mode)", None, None)]
    for ax, (var, title, invalid, lab) in zip(axs, panels):
        im = ax.contourf(Wk, d.H, np.log10(d[var]), levels=levels, cmap="viridis", extend="both")
        if invalid is not None:
            ax.contourf(Wk, d.H, invalid.astype(float), levels=[0.5, 1.5], colors=["white"], alpha=0.5)
            ax.text(0.97, 0.03, f"faded: outside validity ({lab})", transform=ax.transAxes, ha="right",
                    fontsize=9, color=ps.INK)
        for eta_f, ls in [(1.0, "-"), (0.1, "--")]:          # tau = 1 yr at eta = 1e15 and 1e14
            c = ax.contour(Wk, d.H, d[var] * eta_f, levels=[1.0], colors="w", linestyles=ls, linewidths=1.6)
        ax.contour(Wk, d.H, d.kH, levels=[1, 2], colors=ps.MUTED, linewidths=0.9, linestyles=":")
        for e, col in zip(ds.E.values, [ps.ELASTIC, ps.TENSION]):
            ax.plot(ds.ell.sel(E=e) / 1e3, ds.H, color=col, lw=1.6, label=rf"$W=\ell(H)$, E = {e / 1e9:g} GPa")
        ax.set(xlabel="lake width W (km)", title=title)
        km_axis(ax)
        ax.grid(False)
    axs[0].set_ylabel("ice thickness H (m)")
    axs[0].legend(loc="upper left", fontsize=8.5)
    cb = fig.colorbar(im, ax=axs, shrink=0.9)
    cb.set_label(r"$\log_{10}\,\tau$ (years) at $\eta=10^{15}$ Pa s")
    fig.suptitle(rf"$k = {k_factor / np.pi:g}\pi/W$.  White: $\tau$ = 1 yr for $\eta = 10^{{15}}$ (solid) "
                 r"and $10^{14}$ Pa s (dashed).  Dotted: $kH$ = 1, 2.", fontsize=10.5)
    return fig


def plot_curves(ds):
    import matplotlib.pyplot as plt
    kH = ds.kH_curve
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6), constrained_layout=True)
    ax[0].loglog(kH, ds.curve_tau_tp, color=ps.ELASTIC, ls="--", label=r"thin plate $\eta H^3k^4/3\rho_{sw}g$")
    ax[0].loglog(kH, ds.curve_tau_hs, color=ps.ISO, ls="--", label=r"half-space $2\eta k/\rho_i g$")
    ax[0].axhline(float(ds.curve_tau_th), color=ps.ACCENT, ls=":", label="thickness mode, long-wave limit")
    ax[0].loglog(kH, ds.curve_tau_fast, color=ps.VISCO, lw=2.6, label="exact layer: fast mode")
    ax[0].loglog(kH, ds.curve_tau_slow, color=ps.ACCENT, lw=2.6, label="exact layer: slow mode")
    ax[0].axhline(1.0, color=ps.MUTED, lw=0.8)
    ax[0].set(xlabel="kH", ylabel="relaxation time (years)", ylim=(1e-4, 1e4),
              title=rf"H = {ds.curve_H:.0f} m, $\eta$ = {ds.eta:.0e} Pa s")
    ax[0].legend(fontsize=8.5)
    ax[1].semilogx(kH, ds.curve_alpha_fast, color=ps.VISCO, lw=2.6, label="fast mode")
    ax[1].semilogx(kH, ds.curve_alpha_slow, color=ps.ACCENT, lw=2.6, label="slow mode")
    ax[1].semilogx(kH, ds.curve_alpha_fast + ds.curve_alpha_slow, color=ps.INK, ls=":", label=r"total = $\rho_w/\rho_i$")
    ax[1].axhline(1000 / 1028, color=ps.ELASTIC, ls="--", lw=1, label=r"$\rho_w/\rho_{sw}$")
    ax[1].set(xlabel="kH", ylabel=r"share of the final deflection, $\alpha_j$", ylim=(-0.05, 1.2),
              title="how much each mode moves the lake floor")
    ax[1].legend(fontsize=8.5)
    return fig


def main():
    ds = run()
    print(save(ds, "exp2_tau_maps"))
    for kf in ds.k_factor.values:
        print(savefig(plot_maps(ds, kf), f"exp2_tau_maps_k{kf / np.pi:g}pi"))
    print(savefig(plot_curves(ds), "exp2_tau_curves"))


if __name__ == "__main__":
    main()
