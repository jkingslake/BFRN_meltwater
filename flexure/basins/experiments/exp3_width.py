"""Experiment 3: which lakes overtop in one melt season, as a function of width.

(i)  Fixed catchment melt: the same volume V reaches every lake, so the fill
     depth V/A falls as 1/W^2 while tau(W) falls too.
(ii) Inflow proportional to catchment area: Q = ratio * m * A / t_m, so the
     fill depth ratio * m is the same for every width and only tau(W)
     decides.

Each lake is a basin of area W^2 (PLACEHOLDER shape) with (alpha_e, alpha,
tau) from the exact viscous layer plus the Maxwell time
(``physics.basin_response``, method 'layer'), for k = pi/W and 2 pi/W.

    python -m basins.experiments.exp3_width
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from ..basin import Basin, fill_season
from ..physics import DAY, YEAR
from .common import km_axis, ps, save, savefig


def _case(W, H, eta, D0, Q, t_m, k_factor, area_factor):
    b = Basin.from_width(W, H, eta, D0, area=area_factor * W**2, k_factor=k_factor)
    r = fill_season(b, Q, t_m, n_out=2)
    return b, r.attrs["overtopped"], r.attrs["t_star"]


def run(H=700.0, D0=2.0, t_m=45 * DAY, etas=(1e14, 1e15), k_factors=(np.pi, 2 * np.pi),
        V=(6e6, 25e6), W=np.geomspace(200.0, 5000.0, 60), melt=0.1, ratio=np.geomspace(1.0, 100.0, 41),
        W2=np.geomspace(200.0, 5000.0, 41), area_factor=1.0) -> xr.Dataset:
    etas, kfs, Vs = np.asarray(etas, float), np.asarray(k_factors, float), np.atleast_1d(np.asarray(V, float))
    # (i) fixed catchment volume
    sh = (Vs.size, etas.size, kfs.size, W.size)
    o1, ts1 = np.zeros(sh), np.zeros(sh)
    tau1, al1, ae1 = (np.zeros(sh[1:]) for _ in range(3))
    for v, Vv in enumerate(Vs):
        for a, eta in enumerate(etas):
            for b_, kf in enumerate(kfs):
                for i, w in enumerate(W):
                    b, o, ts = _case(w, H, eta, D0, Vv / t_m, t_m, kf, area_factor)
                    o1[v, a, b_, i], ts1[v, a, b_, i] = o, ts / t_m
                    tau1[a, b_, i], al1[a, b_, i], ae1[a, b_, i] = b.tau / YEAR, b.alpha, b.alpha_e
    # (ii) inflow proportional to catchment area
    sh2 = (etas.size, kfs.size, ratio.size, W2.size)
    o2, ts2 = np.zeros(sh2), np.zeros(sh2)
    for a, eta in enumerate(etas):
        for b_, kf in enumerate(kfs):
            for j, r in enumerate(ratio):
                for i, w in enumerate(W2):
                    A = area_factor * w**2
                    _, o, ts = _case(w, H, eta, D0, r * melt * A / t_m, t_m, kf, area_factor)
                    o2[a, b_, j, i], ts2[a, b_, j, i] = o, ts / t_m
    ds = xr.Dataset(
        {"overtopped_fixedV": (("V", "eta", "k_factor", "W"), o1.astype(bool)),
         "tstar_fixedV": (("V", "eta", "k_factor", "W"), ts1, {"long_name": "t* / t_m"}),
         "tau": (("eta", "k_factor", "W"), tau1, {"units": "yr"}),
         "alpha": (("eta", "k_factor", "W"), al1),
         "alpha_e": (("eta", "k_factor", "W"), ae1),
         "fill_depth": (("V", "W"), Vs[:, None] / (area_factor * W**2), {"units": "m", "long_name": "V / A"}),
         "overtopped_ratio": (("eta", "k_factor", "ratio", "W2"), o2.astype(bool)),
         "tstar_ratio": (("eta", "k_factor", "ratio", "W2"), ts2, {"long_name": "t* / t_m"})},
        coords={"V": ("V", Vs, {"units": "m3"}), "eta": ("eta", etas, {"units": "Pa s"}), "k_factor": kfs,
                "W": ("W", W, {"units": "m"}),
                "ratio": ("ratio", ratio, {"long_name": "catchment-to-lake area ratio"}),
                "W2": ("W2", W2, {"units": "m"})},
        attrs={"H": H, "D0": D0, "t_m_days": t_m / DAY, "melt_m": melt, "area_factor": area_factor})
    # widest lake that still overtops, and the rigid-basin reference
    ws = np.full(sh[:3], np.nan)
    for idx in np.ndindex(sh[:3]):
        if o1[idx].any():
            ws[idx] = W[np.flatnonzero(o1[idx])].max()
    ds["W_switch"] = (("V", "eta", "k_factor"), ws, {"units": "m", "long_name": "widest lake that overtops"})
    ds["W_switch_rigid"] = ("V", np.sqrt(Vs / (area_factor * D0)), {"units": "m"})
    return ds


def plot_fixed(ds):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    Wk = ds.W / 1e3
    fig = plt.figure(figsize=(15, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(ds.sizes["V"], 2, width_ratios=[1, 1.15])
    ax0 = fig.add_subplot(gs[:, 0])
    cols = {1e14: ps.VISCO, 1e15: ps.ACCENT}
    for eta in ds.eta.values:
        for kf, ls in zip(ds.k_factor.values, ["-", "--"]):
            ax0.loglog(Wk, ds.tau.sel(eta=eta, k_factor=kf), color=cols.get(eta, ps.INK), ls=ls,
                       label=rf"$\eta$ = {eta:.0e}, k = {kf / np.pi:g}$\pi$/W")
    ax0.axhline(ds.t_m_days / 365.25, color=ps.MUTED, ls=":", lw=2, label="melt-season length")
    ax0.set(ylabel=r"relaxation time $\tau$ (yr)", title=f"H = {ds.H:.0f} m")
    km_axis(ax0)
    ax0.set_xlabel("lake width W (km)")
    ax0.legend(fontsize=8.5, loc="lower left")
    cmap = ListedColormap([ps.SEQ[1], ps.ELASTIC])
    for v, Vv in enumerate(ds.V.values):
        ax = fig.add_subplot(gs[v, 1])
        rows = [(ds.fill_depth.sel(V=Vv) > ds.D0).values]
        labels = ["rigid basin"]
        for eta in ds.eta.values:
            for kf in ds.k_factor.values:
                rows.append(ds.overtopped_fixedV.sel(V=Vv, eta=eta, k_factor=kf).values)
                labels.append(rf"$\eta$={eta:.0e}, k={kf / np.pi:g}$\pi$/W")
        ax.pcolormesh(Wk, np.arange(len(rows)), np.array(rows, float), cmap=cmap, vmin=0, vmax=1, shading="nearest")
        ax.set_yticks(np.arange(len(rows)), labels, fontsize=8.5)
        ax.invert_yaxis()
        km_axis(ax)
        ax.grid(False)
        ax.set_title(f"V = {Vv / 1e6:g} Mm$^3$ per season: overtops (orange) or pinned (blue)", fontsize=10.5)
    ax.set_xlabel("lake width W (km)")
    return fig


def plot_ratio(ds, k_factor=np.pi):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    fig, axs = plt.subplots(1, ds.sizes["eta"], figsize=(13, 4.6), sharey=True, constrained_layout=True)
    for ax, eta in zip(axs, ds.eta.values):
        d = ds.sel(eta=eta)
        ax.pcolormesh(ds.W2 / 1e3, ds.ratio, d.overtopped_ratio.sel(k_factor=k_factor).astype(float),
                      cmap=ListedColormap([ps.SEQ[1], ps.ELASTIC]), vmin=0, vmax=1, shading="auto")
        other = [k for k in ds.k_factor.values if not np.isclose(k, k_factor)]
        for kf in other:
            ax.contour(ds.W2 / 1e3, ds.ratio, d.overtopped_ratio.sel(k_factor=kf).astype(float), levels=[0.5],
                       colors=ps.INK, linestyles="--", linewidths=1.3)
        ax.axhline(ds.D0 / ds.melt_m, color=ps.INK, ls=":", lw=1.4)
        ax.text(0.22, ds.D0 / ds.melt_m * 1.06, "rigid threshold: fill depth = $D_0$", fontsize=8.5)
        ax.set(yscale="log", xlabel="lake width W (km)",
               title=rf"$\eta$ = {eta:.0e} Pa s  (k = {k_factor / np.pi:g}$\pi$/W; dashed: other k)")
        km_axis(ax)
        ax.grid(False)
    axs[0].set_ylabel(f"catchment-to-lake area ratio\n(fill depth = ratio x {ds.melt_m:g} m)")
    return fig


def main():
    ds = run()
    print("widest lake that overtops [km] (V, eta, k):", (ds.W_switch / 1e3).round(2).values,
          "rigid:", (ds.W_switch_rigid / 1e3).round(2).values)
    print(save(ds, "exp3_width"), savefig(plot_fixed(ds), "exp3_fixed_melt"),
          savefig(plot_ratio(ds), "exp3_catchment_ratio"))


if __name__ == "__main__":
    main()
