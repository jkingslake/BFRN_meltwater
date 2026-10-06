"""
Vertical profiles from the idealized MONARCHS run at a few locations/times.

Needs the firn/lake/lid profile variables in vars_to_save (see model_setup.py).

Usage:
    python plot_profiles.py [output/idealized_output.nc]

Produces:
    profiles_full_column.png   - snapshots over the whole column
    profiles_near_surface.png  - same, zoomed on the top ~12 m
    profiles_depth_time.png    - depth-time sections of temperature (firn + lake + lid)
                                 and near-surface liquid water

All profiles are plotted against elevation above the column base [m], so the
firn surface moving down (melt) and lakes/lids building on top are visible.
MONARCHS profile index 0 is the top of each sub-column (firn, lake, lid).
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from netCDF4 import Dataset

path = sys.argv[1] if len(sys.argv) > 1 else "output/idealized_output.nc"
WARM_DAYS = 50  # must match model_setup.py
T0 = 273.15

# Locations along the centre line (row, col): upslope -> basin -> downslope
ROW = 7
LOCATIONS = {
    "Upslope (col 2)": (ROW, 2),
    "Mid-slope (col 6)": (ROW, 6),
    "Basin centre (col 10)": (ROW, 10),
    "Downslope (col 13)": (ROW, 13),
}
DAYS = [0, 5, 15, 30, 50, 65, 80, 100]
# warm-period days: blue ramp, cold-period days: orange ramp (each light -> dark)
WARM_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
COLD_RAMP = [matplotlib.colormaps["Oranges"](x) for x in (0.45, 0.7, 0.95)]

with Dataset(path) as ds:
    v = {k: np.asarray(ds.variables[k][:]) for k in (
        "firn_depth", "lake_depth", "lid_depth", "firn_temperature", "rho",
        "Sfrac", "Lfrac", "lake_temperature", "lid_temperature",
    )}
nt = v["firn_depth"].shape[0]
DAYS = [d for d in DAYS if d < nt]
_w = [d for d in DAYS if d <= WARM_DAYS]
_c = [d for d in DAYS if d > WARM_DAYS]
RAMP = WARM_RAMP[-len(_w):] + COLD_RAMP[: len(_c)]


def column(t, i, j):
    """Elevations and profiles for one cell at one output time."""
    fd, ld, lid = v["firn_depth"][t, i, j], v["lake_depth"][t, i, j], v["lid_depth"][t, i, j]
    nz = v["firn_temperature"].shape[-1]
    z_firn = fd * (1 - np.arange(nz) / (nz - 1))  # top -> base
    out = {
        "z_firn": z_firn,
        "T_firn": v["firn_temperature"][t, i, j] - T0,
        "rho": v["rho"][t, i, j],
        "Lfrac": v["Lfrac"][t, i, j],
        "air": np.clip(1 - v["Sfrac"][t, i, j] - v["Lfrac"][t, i, j], 0, None),
        "firn_top": fd,
        "lake_top": fd + ld,
        "lid_top": fd + ld + lid,
    }
    if ld > 0:
        nzl = v["lake_temperature"].shape[-1]
        out["z_lake"] = fd + ld * (1 - np.arange(nzl) / (nzl - 1))
        out["T_lake"] = v["lake_temperature"][t, i, j] - T0
    if lid > 0:
        nzd = v["lid_temperature"].shape[-1]
        out["z_lid"] = fd + ld + lid * (1 - np.arange(nzd) / (nzd - 1))
        out["T_lid"] = v["lid_temperature"][t, i, j] - T0
    return out


ROWS = [
    ("T", "Temperature [°C]"),
    ("rho", "Bulk density [kg m⁻³]"),
    ("Lfrac", "Liquid water fraction"),
    ("air", "Air (pore) fraction"),
]


def snapshot_figure(fname, zoom=None):
    fig, axes = plt.subplots(
        len(ROWS), len(LOCATIONS), figsize=(15, 15.5), sharey="col", layout="constrained"
    )
    for c, (name, (i, j)) in enumerate(LOCATIONS.items()):
        for r, (key, label) in enumerate(ROWS):
            ax = axes[r, c]
            for d, color in zip(DAYS, RAMP):
                col = column(d, i, j)
                ls = "-" if d <= WARM_DAYS else "--"
                if key == "T":
                    ax.plot(col["T_firn"], col["z_firn"], color=color, ls=ls, lw=1.5)
                    if "T_lake" in col:
                        ax.plot(col["T_lake"], col["z_lake"], color=color, ls=ls, lw=1.5)
                    if "T_lid" in col:
                        ax.plot(col["T_lid"], col["z_lid"], color=color, ls=ls, lw=1.5)
                else:
                    ax.plot(col[key], col["z_firn"], color=color, ls=ls, lw=1.5)
                # tick showing the top of water/ice above the firn at this time
                if col["lid_top"] > col["firn_top"] + 0.01:
                    ax.plot(1.0, col["lid_top"], marker="<", ms=6, color=color,
                            transform=ax.get_yaxis_transform(), clip_on=False)
            ax.grid(alpha=0.25, lw=0.5)
            if r == 0:
                ax.set_title(name)
            if c == 0:
                ax.set_ylabel("Elevation above base [m]")
            ax.set_xlabel(label)
            if key == "T":
                ax.axvline(0, color="0.6", lw=0.8)
        if zoom is not None:
            top = max(column(d, i, j)["lid_top"] for d in DAYS)
            axes[0, c].set_ylim(top - zoom, top + 0.5)

    handles = [Line2D([], [], color=c, lw=2, ls="-" if d <= WARM_DAYS else "--",
                      label=f"day {d}") for d, c in zip(DAYS, RAMP)]
    handles.append(Line2D([], [], color="0.3", marker="<", ls="none",
                          label="top of lake/lid"))
    fig.suptitle(
        "MONARCHS idealized run: vertical profiles "
        f"({'top ' + str(zoom) + ' m' if zoom else 'full column'}; "
        f"blue/solid = warm period, orange/dashed = cold period after day {WARM_DAYS})"
    )
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False)
    fig.savefig(fname, dpi=110)
    plt.close(fig)
    print("Saved", fname)


def depth_time_figure(fname):
    from matplotlib.colors import TwoSlopeNorm

    nz = v["firn_temperature"].shape[-1]
    nzl = v["lake_temperature"].shape[-1]
    nzd = v["lid_temperature"].shape[-1]
    days = np.arange(nt)
    norm = TwoSlopeNorm(vmin=-20, vcenter=0, vmax=4)
    wet_depth = 2.0  # m below firn surface shown in the liquid-water panels
    fig, axes = plt.subplots(len(LOCATIONS), 2, figsize=(14, 13), sharex=True,
                             layout="constrained")
    for r, (name, (i, j)) in enumerate(LOCATIONS.items()):
        fd = v["firn_depth"][:, i, j]
        top_lake = fd + v["lake_depth"][:, i, j]
        top_lid = top_lake + v["lid_depth"][:, i, j]

        # (a) temperature through firn + lake + lid, each sub-column on its own mesh
        ax = axes[r, 0]
        for base, top, T, n in [
            (np.zeros_like(fd), fd, v["firn_temperature"][:, i, j], nz),
            (fd, top_lake, v["lake_temperature"][:, i, j], nzl),
            (top_lake, top_lid, v["lid_temperature"][:, i, j], nzd),
        ]:
            Z = top[:, None] - (top - base)[:, None] * np.arange(n)[None, :] / (n - 1)
            D = np.broadcast_to(days[:, None], Z.shape)
            pc_T = ax.pcolormesh(D, Z, T - T0, cmap="RdBu_r", norm=norm, shading="gouraud")
        ax.plot(days, fd, color="k", lw=1, label="firn surface")
        ax.plot(days, top_lake, color="k", lw=0.8, ls="--", label="lake surface / lid base")
        ax.plot(days, top_lid, color="k", lw=0.8, ls=":", label="lid surface")
        ymax = top_lid.max()
        ax.set_ylim(ymax - 15, ymax + 0.5)
        ax.set_ylabel("Elevation above base [m]")
        ax.set_title(f"{name}: temperature (firn + lake + lid)", fontsize=10)

        # (b) liquid water fraction vs depth below the firn surface
        ax = axes[r, 1]
        dz = fd / (nz - 1)
        depth = dz[:, None] * np.arange(nz)[None, :]
        D = np.broadcast_to(days[:, None], depth.shape)
        pc_L = ax.pcolormesh(D, depth, v["Lfrac"][:, i, j], cmap="Blues", vmin=0, vmax=0.12,
                             shading="nearest")
        ax.set_ylim(wet_depth, 0)
        ax.set_ylabel("Depth below firn surface [m]")
        ax.set_title(f"{name}: liquid water fraction in firn", fontsize=10)

        for ax in axes[r]:
            ax.axvline(WARM_DAYS, color="0.4", ls=":", lw=1)
    axes[-1, 0].set_xlabel("Day")
    axes[-1, 1].set_xlabel("Day")
    fig.colorbar(pc_T, ax=axes[:, 0], location="top", shrink=0.6, label="Temperature [°C]",
                 extend="both")
    fig.colorbar(pc_L, ax=axes[:, 1], location="top", shrink=0.6, label="Liquid water fraction")
    axes[0, 0].legend(loc="lower left", fontsize=8, frameon=True, framealpha=0.8)
    fig.suptitle(f"Depth-time sections (dotted vertical line = end of warm period, day {WARM_DAYS})")
    fig.savefig(fname, dpi=110)
    plt.close(fig)
    print("Saved", fname)


snapshot_figure("profiles_full_column.png")
snapshot_figure("profiles_near_surface.png", zoom=12)
depth_time_figure("profiles_depth_time.png")
