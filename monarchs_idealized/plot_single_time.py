"""
Profiles at a single time, showing what each part of the column is made of.

MONARCHS stores a column as three separate pieces:
  - lid  : solid ice (thickness + temperature profile only)
  - lake : liquid water (thickness + temperature profile only)
  - firn : layers with an ice fraction (Sfrac) and liquid fraction (Lfrac);
           the remainder is air
Here they are stitched into one ice/water/air composition column, so the lake
shows up as 100 % water and the lid as 100 % ice.

Note on the "ice lens": during percolation, a firn layer whose ice density
exceeds pore close-off (830 kg m^-3) is flagged as an impermeable ice lens;
water saturates the firn above it and the excess at the surface becomes lake
depth (exposed_water = True). Once exposed_water is set, MONARCHS stops calling
the firn hydrology (firn_column / percolation) for that cell: under a lake the
firn only conducts heat (top at 0 C) and refreezes liquid it already holds.
So the firn under a lake stays dry because percolation is switched off, and
the ice-lens flag (ice_lens_depth) is sticky even if the current top layer has
been melted down to ice density < 830.

Usage:
    python plot_single_time.py [day] [output/idealized_output.nc]
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from netCDF4 import Dataset

DAY = int(sys.argv[1]) if len(sys.argv) > 1 else 30
path = sys.argv[2] if len(sys.argv) > 2 else "output/idealized_output.nc"
T0 = 273.15
RHO_ICE = 917.0
PORE_CLOSURE = 830.0  # monarchs.physics.constants.pore_closure
FIRN_SHOWN = 2.5  # metres of firn shown below the firn surface

ROW = 7
LOCATIONS = {
    "Upslope (col 2)": (ROW, 2),
    "Mid-slope (col 6)": (ROW, 6),
    "Basin centre (col 10)": (ROW, 10),
    "Downslope (col 13)": (ROW, 13),
}
C_ICE, C_WATER, C_AIR = "#b8c2cf", "#2a78d6", "#fbf7ee"

with Dataset(path) as ds:
    get = lambda k: np.asarray(ds.variables[k][DAY])  # noqa: E731
    fd_all, ld_all, lid_all = get("firn_depth"), get("lake_depth"), get("lid_depth")
    S_all, L_all, T_all = get("Sfrac"), get("Lfrac"), get("firn_temperature")
    Tlake_all, Tlid_all = get("lake_temperature"), get("lid_temperature")
    lens_all = get("ice_lens_depth")


def stairs_x(values):
    """Repeat per-layer values so they plot as flat bands between edges."""
    return np.repeat(values, 2)


fig, axes = plt.subplots(
    len(LOCATIONS), 3, figsize=(13, 16), layout="constrained",
    gridspec_kw={"width_ratios": [1.3, 1, 1]},
)

for r, (name, (i, j)) in enumerate(LOCATIONS.items()):
    fd, ld, lid = fd_all[i, j], ld_all[i, j], lid_all[i, j]
    S, L, T = S_all[i, j], L_all[i, j], T_all[i, j] - T0
    nz = len(S)
    dz = fd / nz
    # layer k spans [fd - (k+1) dz, fd - k dz]; index 0 is the top layer
    top_edges = fd - np.arange(nz) * dz
    z_st = np.ravel(np.column_stack([top_edges, top_edges - dz]))
    keep = z_st >= fd - FIRN_SHOWN - dz
    lens = int(lens_all[i, j])
    has_lens = lens < nz

    # --- (a) composition -------------------------------------------------
    ax = axes[r, 0]
    ice, wat = stairs_x(S)[keep], stairs_x(L)[keep]
    z = z_st[keep]
    ax.fill_betweenx(z, 0, ice, color=C_ICE, lw=0)
    ax.fill_betweenx(z, ice, ice + wat, color=C_WATER, lw=0)
    ax.fill_betweenx(z, ice + wat, 1, color=C_AIR, lw=0)
    if ld > 0:
        ax.fill_betweenx([fd, fd + ld], 0, 1, color=C_WATER, lw=0)
        ax.text(0.5, fd + ld / 2, f"LAKE\n100 % water\n{ld:.2f} m deep",
                ha="center", va="center", color="white", fontsize=9, weight="bold")
    if lid > 0:
        ax.fill_betweenx([fd + ld, fd + ld + lid], 0, 1, color=C_ICE, lw=0)
        ax.text(0.5, fd + ld + lid / 2, f"LID (ice) {lid:.2f} m",
                ha="center", va="center", fontsize=8)
    ax.axhline(fd, color="k", lw=1)
    ax.text(1.02, fd, "firn surface", transform=ax.get_yaxis_transform(),
            va="center", fontsize=8)
    if has_lens:
        z_lens = fd - lens * dz
        ax.axhspan(z_lens - dz, z_lens, facecolor="none", edgecolor="#c0392b",
                   hatch="///", lw=1.2)
        ax.annotate("layer flagged as ice lens\n(top firn layer, pores\nfull of water)",
                    xy=(1, z_lens - dz / 2), xycoords=("axes fraction", "data"),
                    xytext=(1.04, z_lens - 0.9), textcoords=("axes fraction", "data"),
                    fontsize=8, color="#c0392b", va="top",
                    arrowprops=dict(arrowstyle="->", color="#c0392b"))
    below = L[(lens + 1 if has_lens else 0):]
    firn_label = "FIRN: ice + air"
    firn_label += (",\nno liquid water" if below.max() < 1e-6 else
                   f",\nliquid fraction up to {below.max():.2f}")
    if ld > 0:
        firn_label += "\n(no percolation\nbeneath a lake)"
    ax.text(0.5, fd - FIRN_SHOWN * 0.55, firn_label, ha="center", va="center", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Volume fraction (ice | water | air)")
    ax.set_ylabel("Elevation above base [m]")
    ax.set_title(name, loc="left", fontsize=11, weight="bold")

    # --- (b) temperature ---------------------------------------------------
    ax = axes[r, 1]
    zc = top_edges - dz / 2
    m = zc >= fd - FIRN_SHOWN - dz
    ax.plot(T[m], zc[m], color="k", lw=1.5, label="firn")
    if ld > 0:
        n = len(Tlake_all[i, j])
        ax.plot(Tlake_all[i, j] - T0, fd + ld * (1 - np.arange(n) / (n - 1)),
                color=C_WATER, lw=2, label="lake")
    if lid > 0:
        n = len(Tlid_all[i, j])
        ax.plot(Tlid_all[i, j] - T0, fd + ld + lid * (1 - np.arange(n) / (n - 1)),
                color="0.45", lw=2, label="lid")
    ax.axvline(0, color="0.6", lw=0.8, ls=":")
    ax.axhline(fd, color="k", lw=0.6)
    ax.set_xlabel("Temperature [°C]")
    ax.legend(fontsize=8, loc="lower left", frameon=False)

    # --- (c) ice density vs pore close-off ----------------------------------
    ax = axes[r, 2]
    ax.plot(stairs_x(S * RHO_ICE)[keep], z, color="k", lw=1.5)
    ax.axvline(PORE_CLOSURE, color="#c0392b", lw=1, ls="--")
    ax.text(PORE_CLOSURE + 2, fd - FIRN_SHOWN * 0.95, "pore close-off\n830 kg m⁻³\n(lens threshold)",
            color="#c0392b", fontsize=8, va="bottom")
    ax.axhline(fd, color="k", lw=0.6)
    if has_lens:
        ax.axhspan(z_lens - dz, z_lens, color="#c0392b", alpha=0.15, lw=0)
    ax.set_xlim(780, 940)
    ax.set_xlabel("Ice density, Sfrac × 917 [kg m⁻³]")

    top = fd + ld + lid
    for ax in axes[r]:
        ax.set_ylim(fd - FIRN_SHOWN, top + 0.25)
        ax.grid(alpha=0.2, lw=0.5)

fig.legend(
    handles=[Patch(color=C_ICE, label="ice"), Patch(color=C_WATER, label="water"),
             Patch(facecolor=C_AIR, edgecolor="0.6", label="air"),
             Patch(facecolor="none", edgecolor="#c0392b", hatch="///", label="layer flagged as ice lens")],
    loc="outside lower center", ncol=4, frameon=False,
)
fig.suptitle(
    f"Column structure on day {DAY}: lid / lake / top {FIRN_SHOWN} m of firn\n"
    "Firn layers are ~0.2 m thick; the lake and lid are separate model "
    "columns stacked on top of the firn",
)
out = f"profiles_day{DAY:03d}.png"
fig.savefig(out, dpi=110)
print("Saved", out)
