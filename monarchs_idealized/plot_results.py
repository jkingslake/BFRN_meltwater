"""
Quick-look plots of the idealized MONARCHS run.

Usage:
    python plot_results.py [output/idealized_output.nc]
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

path = sys.argv[1] if len(sys.argv) > 1 else "output/idealized_output.nc"

with Dataset(path) as ds:
    firn_depth = np.asarray(ds.variables["firn_depth"][:])
    lake_depth = np.asarray(ds.variables["lake_depth"][:])
    lid_depth = np.asarray(ds.variables["lid_depth"][:])

nt = firn_depth.shape[0]
days = np.arange(nt)
surface = firn_depth + lake_depth + lid_depth

# Which time is the lake extent largest?
lake_area = (lake_depth > 0.01).sum(axis=(1, 2))
i_peak = int(np.argmax(lake_area)) if lake_area.max() > 0 else nt - 1

fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)

ax = axes[0, 0]
im = ax.imshow(firn_depth[0], cmap="viridis")
ax.set_title("Initial surface (firn depth) [m]")
fig.colorbar(im, ax=ax)

ax = axes[0, 1]
im = ax.imshow(lake_depth[i_peak], cmap="Blues")
ax.set_title(f"Lake depth, day {i_peak} (max extent) [m]")
fig.colorbar(im, ax=ax)

ax = axes[0, 2]
im = ax.imshow(lid_depth[-1], cmap="Greys")
ax.set_title(f"Frozen lid depth, day {nt - 1} [m]")
fig.colorbar(im, ax=ax)

ax = axes[1, 0]
im = ax.imshow(surface[-1] - surface[0], cmap="RdBu_r", vmin=-np.abs(surface[-1] - surface[0]).max(), vmax=np.abs(surface[-1] - surface[0]).max())
ax.set_title("Change in surface height, end - start [m]")
fig.colorbar(im, ax=ax)

ax = axes[1, 1]
ax.plot(days, lake_depth.max(axis=(1, 2)), label="max lake depth")
ax.plot(days, lid_depth.max(axis=(1, 2)), label="max lid depth")
ax.plot(days, lake_depth.mean(axis=(1, 2)), "--", label="mean lake depth")
ax.set_xlabel("day")
ax.set_ylabel("m")
ax.legend()
ax.set_title("Lake / lid evolution")

ax = axes[1, 2]
mid = firn_depth.shape[1] // 2
ax.plot(firn_depth[0, mid], "k-", label="firn, day 0")
ax.plot(firn_depth[-1, mid], "k--", label=f"firn, day {nt - 1}")
ax.plot(surface[i_peak, mid], "b-", label=f"surface incl. lake, day {i_peak}")
ax.set_xlabel("column index (downslope →)")
ax.set_ylabel("height [m]")
ax.legend()
ax.set_title("Centre-line profile")

fig.savefig("idealized_results.png", dpi=120)
print(f"Saved idealized_results.png ({nt} daily outputs, peak lake extent day {i_peak}: {lake_area[i_peak]} cells)")
