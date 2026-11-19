"""
gmt_flexure.py
==============

Run GMT's ``grdflexure`` (Wessel's FFT flexure module, GMT >= 6.5) on the
ice-shelf problem, for cross-verification of ``viscoelastic_plate.py``.

The equations solved here are
    D lap^2 w + rho_sw g w = rho_w g d     (elastic), and
the Maxwell viscoelastic plate over an inviscid half-space (``-M``), whose
response to a load switched on at t = 0 is, mode by mode,
    w(k, t) / w_iso = 1 - [1 - Phi_e] exp(-Phi_e t / t_m),  Phi_e = 1/(1 + (k ell)^4).
That is the single-relaxation-time plate of ``viscoelastic_beam.py`` with
nu_v = nu and t_m = tau = 4 eta (1 - nu^2) / E.

Density mapping.  With its moat infill equal to the load density, grdflexure
solves  D lap^2 w + (rho_m - rho_l) g w = (rho_l - rho_w) g h  for a load
"topography" h.  Choosing the dummy densities rho_l = 2000, rho_m = 2000 +
rho_sw and rho_w = 2000 - rho_water, and h = lake depth d, gives exactly our
equation.

Time.  grdflexure's time axis is *age*, positive back in time.  A load history
is given as a list of load increments, each switched on at its own time;
grdflexure superposes their responses.  (In GMT 6.5, ``-T<file>`` and
filename-template inputs crash; a load list ``<file>+l`` with tab-separated
``age  grid  tag`` records and a regular ``-T t0/t1/dt`` sequence works.)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import numpy as np
import xarray as xr

YEAR = 365.25 * 86400.0
RHO_DUMMY = 2000.0
#: gravity used inside grdflexure.  Found empirically: with this value the elastic
#: response matches SpectralPlate to float32 round-off (with 9.81 the misfit is ~1e-4).
GMT_G = 9.806199203


def gmt_available() -> bool:
    return shutil.which("gmt") is not None


def gmt_version() -> str:
    return subprocess.run(["gmt", "--version"], capture_output=True, text=True).stdout.strip()


def _densities(rho_sw=1028.0, rho_water=1000.0):
    return f"{RHO_DUMMY + rho_sw:g}/{RHO_DUMMY:g}/{RHO_DUMMY:g}/{RHO_DUMMY - rho_water:g}"


def _write(da, path):
    da = xr.DataArray(np.asarray(da, dtype=np.float64), coords={"y": da.y.values, "x": da.x.values},
                      dims=("y", "x"), name="z")
    da.to_netcdf(path)


def grdflexure(load_increments, out_times, H, E=1.0e9, nu=0.3, tm=None, rho_sw=1028.0,
               rho_water=1000.0, fft="-Nf+n+l", keep=None) -> xr.DataArray:
    """Deflection (positive down) from grdflexure.

    load_increments : list of (t [s], depth increment [m] as a (y, x) DataArray)
                      - the lake depth changes by that amount at time t
    out_times       : output times [s]; they must lie on a regular grid
                      (grdflexure's -T t0/t1/dt)
    H, E, nu        : plate thickness [m], Young's modulus [Pa], Poisson's ratio
    tm              : Maxwell time [s], or None for a purely elastic plate (then
                      the response at every output time is elastic)
    fft             : grdflexure -N option.  The default, ``-Nf+n+l``, transforms
                      the grid as it is - no extension, taper or detrending - i.e.
                      a periodic domain, like ``SpectralPlate``.
    keep            : optional folder to keep GMT's input and output files in
    """
    out_times = np.sort(np.asarray(out_times, float))
    t_ref = out_times[-1]
    work = keep or tempfile.mkdtemp(prefix="grdflexure_")
    os.makedirs(work, exist_ok=True)
    try:
        lines = []
        for j, (t, dd) in enumerate(load_increments):
            age = (t_ref - t) / YEAR
            name = f"load_{j:04d}.nc"
            _write(dd, os.path.join(work, name))
            lines.append(f"{age:.12g}\t{name}\t{age:.12g}")
        with open(os.path.join(work, "loads.lis"), "w") as f:
            f.write("\n".join(lines) + "\n")
        ages = (t_ref - out_times) / YEAR
        if out_times.size > 1:
            steps = np.diff(out_times)
            if not np.allclose(steps, steps[0], rtol=1e-9):
                raise ValueError("out_times must be regularly spaced for grdflexure -T")
            T = f"-T{ages[0]:.12g}/{ages[-1]:.12g}/{steps[0] / YEAR:.12g}"
        else:
            T = f"-T{ages[0]:.12g}"
        cmd = ["gmt", "grdflexure", "loads.lis+l", "-Gw_%.12g.nc", T, f"-D{_densities(rho_sw, rho_water)}",
               f"-E{H:g}", f"-Cy{E:g}", f"-Cp{nu:g}", fft, "-Lout.lis"]
        if tm is not None:
            cmd.append(f"-M{tm / YEAR:.12g}")
        r = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"grdflexure failed:\n{' '.join(cmd)}\n{r.stderr[-2000:]}")
        out = {}
        for line in open(os.path.join(work, "out.lis")):
            parts = line.split()
            if len(parts) >= 2 and not line.startswith("#"):
                out[float(parts[0])] = parts[1]
        frames = []
        for a in ages:
            key = min(out, key=lambda v: abs(v - a))
            g = xr.open_dataarray(os.path.join(work, out[key]), engine="netcdf4").load()
            frames.append(-g.values.astype(np.float64))          # GMT: negative down
        da0 = load_increments[0][1]
        return xr.DataArray(np.array(frames), dims=("time", "y", "x"),
                            coords={"time": out_times, "time_days": ("time", out_times / 86400.0),
                                    "y": da0.y.values, "x": da0.x.values},
                            name="w", attrs={"units": "m", "long_name": "deflection (+ down), GMT grdflexure",
                                             "gmt_version": gmt_version(), "command": " ".join(cmd)})
    finally:
        if keep is None:
            shutil.rmtree(work, ignore_errors=True)
