"""
lake_routing.py
===============

Glue between meltwater routing by **fill-spill-merge** and the viscoelastic
plate in ``viscoelastic_plate.py``.

The routing uses the original fill-spill-merge algorithm (Barnes, Callaghan &
Wickert 2021) through the Python wrapper ``fillspillmerge``
(github.com/jkingslake/fillspillmerge).  This module never re-implements the
routing; it only

* loads a DEM and a floating/grounded mask as ``xarray.DataArray``s,
* calls ``fill_spill_merge`` and labels the lakes it makes,
* puts a finite-difference plate (grounded ice held fixed) under the routing
  grid, moving loads down to the plate grid and deflections back up,
* runs the unloading response of a drained lake,
* runs routing and flexure together in time (each one moves the other), and
* maps which lake (or the domain edge) each cell drains to.

Conventions follow the flexure model: deflection ``w`` is positive *down*,
so the deformed surface is ``dem - w``.  Times are in seconds.  Every routine
returns xarray objects with the DEM's x, y coordinates.

Finding the wrapper
-------------------
``import fillspillmerge`` must work, either because it is installed or because
its folder (with the compiled ``build/_fsm*.so``) can be found.  This module
looks in ``$FILLSPILLMERGE_PATH``, then next to this repository
(``../fillspillmerge``, ``../fsm_pywrapper/fillspillmerge``).
"""
from __future__ import annotations

import contextlib
import heapq
import os
import sys
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import xarray as xr
from scipy import ndimage as ndi

from viscoelastic_beam import SECONDS_PER_DAY, IceShelf
from viscoelastic_plate import FDPlate

DAY = SECONDS_PER_DAY
REPO = Path(__file__).resolve().parents[1]

try:
    import numba
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False


# ===========================================================================
# Fill-spill-merge
# ===========================================================================
def _import_fsm():
    try:
        from fillspillmerge import fill_spill_merge
        return fill_spill_merge
    except ImportError:
        pass
    candidates = [os.environ.get("FILLSPILLMERGE_PATH"),
                  REPO.parent / "fillspillmerge",
                  REPO.parent / "fsm_pywrapper" / "fillspillmerge"]
    for c in candidates:
        if c and (Path(c) / "fillspillmerge" / "__init__.py").exists():
            sys.path.insert(0, str(c))
            from fillspillmerge import fill_spill_merge
            return fill_spill_merge
    raise ImportError(
        "Cannot import the fill-spill-merge wrapper.  Clone and build "
        "github.com/jkingslake/fillspillmerge, then set FILLSPILLMERGE_PATH "
        "to its folder (or pip install it).")


def fill_spill_merge(*args, **kwargs):
    """The wrapper's ``fill_spill_merge``, imported on first use."""
    global _FSM
    if "_FSM" not in globals():
        _FSM = _import_fsm()
    return _FSM(*args, **kwargs)


@contextlib.contextmanager
def _quiet_fds():
    """Silence C-level stdout/stderr (the wrapper's progress bars) for a while."""
    for f in (sys.stdout, sys.stderr):
        try:
            f.flush()
        except Exception:  # pragma: no cover
            pass
    saved = [os.dup(1), os.dup(2)]
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 1)
        os.dup2(null, 2)
        yield
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for f in saved + [null]:
            os.close(f)


def route(surface: xr.DataArray, water: xr.DataArray | float, ocean_level: float = 0.0,
          hierarchy=None, quiet: bool = True) -> xr.DataArray:
    """Route ``water`` (depth of water released in each cell [m], or a single
    number for a uniform input) over ``surface`` with fill-spill-merge.

    Returns the equilibrium water depth [m].  Cells below ``ocean_level`` and
    the domain edges are outlets: water that reaches them leaves the domain.
    Pass a ``hierarchy`` (from ``fill_spill_merge(..., return_hierarchy=True)``)
    to reuse it when the surface has not changed.  ``quiet`` hides the
    wrapper's progress output."""
    if np.ndim(water) == 0:
        water = xr.full_like(surface, float(water))
    with (_quiet_fds() if quiet else contextlib.nullcontext()):
        if hierarchy is None:
            out = fill_spill_merge(wtd=water, topography=surface, ocean_level=ocean_level)
        else:
            out = fill_spill_merge(wtd=water, hierarchy=hierarchy)
    out = xr.DataArray(np.asarray(out), coords=surface.coords, dims=surface.dims)
    return out.clip(min=0).rename("water_depth").assign_attrs(units="m", long_name="water depth")


# ===========================================================================
# Data
# ===========================================================================
def load_dem(path, coarsen: int = 1, ocean_edges: bool = True, ocean_level: float = 0.0) -> xr.DataArray:
    """Read a DEM GeoTIFF as a (y, x) DataArray in metres.  Nodata cells are
    filled from their nearest valid neighbour, and the grid is block-averaged
    by ``coarsen``.

    Fill-spill-merge only lets water leave through cells at or below the
    ocean level that connect to the domain edge; any other edge is a wall.
    With ``ocean_edges`` (default) the outermost ring of cells is set to
    ``ocean_level``, so water that reaches the edge of the domain leaves it
    (as in the repository's Nivlisen DEM)."""
    import rioxarray  # noqa: F401  (registers the .rio accessor)
    dem = xr.open_dataarray(path, engine="rasterio").squeeze(drop=True).astype("float64")
    nod = dem.attrs.get("_FillValue", dem.rio.nodata)
    z = dem.values
    bad = ~np.isfinite(z) | ((z == nod) if nod is not None else False)
    if bad.any():
        idx = ndi.distance_transform_edt(bad, return_distances=False, return_indices=True)
        z = z[tuple(idx)]
    dem = dem.copy(data=z)
    if coarsen > 1:
        dem = dem.coarsen(x=coarsen, y=coarsen, boundary="trim").mean()
    if ocean_edges:
        z = dem.values.copy()
        z[[0, -1], :] = z[:, [0, -1]] = ocean_level
        dem = dem.copy(data=z)
    dem.attrs = {"units": "m", "long_name": "surface elevation", "source": str(path)}
    return dem.rename("dem")


def floating_mask(grid: xr.DataArray, mask_path=REPO / "data" / "Mask_Antarctica_v02.tif",
                  floating_value: int = 125) -> xr.DataArray:
    """True where the ice is floating, sampled (nearest) from a mask raster on
    the same projection (default: the 500 m mask in ``data/``, where 125 is
    floating and 255 grounded)."""
    m = xr.open_dataarray(mask_path, engine="rasterio").squeeze(drop=True)
    f = (m.sel(x=grid.x, y=grid.y, method="nearest") == floating_value)
    return f.assign_coords(x=grid.x, y=grid.y).rename("floating")


def cell_area(grid: xr.DataArray) -> float:
    return abs(float(grid.x[1] - grid.x[0]) * float(grid.y[1] - grid.y[0]))


# ===========================================================================
# Lakes
# ===========================================================================
def find_lakes(depth: xr.DataArray, floating: xr.DataArray | None = None,
               min_depth: float = 0.01):
    """Label connected areas deeper than ``min_depth`` (4-connected).

    Returns ``(labels, table)``: ``labels`` is an integer DataArray (0 = dry)
    and ``table`` an ``xarray.Dataset`` along dimension ``lake`` (sorted by
    volume, largest first) with volume [m^3], area [m^2], max depth [m], the
    depth-weighted centre (x, y) and the floating fraction."""
    d = depth.values
    lab, n = ndi.label(d > min_depth)
    idx = np.arange(1, n + 1)
    a = cell_area(depth)
    vol = ndi.sum(d, lab, idx) * a
    area = ndi.sum(np.ones_like(d), lab, idx) * a
    dmax = ndi.maximum(d, lab, idx)
    ci = np.array(ndi.center_of_mass(d, lab, idx)).reshape(-1, 2)
    xc = np.interp(ci[:, 1], np.arange(depth.x.size), depth.x.values)
    yc = np.interp(ci[:, 0], np.arange(depth.y.size), depth.y.values)
    fl = (ndi.mean(np.asarray(floating, float), lab, idx) if floating is not None
          else np.full(n, np.nan))
    table = xr.Dataset(
        {"volume": ("lake", vol, {"units": "m3"}),
         "area": ("lake", area, {"units": "m2"}),
         "max_depth": ("lake", dmax, {"units": "m"}),
         "x_centre": ("lake", xc, {"units": "m"}),
         "y_centre": ("lake", yc, {"units": "m"}),
         "floating_fraction": ("lake", fl)},
        coords={"lake": idx}).sortby("volume", ascending=False)
    labels = xr.DataArray(lab, coords=depth.coords, dims=depth.dims, name="lake_label")
    return labels, table


def pick_lake(table: xr.Dataset, lake: int | None = None, near: tuple | None = None,
              min_floating: float = 0.9) -> int:
    """Choose a lake: by label, by the lake whose centre is nearest a point
    ``near=(x, y)``, or (default) the largest lake that is at least
    ``min_floating`` afloat."""
    if lake is not None:
        return int(lake)
    if near is not None:
        dist = np.hypot(table.x_centre - near[0], table.y_centre - near[1])
        return int(table.lake[int(dist.argmin())])
    ok = table.floating_fraction >= min_floating
    return int(table.lake.where(ok, drop=True)[0])


# ===========================================================================
# Plate under the routing grid
# ===========================================================================
class ShelfPlate:
    """A finite-difference Maxwell plate under a routing grid.

    The plate lives on a grid ``coarsen`` times coarser than the routing grid
    (block averages; keep its spacing below about ell/5).  Nodes that are
    mostly grounded are held at w = 0, so the grounding line acts as a
    clamped edge, as do the outer edges of the domain.

    Use ``load(depth)`` to turn a water depth on the routing grid into a load
    on the plate grid, ``start``/``advance`` to step the plate, and
    ``deflection()`` to get w back on the routing grid.
    """

    def __init__(self, grid: xr.DataArray, floating: xr.DataArray, shelf: IceShelf | None = None,
                 coarsen: int = 2, edges: str = "clamped"):
        self.shelf = shelf if shelf is not None else IceShelf()
        self.grid, self.c = grid, int(coarsen)
        coarse = grid.coarsen(x=self.c, y=self.c, boundary="trim").mean()
        self.xc, self.yc = coarse.x.values, coarse.y.values
        fl = floating.astype(float).coarsen(x=self.c, y=self.c, boundary="trim").mean().values > 0.5
        dx = abs(float(grid.x[1] - grid.x[0])) * self.c
        self.plate = FDPlate(self.xc.size, self.yc.size, dx, self.shelf, edges=edges,
                             fixed=~fl, x=self.xc, y=self.yc)
        self.dx = dx
        self.shape = (self.yc.size, self.xc.size)

    # -- moving fields between the two grids -------------------------------
    def coarse(self, field: xr.DataArray) -> np.ndarray:
        """Block-average a routing-grid field onto the plate grid."""
        return field.coarsen(x=self.c, y=self.c, boundary="trim").mean().values

    def load(self, depth: xr.DataArray) -> np.ndarray:
        """Load of a water depth [m] on the plate grid [Pa], volume-conserving."""
        return self.shelf.lake_load(self.coarse(depth))

    def to_grid(self, a: np.ndarray, name: str = "deflection") -> xr.DataArray:
        """Bilinear interpolation of a plate-grid array onto the routing grid."""
        da = xr.DataArray(np.asarray(a), coords={"y": self.yc, "x": self.xc}, dims=("y", "x"))
        out = da.interp(y=self.grid.y, x=self.grid.x, kwargs={"fill_value": "extrapolate"})
        return out.rename(name)

    def as_plate(self, a: np.ndarray, name: str) -> xr.DataArray:
        return xr.DataArray(np.asarray(a), coords={"y": self.yc, "x": self.xc},
                            dims=("y", "x"), name=name)

    # -- stepping ------------------------------------------------------------
    def start(self, q0: np.ndarray | None = None, t0: float = 0.0):
        """Elastic response to q0 (default: unloaded) at time t0."""
        self.plate.start(np.zeros(self.shape) if q0 is None else q0, t0)
        return self

    @property
    def t(self) -> float:
        return self.plate.state["t"]

    @property
    def q(self) -> np.ndarray:
        return self.plate.state["q"].reshape(self.shape)

    @property
    def w(self) -> np.ndarray:
        return self.plate.w

    def advance(self, q1: np.ndarray, dt: float, substeps: int = 1):
        """Advance by dt with the load changing linearly to q1, in ``substeps``
        Crank-Nicolson steps (keep dt/substeps well below tau)."""
        q0 = self.q
        for k in range(1, substeps + 1):
            self.plate.advance(q0 + (q1 - q0) * (k / substeps), dt / substeps)
        return self

    def deflection(self) -> xr.DataArray:
        """Current deflection on the routing grid [m, + down]."""
        return self.to_grid(self.w).assign_attrs(units="m", long_name="deflection (+ down)")

    def snapshot(self) -> xr.Dataset:
        return self.plate.snapshot()

    def restore(self, snap: xr.Dataset):
        """Set the plate state from one time of a saved run (e.g.
        ``drain_lake(...).sel(time=t)``), so it can be advanced from there."""
        self.plate.state = {"t": float(snap.time), "q": np.asarray(snap.q, float).ravel(),
                            "w": np.asarray(snap.w, float).ravel(),
                            "mu": np.asarray(snap.mu, float).reshape(2, -1)}
        return self


# ===========================================================================
# Experiment 1: the unloading response of a lake that drains
# ===========================================================================
def time_grid(t_end: float, drain_time: float = 0.5 * DAY, dt_drain: float | None = None,
              dt_early: float = 0.25 * DAY, early_until: float = 20 * DAY,
              dt_late: float = 1.0 * DAY) -> np.ndarray:
    """Time steps for a drainage run: fine while the lake drains, then
    ``dt_early`` until ``early_until``, then ``dt_late``.  Only three step
    sizes, so the plate factorises its matrix only three times."""
    dt_drain = drain_time / 10 if dt_drain is None else dt_drain
    parts = [np.arange(0.0, drain_time, dt_drain), [drain_time]]
    if t_end > drain_time:
        parts.append(np.arange(drain_time, min(early_until, t_end) + dt_early / 2, dt_early))
    if t_end > early_until:
        last = parts[-1][-1]
        parts.append(np.arange(last + dt_late, t_end + dt_late / 2, dt_late))
    return np.unique(np.round(np.concatenate(parts), 6))


def drain_lake(plate: ShelfPlate, depth: xr.DataArray, times: Sequence[float],
               drain_time: float = 0.5 * DAY, probes: dict | None = None,
               **grid_kw) -> xr.Dataset:
    """Flexural response to draining a lake of water depth ``depth``.

    The lake is assumed to have sat long enough for the shelf to relax fully
    beneath it: the deflection is then isostatic, w = q/k, and the plate stores
    no bending moment.  By linearity the state after drainage is that relaxed
    state plus the response of a plate *at rest* to the load ``-q`` (applied
    linearly over ``drain_time``).  Only that second part is computed - only
    the unloading is simulated - so every output is the *change* caused by
    draining.

    The plate is stepped on :func:`time_grid` (``grid_kw`` are passed on) and
    left in its final state, so it can be advanced further.

    Returns an ``xarray.Dataset`` on the plate grid:

    * maps with dims (time, y, x) at the output ``times`` [s after drainage
      starts; rounded onto the time grid]: ``w`` (change in deflection,
      + down), ``uplift`` (= -w), ``q`` (change in load), the surface stresses
      and the moment potentials ``mu``;
    * time series at every step (dim ``step_time``): ``uplift_max``, the mean
      uplift over the lake ``uplift_lake_mean``, the same as a fraction of its
      isostatic value ``rebound_fraction``, and the uplift at each of the
      ``probes`` ({name: (x, y)}; nearest plate node) in ``uplift_probe``;
    * the drained load ``lake_load`` and its isostatic uplift
      ``uplift_isostatic`` = q/k (the long-time limit on floating ice).
    """
    tgrid = time_grid(max(times), drain_time, **grid_kw)
    out = sorted({int(np.argmin(np.abs(tgrid - tt))) for tt in times})
    Q = plate.load(depth)
    Q[plate.plate.fixed] = 0.0
    fp = Q > 0
    iso = Q / plate.shelf.k
    iso_mean = float((iso * Q).sum() / Q.sum())
    probes = probes or {}
    pidx = [(int(np.argmin(np.abs(plate.yc - y))), int(np.argmin(np.abs(plate.xc - x))))
            for x, y in probes.values()]
    series = np.zeros((tgrid.size, 2))
    pser = np.zeros((tgrid.size, len(pidx)))
    snaps = []
    plate.start(None, 0.0)
    for i, t in enumerate(tgrid):
        if i > 0:
            f = min(t / drain_time, 1.0) if drain_time > 0 else 1.0
            plate.advance(-Q * f, t - tgrid[i - 1])
        up = -plate.w
        series[i] = up.max(), float((up * Q).sum() / Q.sum())
        pser[i] = [up[a, b] for a, b in pidx]
        if i in out:
            snaps.append(plate.snapshot())
    ds = xr.concat(snaps, "time")
    ds["uplift"] = (-ds.w).assign_attrs(units="m", long_name="uplift after drainage (+ up)")
    ds = ds.assign_coords(step_time=("step_time", tgrid, {"units": "s"}),
                          step_time_days=("step_time", tgrid / DAY, {"units": "day"}))
    ds["uplift_max"] = ("step_time", series[:, 0], {"units": "m", "long_name": "largest uplift"})
    ds["uplift_lake_mean"] = ("step_time", series[:, 1],
                              {"units": "m", "long_name": "load-weighted mean uplift over the lake"})
    ds["rebound_fraction"] = ("step_time", series[:, 1] / iso_mean,
                              {"long_name": "mean uplift / isostatic mean uplift"})
    if pidx:
        ds = ds.assign_coords(probe=("probe", list(probes)))
        ds["uplift_probe"] = (("step_time", "probe"), pser, {"units": "m"})
    ds["lake_load"] = (("y", "x"), Q, {"units": "Pa", "long_name": "load removed by drainage"})
    ds["uplift_isostatic"] = (("y", "x"), np.where(plate.plate.fixed, 0.0, iso),
                              {"units": "m", "long_name": "isostatic uplift q/k (long-time limit)"})
    ds.attrs.update(experiment="lake drainage from a fully relaxed state (unloading only)",
                    drain_time_s=float(drain_time),
                    lake_volume_m3=float(depth.sum()) * cell_area(depth),
                    plate_dx=plate.dx)
    return ds


# ===========================================================================
# Experiment 2: routing and flexure together
# ===========================================================================
def route_with_flexure(dem: xr.DataArray, melt: Callable[[int], xr.DataArray], times: Sequence[float],
                       plate: ShelfPlate | None = None, water0: xr.DataArray | None = None,
                       extra_load: Callable[[float], np.ndarray] | None = None,
                       reference_water: xr.DataArray | None = None,
                       ocean_level: float = 0.0, substeps: int = 4,
                       store: Sequence[int] | None = None, track: dict | None = None,
                       verbose: bool = False) -> xr.Dataset:
    """Route meltwater and flex the shelf together, one interval at a time.

    In each interval [t_i, t_{i+1}]:

    1. the water already on the surface plus the melt ``melt(i)`` [m] is
       routed by fill-spill-merge over the *current* surface ``dem - w``;
    2. the plate is advanced to t_{i+1} under the new water load (plus
       ``extra_load(t)``, e.g. the removal of a drained lake), so w changes
       and with it the surface used for the next routing.

    With ``plate=None`` the surface stays fixed (routing on a rigid shelf).
    A ``plate`` must already be started (at ``times[0]``).  Water reaching the
    domain edge or the ocean is counted as lost.

    The plate is loaded by ``water - reference_water``: the water the DEM
    does *not* already account for.  Use the default (no water) when the DEM
    is the unloaded surface; pass the lakes the DEM was relaxed under when it
    is a relaxed, loaded surface (then a drained lake shows up as a negative
    load).

    Returns an ``xarray.Dataset``.  Maps on the routing grid with dims
    (time, y, x) are kept at the indices ``store`` (default all):
    ``water_depth``, ``deflection`` and ``surface``.  Time series on the full
    time axis ``step_time``: melt added, water stored, water lost, the
    largest deflection, and (if ``track`` = {name: boolean mask} is given) the
    water volume inside each mask, ``tracked_volume``.
    """
    times = np.asarray(times, float)
    nt = times.size
    store = list(range(nt)) if store is None else sorted(store)
    a = cell_area(dem)
    water = xr.zeros_like(dem) if water0 is None else water0
    w = plate.deflection() if plate is not None else xr.zeros_like(dem)
    Wd = np.zeros((len(store),) + dem.shape, np.float32)
    Wf, S = np.zeros_like(Wd), np.zeros_like(Wd)
    added, stored, lost, wmax = (np.zeros(nt) for _ in range(4))
    track = track or {}
    masks = [np.asarray(m, bool) for m in track.values()]
    tracked = np.zeros((nt, len(masks)))
    tracked[0] = [float(water.values[m].sum()) * a for m in masks]
    stored[0] = float(water.sum()) * a
    wmax[0] = float(w.max())

    def keep(i, water, w):
        if i in store:
            j = store.index(i)
            Wd[j], Wf[j], S[j] = water.values, w.values, (dem - w).where(dem > ocean_level, dem).values

    keep(0, water, w)
    for i in range(nt - 1):
        dt = times[i + 1] - times[i]
        m = melt(i)
        inp = water + m
        surface = (dem - w).where(dem > ocean_level, dem)   # outlets stay outlets
        water = route(surface, inp, ocean_level)
        added[i + 1] = added[i] + float(m.sum()) * a
        stored[i + 1] = float(water.sum()) * a
        lost[i + 1] = lost[i] + (float(inp.sum()) - float(water.sum())) * a
        if plate is not None:
            q = plate.load(water if reference_water is None else water - reference_water)
            if extra_load is not None:
                q = q + extra_load(times[i + 1])
            plate.advance(q, dt, substeps)
            w = plate.deflection()
        wmax[i + 1] = float(w.max())
        tracked[i + 1] = [float(water.values[m].sum()) * a for m in masks]
        keep(i + 1, water, w)
        if verbose and (i + 1) % 10 == 0:
            print(f"  day {times[i + 1] / DAY:6.1f}: stored {stored[i + 1] / 1e6:8.1f} Mm3, "
                  f"lost {lost[i + 1] / 1e6:8.1f} Mm3, max deflection {wmax[i + 1]:6.2f} m")

    tt = times[store]
    coords = {"time": ("time", tt, {"units": "s"}), "time_days": ("time", tt / DAY, {"units": "day"}),
              "y": dem.y, "x": dem.x,
              "step_time_days": ("step_time", times / DAY, {"units": "day"})}
    ds = xr.Dataset(
        {"water_depth": (("time", "y", "x"), Wd, {"units": "m", "long_name": "water depth"}),
         "deflection": (("time", "y", "x"), Wf, {"units": "m", "long_name": "deflection (+ down)"}),
         "surface": (("time", "y", "x"), S, {"units": "m", "long_name": "deformed surface (dem - w)"}),
         "melt_added": ("step_time", added, {"units": "m3", "long_name": "cumulative melt input"}),
         "water_stored": ("step_time", stored, {"units": "m3", "long_name": "water held on the surface"}),
         "water_lost": ("step_time", lost, {"units": "m3", "long_name": "cumulative water lost at edges/ocean"}),
         "max_deflection": ("step_time", wmax, {"units": "m"})},
        coords=coords,
        attrs={"coupled": int(plate is not None), "substeps": substeps, "ocean_level": ocean_level})
    ds = ds.assign_coords(step_time=("step_time", times, {"units": "s"}))
    if masks:
        ds = ds.assign_coords(region=("region", list(track)))
        ds["tracked_volume"] = (("step_time", "region"), tracked,
                                {"units": "m3", "long_name": "water held inside each tracked region"})
    if plate is not None:
        ds.attrs.update({f"shelf_{k}": float(v) for k, v in
                         dict(H=plate.shelf.H, E=plate.shelf.E, eta=plate.shelf.eta,
                              nu=plate.shelf.nu, nu_v=plate.shelf.nu_v).items()})
        ds.attrs["plate_dx"] = plate.dx
    return ds


# ===========================================================================
# Which lake does each cell drain to?
# ===========================================================================
_OFFSETS = np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)])


def _flood_python(z, labels):
    ny, nx = z.shape
    lab = labels.copy()
    parent = np.full(z.shape, -1, np.int64)
    heap = [(z[i, j], i, j) for i, j in zip(*np.nonzero(lab))]
    heapq.heapify(heap)
    while heap:
        h, i, j = heapq.heappop(heap)
        for di, dj in _OFFSETS:
            a, b = i + di, j + dj
            if 0 <= a < ny and 0 <= b < nx and lab[a, b] == 0:
                lab[a, b] = lab[i, j]
                parent[a, b] = i * nx + j
                heapq.heappush(heap, (max(z[a, b], h), a, b))
    return lab, parent


if HAVE_NUMBA:
    @numba.njit(cache=True)
    def _flood_numba(z, labels):  # pragma: no cover - same algorithm as above
        ny, nx = z.shape
        lab = labels.copy()
        parent = np.full(z.shape, -1, np.int64)
        heap = [(0.0, 0, 0)]
        heap.pop()
        for i in range(ny):
            for j in range(nx):
                if lab[i, j] != 0:
                    heap.append((z[i, j], i, j))
        heapq.heapify(heap)
        while len(heap) > 0:
            h, i, j = heapq.heappop(heap)
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    a, b = i + di, j + dj
                    if (di != 0 or dj != 0) and 0 <= a < ny and 0 <= b < nx and lab[a, b] == 0:
                        lab[a, b] = lab[i, j]
                        parent[a, b] = i * nx + j
                        heapq.heappush(heap, (max(z[a, b], h), a, b))
        return lab, parent


def drainage_basins(surface: xr.DataArray, lakes: xr.DataArray, edge_label: int = -1,
                    ocean_level: float = 0.0, return_receivers: bool = False):
    """Which lake each cell drains to.

    Every labelled lake (``lakes`` > 0), the domain edge and the ocean
    (cells at or below ``ocean_level``) are outlets.  A priority flood
    (Barnes et al. 2014) rises from all outlets at once over ``surface``,
    filling every other depression on the way, so each cell gets the label of
    the outlet its water reaches once the small, unlabelled depressions have
    filled and spilled.  Cells that drain off the edge or into the ocean get
    ``edge_label``.

    With ``return_receivers`` it also returns, for each cell, the flat index
    of the neighbour its water flows on to (-1 at outlets); follow them with
    :func:`flow_route`.  The route from inside a depression climbs to its
    spill point and continues downhill from there, so it is the path the
    depression's overflow takes once it is full.
    """
    z = np.asarray(surface, float)
    lab = np.asarray(lakes, np.int64).copy()
    lab[lab < 0] = 0
    border = np.zeros(z.shape, bool)
    border[[0, -1], :] = border[:, [0, -1]] = True
    lab[((z <= ocean_level) | border) & (lab == 0)] = edge_label
    out, parent = _flood_numba(z, lab) if HAVE_NUMBA else _flood_python(z, lab)
    basins = xr.DataArray(out, coords=surface.coords, dims=surface.dims, name="drains_to")
    if return_receivers:
        return basins, xr.DataArray(parent, coords=surface.coords, dims=surface.dims, name="receiver")
    return basins


def flow_route(receivers: xr.DataArray, x: float, y: float) -> xr.Dataset:
    """The path water takes from the cell nearest (x, y) to an outlet,
    following ``receivers`` from :func:`drainage_basins`.  Returns the
    coordinates of the cells along it (dimension ``step``)."""
    rec = np.asarray(receivers).ravel()
    nx = receivers.sizes["x"]
    i = int(np.argmin(np.abs(receivers.y.values - y))) * nx + int(np.argmin(np.abs(receivers.x.values - x)))
    path = [i]
    while rec[path[-1]] >= 0 and len(path) <= rec.size:
        path.append(int(rec[path[-1]]))
    p = np.array(path)
    return xr.Dataset({"x": ("step", receivers.x.values[p % nx]), "y": ("step", receivers.y.values[p // nx])})


def lake_for(labels: xr.DataArray, footprint: xr.DataArray) -> np.ndarray:
    """Labels of the lakes in ``labels`` that overlap ``footprint``."""
    v = np.unique(labels.values[np.asarray(footprint, bool)])
    return v[v > 0]
