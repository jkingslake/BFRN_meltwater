"""Helpers for looking for lake-load deflection of an ice shelf with ICESat-2.

Used by ``icesat2_basin_deflection.ipynb``.  Three groups of tools:

* **ICESat-2** (through SlideRule, no Earthdata login needed): ATL06 land-ice
  segments and ATL03 photons in a polar-stereographic box, cached as parquet;
  reference fields (REMA, floating mask); repeat-pass differencing.
* **Sentinel-2** L2A from the public Element 84 Earth Search STAC: find scenes
  over a point, mosaic a band onto a grid, map a lake with NDWI_ice.
* **Flexure formulas** for fitting profiles: the tidal hinge zone and the edge of
  a broad load on an elastic beam on a fluid foundation.

All coordinates are EPSG:3031 metres unless a name says otherwise.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CACHE = HERE / "outputs" / "icesat2"
STAC_URL = "https://earth-search.aws.element84.com/v1/search"

RHO_W, RHO_SW, G = 1000.0, 1028.0, 9.81

# GDAL settings for fast reads of cloud-optimised GeoTIFFs over HTTP
for _k, _v in {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
               "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES", "GDAL_HTTP_MULTIPLEX": "YES",
               "VSI_CACHE": "TRUE", "GDAL_HTTP_MAX_RETRY": "3", "GDAL_HTTP_RETRY_DELAY": "1"}.items():
    os.environ.setdefault(_k, _v)


# ===========================================================================
# Geometry
# ===========================================================================
def to_lonlat(x, y):
    import pyproj
    return pyproj.Transformer.from_crs(3031, 4326, always_xy=True).transform(x, y)


def region(bounds):
    """SlideRule polygon (lon/lat dicts) for an EPSG:3031 box (x0, y0, x1, y1)."""
    import geopandas as gpd
    import sliderule
    from shapely.geometry import box
    gs = gpd.GeoSeries([box(*bounds)], crs=3031).to_crs(4326)
    return sliderule.toregion(gpd.GeoDataFrame(geometry=gs))["poly"]


def _tag(bounds):
    return "_".join(f"{v / 1e3:.0f}" for v in bounds)


def _flatten(gdf):
    """GeoDataFrame from SlideRule -> plain DataFrame with x, y in EPSG:3031."""
    if len(gdf) == 0:
        return pd.DataFrame()
    gdf = gdf.to_crs(3031)
    df = pd.DataFrame(gdf.drop(columns="geometry"))
    df["x"], df["y"] = gdf.geometry.x.values, gdf.geometry.y.values
    return df.reset_index()


# ===========================================================================
# ICESat-2 ATL06
# ===========================================================================
ATL06_FIELDS = ["geophysical/dac", "fit_statistics/dh_fit_dy", "fit_statistics/snr_significance"]
RENAME = {"gt": "beam", "geophysical/dac": "dac", "fit_statistics/dh_fit_dy": "dh_fit_dy",
          "fit_statistics/snr_significance": "snr_significance"}


def fetch_atl06(bounds, t0="2018-10-01", t1="2026-11-01", freq="4MS", cache=CACHE, verbose=True):
    """ATL06 segments inside ``bounds`` from SlideRule's ``atl06sp``.

    The request is split into ``freq`` chunks because SlideRule caps the granule
    search at 300 granules.  Each chunk is cached as parquet in ``cache`` and only
    fetched once.  Returns one DataFrame (columns renamed by ``RENAME``)."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    edges = pd.date_range(t0, t1, freq=freq)
    if edges[-1] < pd.Timestamp(t1):
        edges = edges.append(pd.DatetimeIndex([pd.Timestamp(t1)]))
    poly, ready = None, False
    parts = []
    for a, b in zip(edges[:-1], edges[1:]):
        f = cache / f"atl06_{_tag(bounds)}_{a:%Y%m%d}_{b:%Y%m%d}.parquet"
        if not f.exists():
            if not ready:
                import sliderule
                from sliderule import icesat2  # noqa: F401
                sliderule.init(verbose=False)
                poly, ready = region(bounds), True
            from sliderule import icesat2
            gdf = icesat2.atl06sp({"poly": poly, "t0": f"{a:%Y-%m-%dT%H:%M:%SZ}",
                                   "t1": f"{b:%Y-%m-%dT%H:%M:%SZ}", "atl06_fields": ATL06_FIELDS})
            _flatten(gdf).to_parquet(f)
            if verbose:
                print(f"fetched {a:%Y-%m-%d} to {b:%Y-%m-%d}: {len(gdf):,} segments")
        parts.append(pd.read_parquet(f))
    df = pd.concat([p for p in parts if len(p)], ignore_index=True).rename(columns=RENAME)
    return df.sort_values("time").reset_index(drop=True)


def add_reference(df, dem, floating):
    """Add reference fields to ATL06 segments.

    * ``rema``: the DEM interpolated to the segment.
    * ``dgl``: distance (m) from the segment to the nearest non-floating cell of
      the ``floating`` mask (0 on grounded ice).
    * ``h``: ``h_li`` minus the ocean tide and the dynamic atmosphere correction
      (ATL06 heights carry neither), i.e. the height of floating ice above its
      mean (tide-free) state.  On grounded ice this wrongly removes the tide.
    * ``res``: ``h - rema``, which takes out the topography so that passes a few
      hundred metres apart can be compared.
    """
    X, Y = xr.DataArray(df.x.values), xr.DataArray(df.y.values)
    df = df.copy()
    df["rema"] = dem.interp(x=X, y=Y).values
    dx = abs(float(floating.x[1] - floating.x[0]))
    dist = xr.DataArray(ndi.distance_transform_edt(floating.values) * dx, coords=floating.coords)
    df["dgl"] = dist.sel(x=X, y=Y, method="nearest").values
    df["h"] = df.h_li - df.tide_ocean - df.dac
    df["res"] = df.h - df.rema
    return df


def passes_near(df, x0, y0, radius):
    """One row per (rgt, cycle, beam) with segments within ``radius`` of (x0, y0)."""
    r = np.hypot(df.x - x0, df.y - y0)
    near = df[r < radius].assign(r=r[r < radius])
    return (near.groupby(["rgt", "cycle", "beam"])
                .agg(time=("time", "first"), closest_m=("r", "min"), n=("r", "size"),
                     tide_ocean=("tide_ocean", "median"))
                .reset_index().sort_values("time"))


def pass_difference(df, rgt, cyc_ref, cyc, max_sep=600.0, far=None, beams=None, field="res"):
    """Elevation change between two passes of one reference ground track.

    Each segment of pass ``cyc`` is paired with the nearest segment of pass
    ``cyc_ref`` (any beam) no more than ``max_sep`` away, and ``dh`` is the
    difference of their REMA residuals (``field``, default ``res``), so the
    topography between the two ground tracks drops out.  ``far(frame) -> bool`` marks far-field
    segments; their median ``dh`` (per beam) is subtracted, which removes errors
    in the tide and atmosphere corrections and any shelf-wide change in the
    surface (snowfall, firn compaction).  Returns a DataFrame, one row per paired
    segment, sorted by beam and y."""
    A = df[(df.rgt == rgt) & (df.cycle == cyc_ref)]
    B = df[(df.rgt == rgt) & (df.cycle == cyc)]
    if beams is not None:
        B = B[B.beam.isin(beams)]
    if len(A) == 0 or len(B) == 0:
        return pd.DataFrame()
    sep, j = cKDTree(np.c_[A.x, A.y]).query(np.c_[B.x, B.y])
    keep = sep < max_sep
    a, b = A.iloc[j[keep]], B[keep]
    out = b[["time", "rgt", "cycle", "beam", "x", "y", "dgl", "h"]].copy()
    out["time_ref"], out["cycle_ref"] = a.time.values, cyc_ref
    out["sep"] = sep[keep]
    out["tide"] = b.tide_ocean.values - a.tide_ocean.values
    out["dh_raw"] = b[field].values - a[field].values
    out["dh"] = out.dh_raw
    if far is not None:
        f = far(out) & (a.dgl.values > 0)
        off = out[f].groupby("beam").dh_raw.median()
        out["offset"] = out.beam.map(off)
        out["dh"] = out.dh_raw - out.offset
        out["far"] = f
    return out.sort_values(["beam", "y"]).reset_index(drop=True)


def smooth(s, n=5):
    """Running median over ``n`` segments (ATL06 segments are 20 m apart)."""
    return pd.Series(np.asarray(s)).rolling(n, center=True, min_periods=1).median().values


# ===========================================================================
# ICESat-2 ATL03 photons
# ===========================================================================
def fetch_atl03(bounds, day, rgt, cache=CACHE):
    """All ATL03 photons (signal and background) of reference ground track
    ``rgt`` on ``day`` (YYYY-MM-DD) inside ``bounds``, cached as parquet."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"atl03_{_tag(bounds)}_{rgt}_{day}.parquet"
    if not f.exists():
        import sliderule
        from sliderule import icesat2
        sliderule.init(verbose=False)
        t0 = pd.Timestamp(day)
        p = icesat2.atl03sp({"poly": region(bounds), "rgt": rgt,
                             "t0": f"{t0:%Y-%m-%dT00:00:00Z}",
                             "t1": f"{t0 + pd.Timedelta(days=1):%Y-%m-%dT00:00:00Z}",
                             "cnf": 0, "srt": icesat2.SRT_LAND_ICE, "len": 40, "res": 40,
                             "cnt": 1, "ats": 0.0, "pass_invalid": True})
        cols = ["time", "rgt", "cycle", "spot", "pair", "track", "height", "atl03_cnf", "x", "y"]
        _flatten(p)[cols].to_parquet(f)
    return pd.read_parquet(f)


N_WATER = 1.336            # refractive index of meltwater at 532 nm
REFRACTION = 1 / N_WATER   # true depth / apparent depth for near-nadir ICESat-2 photons (~0.75)


def track_frame(x, y):
    """Origin and unit vector of the along-track direction (principal axis)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    o = np.array([x.mean(), y.mean()])
    u = np.linalg.svd(np.c_[x - o[0], y - o[1]], full_matrices=False)[2][0]
    if u[1] < 0:            # point northwards in y
        u = -u
    return o, u


def along(x, y, frame):
    o, u = frame
    return (np.asarray(x) - o[0]) * u[0] + (np.asarray(y) - o[1]) * u[1]


def surface_and_bed(ph, frame, bin_m=40.0, surf_cnf=3, max_depth=15.0, min_apparent=1.0):
    """Surface and (under water) bed heights from the photons of one beam.

    For each ``bin_m`` along-track bin: the surface is the 0.1 m height bin with
    most medium/high-confidence photons (refined to the mean height within
    0.15 m of it).  Below it, photons of every confidence between
    ``min_apparent`` and ``max_depth`` m are histogrammed in 0.2 m bins; the
    deepest-clear peak that stands well above the background rate (estimated
    from photons 3-20 m *above* the surface) is taken as the bed.  Apparent bed
    depths are corrected for refraction (x ``REFRACTION``) to give
    ``bed = surface - depth``.  Heights are raw ATL03 (no tide correction).
    Returns a DataFrame with s, x, y, surface, n_surf, apparent_depth, depth, bed."""
    s = along(ph.x, ph.y, frame)
    h = ph.height.values
    cnf = ph.atl03_cnf.values
    edges = np.arange(np.floor(s.min() / bin_m) * bin_m, s.max() + bin_m, bin_m)
    idx = np.digitize(s, edges) - 1
    rows = []
    for k in np.unique(idx):
        m = idx == k
        hk, ck = h[m], cnf[m]
        sig = hk[ck >= surf_cnf]
        if sig.size < 5:
            continue
        hb = np.arange(sig.min(), sig.max() + 0.1, 0.1)
        if hb.size < 2:
            hs = float(np.mean(sig))
        else:
            cnt, e = np.histogram(sig, hb)
            pk = 0.5 * (e[np.argmax(cnt)] + e[np.argmax(cnt) + 1])
            hs = float(sig[np.abs(sig - pk) < 0.15].mean())
        row = dict(s=0.5 * (edges[k] + edges[k + 1]), x=ph.x.values[m].mean(), y=ph.y.values[m].mean(),
                   surface=hs, n_surf=int((np.abs(sig - hs) < 0.15).sum()),
                   apparent_depth=np.nan, depth=np.nan, bed=np.nan)
        above = hk[(hk > hs + 3) & (hk < hs + 20)]
        bg = above.size / 17.0 * 0.2                         # background photons per 0.2 m bin
        below = hk[(hk < hs - min_apparent) & (hk > hs - max_depth)]
        if below.size >= 5:
            cnt, e = np.histogram(hs - below, np.arange(min_apparent, max_depth + 0.2, 0.2))
            c = 0.5 * (e[1:] + e[:-1])
            j = int(np.argmax(cnt))
            up = (c > c[j] - 2.0) & (c < c[j] - 0.6)
            down = (c > c[j] + 0.6) & (c < c[j] + 2.0)
            thr = max(5.0, bg + 4 * np.sqrt(bg + 1))
            # a bed is a distinct peak, well above the background and above the
            # counts both shallower and deeper than it; volume scattering in turbid
            # water instead decays smoothly from the surface and has no such peak
            if (cnt[j] >= thr and up.sum() >= 3 and down.sum() >= 3
                    and cnt[j] >= 2.5 * max(np.median(cnt[up]), 1) and cnt[j] >= 2.5 * max(np.median(cnt[down]), 1)):
                row["apparent_depth"] = c[j]
                row["depth"] = c[j] * REFRACTION
                row["bed"] = hs - row["depth"]
        rows.append(row)
    return pd.DataFrame(rows)


# ===========================================================================
# Sentinel-2 L2A (Element 84 Earth Search, public COGs)
# ===========================================================================
def s2_items(bounds, t0, t1, limit=300):
    """Sentinel-2 L2A STAC items that overlap the EPSG:3031 box ``bounds``
    (x0, y0, x1, y1), between the dates ``t0`` and ``t1`` (inclusive).  Returns a
    DataFrame with the items in column ``item``, sorted so that, for each tile
    and date, the first processing comes first."""
    import requests
    x0, y0, x1, y1 = bounds
    xs, ys = [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0]
    lon, lat = to_lonlat(xs, ys)
    body = {"collections": ["sentinel-2-l2a"],
            "intersects": {"type": "Polygon", "coordinates": [list(map(list, zip(lon, lat)))]},
            "datetime": f"{t0}T00:00:00Z/{t1}T23:59:59Z", "limit": limit}
    feats = requests.post(STAC_URL, json=body, timeout=60).json()["features"]
    rows = [dict(id=f["id"], tile=f["id"].split("_")[1], date=f["properties"]["datetime"][:10],
                 cloud=f["properties"].get("eo:cloud_cover", np.nan), item=f) for f in feats]
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    df["version"] = df.id.str.split("_").str[3].astype(int)
    return df.sort_values(["date", "version", "tile"]).reset_index(drop=True)


def make_grid(x0, x1, y0, y1, res):
    """Empty (y, x) DataArray in EPSG:3031 with cell centres on ``res``."""
    import rioxarray  # noqa: F401
    xs = np.arange(x0, x1, res) + res / 2
    ys = np.arange(y1, y0, -res) - res / 2
    return xr.DataArray(np.zeros((ys.size, xs.size), np.float32), coords={"y": ys, "x": xs},
                        dims=("y", "x")).rio.write_crs(3031)


NATIVE_RES = {"blue": 10, "green": 10, "red": 10, "nir": 10, "swir16": 20, "swir22": 20}


def _overview(band, res):
    """COG overview level whose pixels are no coarser than ``res`` (None = full)."""
    lvl = int(np.floor(np.log2(res / NATIVE_RES.get(band, 10)))) - 1
    return lvl if lvl >= 0 else None


def s2_band(items, band, grid):
    """Surface reflectance (0-1) of ``band`` from one or more items, averaged onto
    ``grid``; where items overlap the first one wins.  Reads the coarsest COG
    overview that is still at least as fine as the grid."""
    import rioxarray
    from rasterio.enums import Resampling
    x0, x1 = float(grid.x.min()), float(grid.x.max())
    y0, y1 = float(grid.y.min()), float(grid.y.max())
    ov = _overview(band, abs(float(grid.x[1] - grid.x[0])))
    out = None
    for it in items:
        da = rioxarray.open_rasterio(it["assets"][band]["href"], overview_level=ov).squeeze(drop=True)
        try:
            da = da.rio.clip_box(x0 - 500, y0 - 500, x1 + 500, y1 + 500, crs="EPSG:3031")
        except Exception:  # no overlap
            continue
        da = da.where(da > 0).rio.write_nodata(np.nan)
        da = da.rio.reproject_match(grid, resampling=Resampling.average).astype(np.float32)
        p = it["properties"]
        offset = 0.0
        if str(p.get("s2:processing_baseline", "00.00")) >= "04.00" and not p.get("earthsearch:boa_offset_applied", False):
            offset = -1000.0
        da = (da + offset) / 1e4
        out = da if out is None else out.fillna(da)
        if not np.isnan(out.values).any():      # grid covered: skip remaining tiles
            break
    return out


def s2_scene(date, grid, bands=("red", "green", "blue"), cache=CACHE):
    """Mosaic of every tile acquired over ``grid`` on ``date``; a Dataset of
    bands, cached as NetCDF in ``cache``."""
    bounds = (float(grid.x.min()), float(grid.y.min()), float(grid.x.max()), float(grid.y.max()))
    res = abs(float(grid.x[1] - grid.x[0]))
    f = Path(cache) / f"s2_{_tag(bounds)}_{res:.0f}m_{date}_{'-'.join(bands)}.nc"
    if f.exists():
        ds = xr.load_dataset(f)
        if all(b in ds for b in bands):
            return ds
    found = s2_items(bounds, date, date)
    if len(found) == 0:
        raise ValueError(f"no Sentinel-2 L2A acquisition over {bounds} on {date}")
    its = list(found.item)
    ds = xr.Dataset({b: s2_band(its, b, grid) for b in bands}, attrs={"date": date})
    ds = ds.drop_vars([v for v in ("spatial_ref", "band") if v in ds.coords])
    for b in bands:
        ds[b].attrs = {}
        ds[b].encoding = {}
    Path(cache).mkdir(parents=True, exist_ok=True)
    enc = {b: {"zlib": True, "dtype": "int16", "scale_factor": 1e-4, "_FillValue": -32768} for b in bands}
    tmp = f.with_suffix(f".{os.getpid()}.tmp")
    ds.to_netcdf(tmp, encoding=enc)
    os.replace(tmp, f)                      # never leave a half-written file behind
    return xr.load_dataset(f)


def s2_scan(bounds, seasons, res=40.0, region=None, cache=CACHE, verbose=True, workers=8):
    """Every Sentinel-2 acquisition over ``bounds`` in the given melt seasons.

    ``seasons`` is a list of (t0, t1) date strings.  Each acquisition (all tiles
    of one date, mosaicked) is read at ``res`` with the bands blue, green, red
    and swir16, and cached.  ``region(x, y) -> bool array`` optionally restricts
    the statistics to part of the window.  Returns one row per date with

    * ``valid``: fraction of the window with data;
    * ``cloud``: fraction with SWIR (B11) reflectance > 0.12 -- snow and water
      are dark at 1.6 um, most clouds are bright;
    * ``water_km2``: area with NDWI_ice > 0.25 and SWIR < 0.05, i.e. open water
      that is not cloud.
    """
    from concurrent.futures import ProcessPoolExecutor
    grid = make_grid(bounds[0], bounds[2], bounds[1], bounds[3], res)
    X, Y = np.meshgrid(grid.x.values, grid.y.values)
    sel = np.ones_like(X, bool) if region is None else region(X, Y)
    dates = []
    for t0, t1 in seasons:
        items = s2_items(bounds, t0, t1)
        dates += sorted(items.date.unique()) if len(items) else []
    bands = ("blue", "green", "red", "swir16")
    with ProcessPoolExecutor(max_workers=workers) as pool:   # reading COGs is I/O bound
        futs = {d: pool.submit(_scene_or_none, d, grid, bands, cache) for d in dates}
    rows = []
    for date, fut in futs.items():
        s = fut.result()
        if s is None:
            if verbose:
                print(f"  {date}: skipped (unreadable)")
            continue
        ok = np.isfinite(s.blue.values) & np.isfinite(s.swir16.values) & sel
        cloud = (s.swir16.values > 0.12) & ok
        water = (ndwi_ice(s).values > 0.25) & (s.swir16.values < 0.05) & ok
        rows.append(dict(date=pd.Timestamp(date), valid=ok.sum() / sel.sum(),
                         cloud=cloud.sum() / max(ok.sum(), 1), water_km2=water.sum() * res ** 2 / 1e6))
    if verbose:
        print(f"{len(rows)} acquisitions read")
    return pd.DataFrame(rows), grid


def _scene_or_none(date, grid, bands, cache):
    try:
        return s2_scene(date, grid, bands=bands, cache=cache)
    except Exception:
        return None


PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
PC = {  # Planetary Computer collections: band names, and reflectance = DN * scale + offset
    "landsat-c2-l2": dict(bands={"blue": "blue", "green": "green", "red": "red", "swir16": "swir16"},
                          scale=2.75e-5, offset=-0.2,
                          query={"platform": {"in": ["landsat-8", "landsat-9"]}}),
    "sentinel-2-l2a": dict(bands={"blue": "B02", "green": "B03", "red": "B04", "swir16": "B11"},
                           scale=1e-4, offset=0.0, query={}),
}


def pc_items(collection, bounds, t0, t1, max_cloud=60, limit=300):
    """Items of a Planetary Computer ``collection`` (``landsat-c2-l2``: Landsat
    8/9 Collection 2 Level 2; ``sentinel-2-l2a``: Sentinel-2 L2A, which there
    reaches back to 2016) overlapping the EPSG:3031 box ``bounds``."""
    import requests
    x0, y0, x1, y1 = bounds
    lon, lat = to_lonlat([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0])
    body = {"collections": [collection],
            "intersects": {"type": "Polygon", "coordinates": [list(map(list, zip(lon, lat)))]},
            "datetime": f"{t0}T00:00:00Z/{t1}T23:59:59Z", "limit": limit,
            "query": {"eo:cloud_cover": {"lt": max_cloud}, **PC[collection]["query"]}}
    feats = requests.post(PC_STAC, json=body, timeout=60).json().get("features", [])
    df = pd.DataFrame([dict(id=f["id"], date=f["properties"]["datetime"][:10],
                            cloud=f["properties"]["eo:cloud_cover"], item=f) for f in feats])
    return df.sort_values(["date", "id"]).reset_index(drop=True) if len(df) else df


def landsat_items(bounds, t0, t1, max_cloud=60, limit=300):
    return pc_items("landsat-c2-l2", bounds, t0, t1, max_cloud, limit)


def pc_scene(collection, date, grid, bands=("blue", "green", "red", "swir16"), cache=CACHE):
    """Surface-reflectance mosaic of everything ``collection`` acquired over
    ``grid`` on ``date`` (cached like ``s2_scene``).  Assets in archive storage
    (HTTP 409 on some older scenes) are skipped."""
    import requests
    import rioxarray
    from rasterio.enums import Resampling
    cfg = PC[collection]
    bounds = (float(grid.x.min()), float(grid.y.min()), float(grid.x.max()), float(grid.y.max()))
    res = abs(float(grid.x[1] - grid.x[0]))
    f = Path(cache) / f"pc-{collection}_{_tag(bounds)}_{res:.0f}m_{date}_{'-'.join(bands)}.nc"
    if f.exists():
        return xr.load_dataset(f)
    token = requests.get(f"https://planetarycomputer.microsoft.com/api/sas/v1/token/{collection}",
                         timeout=30).json().get("token", "")
    items = list(pc_items(collection, bounds, date, date, max_cloud=101).item)
    out = {}
    for b in bands:
        acc = None
        for it_ in items:
            key = cfg["bands"][b]
            if key not in it_["assets"]:
                continue
            href = it_["assets"][key]["href"] + ("?" + token if token else "")
            try:
                da = rioxarray.open_rasterio(href, overview_level=_overview(b, res)
                                             if collection == "sentinel-2-l2a" else None).squeeze(drop=True)
                da = da.rio.clip_box(bounds[0] - 500, bounds[1] - 500, bounds[2] + 500, bounds[3] + 500,
                                     crs="EPSG:3031")
            except Exception:
                continue
            da = da.where(da > 0).rio.write_nodata(np.nan)
            da = da.rio.reproject_match(grid, resampling=Resampling.average).astype(np.float32)
            off = cfg["offset"]
            if collection == "sentinel-2-l2a" and str(it_["properties"].get("s2:processing_baseline", "0")) >= "04.00":
                off = -0.1
            da = da * cfg["scale"] + off
            acc = da if acc is None else acc.fillna(da)
            if not np.isnan(acc.values).any():
                break
        if acc is None:
            raise OSError(f"no readable {collection} {b} band for {date}")
        out[b] = acc
    ds = xr.Dataset(out, attrs={"date": date, "source": collection})
    ds = ds.drop_vars([v for v in ("spatial_ref", "band") if v in ds.coords])
    for b in bands:
        ds[b].attrs, ds[b].encoding = {}, {}
    enc = {b: {"zlib": True, "dtype": "int16", "scale_factor": 1e-4, "_FillValue": -32768} for b in bands}
    tmp = f.with_suffix(f".{os.getpid()}.tmp")
    ds.to_netcdf(tmp, encoding=enc)
    os.replace(tmp, f)
    return xr.load_dataset(f)


def landsat_scene(date, grid, bands=("blue", "green", "red", "swir16"), cache=CACHE):
    return pc_scene("landsat-c2-l2", date, grid, bands, cache)


def ndwi_ice(scene):
    """NDWI_ice = (blue - red) / (blue + red), the usual index for water on ice."""
    return (scene.blue - scene.red) / (scene.blue + scene.red)


def lake_mask(ndwi, seed=None, thresh=0.25):
    """The connected area with NDWI_ice > ``thresh`` that contains (or lies
    nearest to) the point ``seed = (x, y)``, or the largest one if ``seed`` is
    None; boolean DataArray."""
    wet = (ndwi > thresh).values
    lab, n = ndi.label(wet)
    if n == 0:
        return ndwi.copy(data=np.zeros_like(wet))
    if seed is None:
        return ndwi.copy(data=lab == np.argmax(ndi.sum(wet, lab, range(1, n + 1))) + 1)
    yy, xx = np.nonzero(wet)
    X, Y = ndwi.x.values[xx], ndwi.y.values[yy]
    k = np.argmin(np.hypot(X - seed[0], Y - seed[1]))
    return ndwi.copy(data=lab == lab[yy[k], xx[k]])


def rgb(scene, gain=1.0):
    """(y, x, 3) image array for imshow, scaled by the scene's 99.5th percentile."""
    a = np.dstack([scene[b].values for b in ("red", "green", "blue")])
    return np.clip(gain * a / np.nanpercentile(a, 99.5), 0, 1)


# ===========================================================================
# Topography
# ===========================================================================
def depression_depth(dem):
    """Depth (m) of closed depressions: the DEM filled from its edges (priority
    flood, by morphological reconstruction) minus the DEM."""
    from skimage.morphology import reconstruction
    z = dem.values.astype(float)
    z = np.where(np.isfinite(z), z, np.nanmax(z))
    seed = z.copy()
    seed[1:-1, 1:-1] = z.max()
    return dem.copy(data=reconstruction(seed, z, method="erosion") - z)


# ===========================================================================
# Flexure formulas (beam on a fluid foundation, flexural length ell = (D/k)^1/4)
# ===========================================================================
def youngs_modulus(ell, H, nu=0.3):
    """E that gives flexural length ``ell`` for thickness ``H``."""
    return ell ** 4 * RHO_SW * G * 12 * (1 - nu ** 2) / H ** 3


def flexural_length(E, H, nu=0.3):
    return (E * H ** 3 / (12 * (1 - nu ** 2)) / (RHO_SW * G)) ** 0.25


def hinge(s, A, s0, ell, c):
    """Tidal hinge zone: height change of an elastic beam clamped at a hinge at
    ``s0`` (grounded for s < s0) whose floating part rises by ``A``;
    ``c`` is a constant offset.  beta = 1 / (sqrt(2) ell)."""
    x = np.clip(s - s0, 0, None)
    b = 1.0 / (np.sqrt(2.0) * ell)
    return c + A * (1 - np.exp(-b * x) * (np.cos(b * x) + np.sin(b * x)))


def load_edge(s, A, s0, ell, c, sign=1):
    """Edge of a broad uniform load on an elastic beam (Hetenyi 1946).

    The load lies on the side ``sign * (s - s0) > 0`` and sinks the beam far
    inside it by ``A``.  Returns the surface change ``c - w``: inside,
    w = A (1 - D/2), outside w = A D/2, with D = exp(-b a) cos(b a), a the
    distance from the edge and b = 1 / (sqrt(2) ell)."""
    a = sign * (np.asarray(s) - s0)
    b = 1.0 / (np.sqrt(2.0) * ell)
    D = np.exp(-b * np.abs(a)) * np.cos(b * np.abs(a))
    w = np.where(a > 0, A * (1 - D / 2), A * D / 2)
    return c - w


def beam_deflection(s, q, ell, isostatic=False):
    """Deflection (m, + down) of an infinite elastic beam on a fluid foundation
    under the load ``q`` (Pa) sampled at regular positions ``s`` (m):
    (D d4/ds4 + k) w = q, with D = k ell^4 and k = rho_sw g, solved by FFT on
    a grid padded to 4x its length.  ``isostatic`` gives w = q / k."""
    q = np.nan_to_num(np.asarray(q, float))
    k = RHO_SW * G
    if isostatic:
        return q / k
    n = q.size
    pad = 2 * n
    qp = np.pad(q, pad)
    ds = float(s[1] - s[0])
    kk = 2 * np.pi * np.fft.rfftfreq(qp.size, ds)
    w = np.fft.irfft(np.fft.rfft(qp) / (k * ell ** 4 * kk ** 4 + k), qp.size)
    return w[pad:pad + n]


# ===========================================================================
# Water surface against the water-free surface before filling
# ===========================================================================
def wet_crossing(atl, ph, rgt, cyc_wet, cyc_ref, spot, dem, far, seed, max_sep=600.0, tol=0.12,
                 bin_m=40.0):
    """Compare a pass over open water with an earlier, water-free pass.

    ``ph`` are the ATL03 photons of the wet pass and ``spot`` the beam to use.
    The wet surface and the bed beneath the water come from the photons
    (``surface_and_bed``, tide and DAC removed with the ATL06 values of the
    pass).  The water-free surface ``z0`` at each bin is the nearest segment of
    the reference pass ``cyc_ref`` (any beam, within ``max_sep``), moved to the
    bin with REMA and put in the wet pass's frame with the far-field offset
    from ``pass_difference``; so ``surface - z0`` is the far-field-referenced
    change.  The water is the run of bins containing ``seed`` (an along-track
    y or a (y0, y1) range) whose surface lies within ``tol`` of the water level.

    Returns (bins, summary): a DataFrame per along-track bin and a dict."""
    a6 = atl[(atl.rgt == rgt) & (atl.cycle == cyc_wet) & (atl.spot == spot)]
    beam = int(a6.beam.mode().iloc[0])
    corr = float((a6.tide_ocean + a6.dac).median())
    d6 = pass_difference(atl, rgt, cyc_ref, cyc_wet, max_sep=max_sep, far=far, beams=[beam])
    off = float(d6.offset.iloc[0])
    p = ph[ph.spot == spot]
    fr = track_frame(p.x, p.y)
    sb = surface_and_bed(p, fr, bin_m=bin_m)
    sb["surface"] -= corr
    sb["bed"] -= corr
    A = atl[(atl.rgt == rgt) & (atl.cycle == cyc_ref)]
    dist, j = cKDTree(np.c_[A.x, A.y]).query(np.c_[sb.x, sb.y])
    rema = dem.interp(x=xr.DataArray(sb.x.values), y=xr.DataArray(sb.y.values)).values
    sb["z0"] = np.where(dist < max_sep, A.res.values[j] + rema + off, np.nan)
    sb["dh"] = sb.surface - sb.z0
    sb = sb.sort_values("y").reset_index(drop=True)
    # the water: contiguous flat run around the seed
    y0, y1 = (seed, seed) if np.ndim(seed) == 0 else seed
    inseed = sb.y.between(y0 - bin_m, y1 + bin_m)
    H = float(sb.surface[inseed].median())
    flat = ((sb.surface - H).abs() < tol).values
    k0 = int(np.argmin(np.abs(sb.y - 0.5 * (y0 + y1))))
    lo = hi = k0

    def grow(k, step):          # extend while flat, bridging up to 3 odd bins
        while True:
            nxt = [k + step * i for i in (1, 2, 3, 4) if 0 <= k + step * i < len(flat)]
            ok = [i for i in nxt if flat[i]]
            if not ok:
                return k
            k = ok[0]
    lo, hi = grow(k0, -1), grow(k0, +1)
    sb["water"] = False
    sb.loc[lo:hi, "water"] = True
    W = sb[sb.water]
    shore = [sb.z0.iloc[max(lo - 2, 0):lo + 1].median(), sb.z0.iloc[hi:hi + 3].median()]
    summary = dict(rgt=rgt, wet=str(a6.time.iloc[0].date()), ref=str(A.time.iloc[0].date()), spot=spot, beam=beam, corr=corr,
                   water_level=H, length_km=(W.s.max() - W.s.min() + bin_m) / 1e3,
                   z0_floor=float(W.z0.median()), level_minus_floor=float(H - W.z0.median()),
                   level_minus_shore_south=float(H - shore[0]), level_minus_shore_north=float(H - shore[1]),
                   bed_fraction=float(W.bed.notna().mean()), depth_median=float((H - W.bed).median()),
                   depth_max=float((H - W.bed).max()), floor_drop=float((W.z0 - W.bed).median()),
                   sep_m=float(np.median(dist[sb.water.values])))
    return sb, summary


def crossing_predictions(sb, summary, ells=(250.0, 500.0, 873.0), stretch=1.0, ds=20.0, fill="taper"):
    """Surface change that plates predict along a wet crossing.

    The load is the water depth measured along the crossing (water level minus
    ATL03 bed), tapered linearly to zero at the two shorelines where no bed is
    seen near them (gaps in between are interpolated); with ``fill="median"``
    every water bin without a bed takes the median measured depth instead
    (for flat-floored troughs where the bed is seen only here and there).  The
    plate is an infinite elastic beam (plane strain) with flexural length
    ``ell``, plus the isostatic and rigid limits.  ``stretch`` (<= 1) maps
    along-track distance to distance across a long lake that the track crosses
    obliquely (sin of the crossing angle).  The predicted surface change is
    d - w over the water and -w outside it, i.e. it assumes all the water was
    added (imported) and none of the old surface melted.  Returns a DataFrame
    on a regular along-track grid with columns s, depth, rigid, isostatic and
    elastic_<ell>."""
    H = summary["water_level"]
    dep = np.where(sb.water, H - sb.bed, np.nan)
    wi = np.nonzero(sb.water.values)[0]
    if fill == "median" and np.isfinite(dep).any():
        dep = np.where(np.isfinite(dep), dep, np.nanmedian(dep))
    elif np.isfinite(dep).any():
        for k in (wi[0], wi[-1]):          # the shorelines
            if not np.isfinite(dep[k]):
                dep[k] = 0.0
        dep = pd.Series(dep).interpolate(limit_area="inside").values
    dep = np.where(sb.water, np.nan_to_num(dep, nan=0.0), 0.0).clip(0, None)
    s = np.arange(sb.s.min(), sb.s.max(), ds)
    d = np.interp(s, sb.s, dep)
    q = RHO_W * G * d
    out = pd.DataFrame({"s": s, "depth": d, "rigid": d, "isostatic": d - beam_deflection(s, q, 0, isostatic=True)})
    for ell in ells:
        out[f"elastic_{ell:.0f}"] = d - beam_deflection(s * stretch, q, ell)
    return out


def plot_crossing(axs, ph, sb, summary, pred=None, spot=None, corr=0.0, hlim=None, slim=None):
    """Two-panel figure of a wet crossing: (top) photons, the water-free
    surface before filling, the water surface and the refraction-corrected bed;
    (bottom) the observed change against plate predictions."""
    import matplotlib.pyplot as plt  # noqa: F401
    a0, a1 = axs
    H = summary["water_level"]
    p = ph[ph.spot == (spot if spot is not None else summary["spot"])]
    fr = track_frame(p.x, p.y)
    s_ph = along(p.x, p.y, fr) / 1e3
    lim = hlim or (H - 10, H + 6)
    corr = summary.get("corr", corr)
    a0.hist2d(s_ph, p.height - corr, bins=[400, 160], range=[[sb.s.min() / 1e3, sb.s.max() / 1e3], list(lim)],
              cmap="Greys", cmin=1, vmax=60, rasterized=True)
    a0.plot(sb.s / 1e3, sb.z0, color="#eb6834", lw=1.6, label="surface before filling")
    w = sb[sb.water]
    a0.plot(w.s / 1e3, w.surface, color="#2a78d6", lw=0, marker=".", ms=3, label="water surface")
    a0.plot(w.s / 1e3, w.bed, color="#1baf7a", lw=0, marker=".", ms=3, label="bed (refraction-corrected)")
    a0.set(ylim=lim, ylabel="height (m)")
    a0.legend(loc="lower right", fontsize=8)
    a0.grid(False)
    a1.plot(sb.s / 1e3, sb.dh, ".", ms=3, color="k", label="observed: surface now − surface before")
    if pred is not None:
        styles = {"rigid": (":", "#8a8985"), "isostatic": ("--", "#1baf7a")}
        for col in pred.columns:
            if col in ("s", "depth"):
                continue
            ls, c = styles.get(col, ("-", None))
            a1.plot(pred.s / 1e3, pred[col], ls=ls, color=c, lw=1.3,
                    label=col.replace("elastic_", "elastic, ell = ") + (" m" if col.startswith("elastic") else ""))
    a1.axhline(0, color="k", lw=0.6)
    a1.set(ylabel="change (m)", xlabel="distance along track (km)")
    for a in axs:
        a.axvspan(w.s.min() / 1e3, w.s.max() / 1e3, color="#3fa9f5", alpha=0.08, lw=0)
        if slim:
            a.set_xlim(*slim)
    a1.legend(fontsize=7.5, ncol=2, loc="upper right")
