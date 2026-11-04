"""
fsm.py: fill-spill-merge routing of surface meltwater, self-contained.

Meltwater flows downhill, collects in depressions, fills them, spills over
their lowest saddle into neighbouring depressions, and merges into larger
lakes when neighbours fill to the same level.  Water that reaches the edge of
the ocean (or a sink, or optionally the grid edge) is lost.  This is the steady state computed by the
Fill-Spill-Merge algorithm of Barnes, Callaghan & Wickert (2020/2021), which
the notebooks in ``python/`` call through a compiled wrapper.  This module
re-implements it in NumPy/Numba so that it can be called thousands of times
(for example once per time step of the flexure model) and so that it runs
anywhere.

How it works
------------
1. **Flow directions.**  Each cell drains to its steepest-descent neighbour
   (D8).  Exactly flat areas drain towards the edge of the flat.
2. **Depression hierarchy.**  Cells are added in order of increasing
   elevation and joined with union-find.  A cell with no lower neighbour
   starts a new depression.  A cell that touches two depressions is the
   saddle where they *merge* into a parent depression.  This builds a
   merge tree whose leaves are pits and whose roots merge with the "ocean"
   (cells below ``ocean_level``, NaNs, chosen sinks and, optionally, the grid
   edge).  Each depression
   knows its spill elevation and its capacity (volume below the spill).
3. **Fill, spill, merge.**  Water from each cell is moved to the pit its
   flow path ends in.  Going up the tree, a full depression spills its
   excess over the saddle into its sibling (entering at the point where the
   saddle drains to).  When all siblings are full, the water fills the
   merged lake above them.  Excess at the top of the tree goes to the ocean.
4. **Water levels.**  Each lake's level is the flat surface that holds its
   volume over the cells of its depression.

``route(topography, water)`` returns an ``xarray.Dataset`` with the water
depth, water-surface elevation, lake labels and the basin (catchment) that
every cell drains to.  ``Hierarchy`` can be built once and reused for many
water inputs on the same topography.

If the compiled ``fillspillmerge`` wrapper used elsewhere in this repository
is importable, ``route(..., backend="wrapper")`` uses it for the water depths.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import xarray as xr

try:
    import numba
    njit = numba.njit(cache=True)
except ImportError:  # pragma: no cover - slow but works
    def njit(f):
        return f

NO_NODE = -1
OCEAN = 0   # node 0 is the ocean


# ---------------------------------------------------------------------------
# compiled kernels
# ---------------------------------------------------------------------------
@njit
def _d8_receivers(z, sink, ny, nx, dx, dy):
    """Steepest-descent receiver of every cell (-2 = sink/ocean, -1 = pit).
    Exactly flat areas are drained towards cells that have a lower neighbour."""
    n = ny * nx
    rec = np.full(n, -1, np.int64)
    dd = np.sqrt(dx * dx + dy * dy)
    for i in range(ny):
        for j in range(nx):
            c = i * nx + j
            if sink[c]:
                rec[c] = -2
                continue
            best, bs = -1, 0.0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ii, jj = i + di, j + dj
                    if ii < 0 or ii >= ny or jj < 0 or jj >= nx:
                        continue
                    nb = ii * nx + jj
                    dist = dd if (di != 0 and dj != 0) else (dy if di != 0 else dx)
                    s = (z[c] - z[nb]) / dist
                    if s > bs:
                        bs, best = s, nb
            rec[c] = best
    # flats: breadth-first from resolved cells into equal-elevation pits
    queue = np.empty(n, np.int64)
    head, tail = 0, 0
    for c in range(n):
        if rec[c] != -1:
            queue[tail] = c
            tail += 1
    while head < tail:
        u = queue[head]
        head += 1
        i, j = u // nx, u % nx
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ii, jj = i + di, j + dj
                if ii < 0 or ii >= ny or jj < 0 or jj >= nx:
                    continue
                v = ii * nx + jj
                if rec[v] == -1 and z[v] == z[u] and not sink[u]:
                    rec[v] = u
                    queue[tail] = v
                    tail += 1
    return rec


@njit
def _find(uf, a):
    r = a
    while uf[r] != r:
        r = uf[r]
    while uf[a] != r:          # path compression
        nxt = uf[a]
        uf[a] = r
        a = nxt
    return r


@njit
def _merge_tree(z, sink, order, ny, nx):
    """Union-find over cells in increasing elevation.  Returns per-node arrays
    (node 0 = ocean) and, per cell, the node it was added to."""
    n = ny * nx
    OC = n                                   # virtual ocean cell in the union-find
    uf = np.arange(n + 1)
    done = np.zeros(n, np.bool_)
    node_of_root = np.full(n + 1, -1, np.int64)
    node_of_root[OC] = 0
    maxn = n + 1
    parent = np.full(maxn, -1, np.int64)
    spill = np.full(maxn, np.inf)
    saddle = np.full(maxn, -1, np.int64)
    entry = np.full(maxn, -1, np.int64)      # cell through which spilled water enters
    ncell = np.zeros(maxn, np.int64)
    sumz = np.zeros(maxn)
    is_leaf = np.zeros(maxn, np.bool_)
    own = np.zeros(n, np.int64)              # node each cell was added to
    nn = 1
    roots = np.empty(9, np.int64)
    ent = np.empty(9, np.int64)
    for k in range(n):
        c = order[k]
        i, j = c // nx, c % nx
        nr = 0
        touch_ocean = sink[c]
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ii, jj = i + di, j + dj
                if ii < 0 or ii >= ny or jj < 0 or jj >= nx:
                    continue
                nb = ii * nx + jj
                if not done[nb]:
                    continue
                r = _find(uf, nb)
                if r == OC:
                    touch_ocean = True
                    continue
                dup = -1
                for q in range(nr):
                    if roots[q] == r:
                        dup = q
                if dup < 0:
                    roots[nr] = r
                    ent[nr] = nb
                    nr += 1
                elif z[nb] < z[ent[dup]]:
                    ent[dup] = nb
        done[c] = True
        if touch_ocean:
            for q in range(nr):
                nd = node_of_root[roots[q]]
                parent[nd] = 0
                spill[nd] = z[c]
                saddle[nd] = c
                uf[roots[q]] = OC
            uf[c] = OC
            own[c] = 0
        elif nr == 0:                         # a new pit
            nd = nn
            nn += 1
            is_leaf[nd] = True
            node_of_root[c] = nd
            ncell[nd] = 1
            sumz[nd] = z[c]
            own[c] = nd
        elif nr == 1:                         # grows an existing depression
            r = roots[0]
            nd = node_of_root[r]
            uf[c] = r
            ncell[nd] += 1
            sumz[nd] += z[c]
            own[c] = nd
        else:                                 # a saddle: depressions merge
            P = nn
            nn += 1
            tot_n, tot_z = 1, z[c]
            for q in range(nr):
                nd = node_of_root[roots[q]]
                parent[nd] = P
                spill[nd] = z[c]
                saddle[nd] = c
                entry[nd] = ent[q]
                tot_n += ncell[nd]
                tot_z += sumz[nd]
                uf[roots[q]] = c
            uf[c] = c
            node_of_root[c] = P
            ncell[P] = tot_n
            sumz[P] = tot_z
            own[c] = P
    return (parent[:nn], spill[:nn], saddle[:nn], entry[:nn], ncell[:nn], sumz[:nn],
            is_leaf[:nn], own)


@njit
def _drain_targets(rec, own, is_leaf, first_child, n):
    """Node (pit) that each cell's water reaches; 0 = lost to the ocean."""
    tgt = np.full(n, -1, np.int64)
    stack = np.empty(n, np.int64)
    for c0 in range(n):
        if tgt[c0] >= 0:
            continue
        sp = 0
        c = c0
        while True:
            if tgt[c] >= 0:
                t = tgt[c]
                break
            r = rec[c]
            if r == -2:
                t = 0
                break
            if r == -1:                        # pit
                nd = own[c]
                while nd > 0 and not is_leaf[nd]:   # pit on a saddle: go down a branch
                    nd = first_child[nd]
                t = nd
                break
            stack[sp] = c
            sp += 1
            c = r
        tgt[c] = t
        for q in range(sp):
            tgt[stack[q]] = t
    return tgt


# ---------------------------------------------------------------------------
# the hierarchy
# ---------------------------------------------------------------------------
@dataclass
class Hierarchy:
    """Depression hierarchy of a topography (reusable for many water inputs)."""

    z: np.ndarray            # topography, flattened (y-major)
    shape: tuple
    dx: float
    dy: float
    sink: np.ndarray         # cells that drain off the grid (edge, ocean, NaN, user sinks)
    rec: np.ndarray          # D8 receivers
    parent: np.ndarray       # per node
    spill: np.ndarray
    saddle: np.ndarray
    entry: np.ndarray
    ncell: np.ndarray
    sumz: np.ndarray
    is_leaf: np.ndarray
    own: np.ndarray          # node each cell was added to
    target: np.ndarray       # node (pit) each cell drains to (0 = ocean)
    child_ptr: np.ndarray    # CSR children lists
    child_idx: np.ndarray
    own_ptr: np.ndarray      # CSR lists of each node's own cells, in elevation order
    own_idx: np.ndarray

    @property
    def n_nodes(self):
        return self.parent.size

    @property
    def capacity(self):
        """Volume (m^3) each depression holds when full to its spill elevation
        (infinite for the ocean and for the top of a closed grid)."""
        return _capacity(self) * self.dx * self.dy

    def children(self, nd):
        return self.child_idx[self.child_ptr[nd]:self.child_ptr[nd + 1]]

    def own_cells(self, nd):
        return self.own_idx[self.own_ptr[nd]:self.own_ptr[nd + 1]]

    def subtree(self, nd):
        """All nodes in the subtree of nd (including nd)."""
        out, stack = [], [nd]
        while stack:
            a = stack.pop()
            out.append(a)
            stack.extend(self.children(a).tolist())
        return np.array(out, dtype=np.int64)

    def region_cells(self, nd):
        """Cells of depression nd (its own cells and those of all descendants)."""
        return np.concatenate([self.own_cells(a) for a in self.subtree(nd)])


def build_hierarchy(topography, ocean_level=0.0, sinks=None, dx=None, dy=None,
                    edges="closed") -> Hierarchy:
    """Depression hierarchy of ``topography`` (2-D array or DataArray (y, x)).

    ``sinks``: optional boolean mask of extra cells where water leaves the
    surface (moulins, dolines, rifts).  NaNs and cells strictly below
    ``ocean_level`` are always sinks (as in RichDEM).  ``edges="closed"`` (the RichDEM
    fill-spill-merge convention) keeps water inside the grid;
    ``edges="drain"`` lets water that reaches the edge leave it.
    """
    z, dx, dy = _as_grid(topography, dx, dy)
    ny, nx = z.shape
    zf = np.ascontiguousarray(z, dtype=np.float64).ravel()
    sink = ~np.isfinite(zf) | (zf < ocean_level)
    if edges == "drain":
        edge = np.zeros((ny, nx), bool)
        edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
        sink |= edge.ravel()
    elif edges != "closed":
        raise ValueError("edges must be 'closed' or 'drain'")
    if sinks is not None:
        sink |= np.asarray(sinks, bool).ravel()
    zf = np.where(np.isfinite(zf), zf, ocean_level)
    rec = _d8_receivers(zf, sink, ny, nx, float(dx), float(dy))
    order = np.argsort(zf, kind="stable")
    parent, spill, saddle, entry, ncell, sumz, is_leaf, own = _merge_tree(zf, sink, order, ny, nx)
    nn = parent.size
    # children (CSR), ordered by node id
    kids = np.flatnonzero(parent >= 0)
    srt = kids[np.argsort(parent[kids], kind="stable")]
    child_ptr = np.zeros(nn + 1, np.int64)
    np.add.at(child_ptr, parent[srt] + 1, 1)
    child_ptr = np.cumsum(child_ptr)
    first_child = np.full(nn, -1, np.int64)
    has = child_ptr[1:] > child_ptr[:-1]
    first_child[has] = srt[child_ptr[:-1][has]]
    # own cells per node, in elevation order
    o = own[order]
    grp = np.argsort(o, kind="stable")
    own_idx = order[grp]
    own_ptr = np.zeros(nn + 1, np.int64)
    np.add.at(own_ptr, o + 1, 1)
    own_ptr = np.cumsum(own_ptr)
    target = _drain_targets(rec, own, is_leaf, first_child, zf.size)
    return Hierarchy(zf, (ny, nx), float(dx), float(dy), sink, rec, parent, spill, saddle, entry,
                     ncell, sumz, is_leaf, own, target, child_ptr, srt, own_ptr, own_idx)


# ---------------------------------------------------------------------------
# fill, spill, merge
# ---------------------------------------------------------------------------
def _capacity(h: Hierarchy):
    """Capacity in cell-depth units (m x cells).  A depression that never
    spills (the top of a closed grid) holds any amount of water."""
    C = h.ncell * np.where(np.isfinite(h.spill), h.spill, 0.0) - h.sumz
    C = np.maximum(C, 0.0)
    C[0] = np.inf
    C[(h.parent < 0) & (np.arange(C.size) > 0)] = np.inf
    return C


def _distribute(h: Hierarchy, inflow):
    """Water volume (in cell-depth units, m x cells) stored in each node's
    subtree, given the volume arriving at each pit.  Returns (W, lost)."""
    nn = h.n_nodes
    C = _capacity(h)
    finite = C[1:][np.isfinite(C[1:])]
    tol = 1e-12 * max(1.0, float(finite.max()) if finite.size else 1.0)
    W = np.zeros(nn)
    E = np.zeros(nn)                        # pending excess at each node's spill point
    entry_leaf = np.where(h.entry >= 0, h.target[np.maximum(h.entry, 0)], -1)
    full = lambda nd: W[nd] >= C[nd] - tol
    parent, cp, ci = h.parent, h.child_ptr, h.child_idx

    def kids(nd):
        return ci[cp[nd]:cp[nd + 1]]

    def absorb(nd, a, top):
        # every ancestor's stored volume includes this water; ancestors that
        # are not processed yet are recomputed from their children later
        while nd > 0:
            W[nd] += a
            nd = parent[nd]

    def push(nd, v, top):
        """Water v arrives at the bottom of leaf/entry node nd; returns what
        does not fit inside the subtree of ``top``."""
        while True:
            ks = kids(nd)
            if ks.size:
                for s in ks:
                    if v <= tol:
                        return 0.0
                    if not full(s):
                        el = entry_leaf[s] if entry_leaf[s] > 0 else s
                        v = push(el, v, s)
            room = C[nd] - W[nd]
            if room > 0:
                a = min(room, v)
                absorb(nd, a, top)
                v -= a
            if v <= tol:
                return 0.0
            if nd == top:
                return v
            nd = parent[nd]

    old = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old, 10 * nn + 1000))
    try:
        for nd in range(1, nn):              # children are created before parents
            if h.is_leaf[nd]:
                v = inflow[nd]
                a = min(v, C[nd])
                W[nd] = a
                E[nd] = v - a
                continue
            ks = kids(nd)
            W[nd] = W[ks].sum()
            excess = E[ks].sum()
            E[ks] = 0.0
            if excess > tol:
                # spill into the not-yet-full siblings, then the merged lake
                for s in ks:
                    if excess <= tol:
                        break
                    if not full(s):
                        el = entry_leaf[s] if entry_leaf[s] > 0 else s
                        excess = push(el, excess, s)
                        W[nd] = W[ks].sum()
                room = C[nd] - W[nd]
                a = min(max(room, 0.0), excess)
                W[nd] += a
                excess -= a
            E[nd] = excess
    finally:
        sys.setrecursionlimit(old)
    lost = inflow[0] + E[h.parent == 0].sum()
    return W, C, lost


def _levels(h: Hierarchy, W, C):
    """Water level of every lake and the depth/label of every flooded cell."""
    n = h.z.size
    depth = np.zeros(n)
    label = np.full(n, -1, np.int64)
    finite = C[1:][np.isfinite(C[1:])]
    tol = 1e-12 * max(1.0, float(finite.max()) if finite.size else 1.0)
    nn = h.n_nodes
    lakes = []
    for nd in range(1, nn):
        ks = h.children(nd)
        if ks.size and not all(W[k] >= C[k] - tol for k in ks):
            continue                          # water sits in the children instead
        Wc = C[ks].sum() if ks.size else 0.0
        if W[nd] - Wc <= tol:
            continue
        p = h.parent[nd]
        if p > 0 and all(W[k] >= C[k] - tol for k in h.children(p)) and W[p] - C[h.children(p)].sum() > tol:
            continue                          # part of a bigger merged lake
        lakes.append(nd)
    for nd in lakes:
        ks = h.children(nd)
        Nc = int(h.ncell[ks].sum()) if ks.size else 0
        Zc = float(h.sumz[ks].sum()) if ks.size else 0.0
        own = h.own_cells(nd)
        zo = h.z[own]
        m, S = Nc, Zc
        lev = None
        for k in range(zo.size):
            m_new, S_new = m + 1, S + zo[k]
            hcand = (W[nd] + S_new) / m_new
            nxt = zo[k + 1] if k + 1 < zo.size else np.inf
            if hcand <= nxt:
                lev = hcand
                break
            m, S = m_new, S_new
        if lev is None:
            lev = (W[nd] + S) / max(m, 1)
        cells = h.region_cells(nd)
        wet = cells[h.z[cells] < lev]
        depth[wet] = lev - h.z[wet]
        label[wet] = nd
    return depth, label


# ---------------------------------------------------------------------------
# public interface
# ---------------------------------------------------------------------------
def _as_grid(topography, dx=None, dy=None):
    if isinstance(topography, xr.DataArray):
        da = topography.squeeze()
        z = da.values
        xs, ys = da[da.dims[-1]].values, da[da.dims[-2]].values
        dx = dx or float(abs(xs[1] - xs[0]))
        dy = dy or float(abs(ys[1] - ys[0]))
        return z, dx, dy
    z = np.asarray(topography, dtype=float)
    return z, float(dx or 1.0), float(dy or dx or 1.0)


def _coords(topography, shape, dx, dy):
    if isinstance(topography, xr.DataArray):
        da = topography.squeeze()
        return {da.dims[-2]: da[da.dims[-2]].values, da.dims[-1]: da[da.dims[-1]].values}, \
            (da.dims[-2], da.dims[-1])
    ny, nx = shape
    return {"y": np.arange(ny) * dy, "x": np.arange(nx) * dx}, ("y", "x")


def route(topography, water, ocean_level=0.0, hierarchy: Hierarchy | None = None,
          sinks=None, dx=None, dy=None, edges="closed", backend="python") -> xr.Dataset:
    """Fill-spill-merge: where does surface water end up?

    Parameters
    ----------
    topography : 2-D array or ``xarray.DataArray`` (y, x) [m]
    water : water input as a depth on each cell [m] (melt, or the current
        lakes to be re-routed), same shape; a scalar means uniform
    ocean_level : cells strictly below it drain to the ocean
    hierarchy : a :class:`Hierarchy` from :func:`build_hierarchy` to reuse
    sinks : boolean mask of extra drainage points (moulins, dolines)
    edges : ``"closed"`` (default, as RichDEM's fill-spill-merge) or ``"drain"``
    backend : ``"python"`` (this module) or ``"wrapper"`` (the compiled
        ``fillspillmerge`` package, if installed; lake labels still come
        from this module)

    Returns
    -------
    xarray.Dataset with ``water_depth``, ``water_surface`` (topography plus
    depth), ``lake`` (label of the lake each wet cell belongs to, -1 if dry),
    ``basin`` (the pit/depression each cell drains to, 0 = off the grid) and
    attributes with the input, stored and lost volumes [m^3].
    """
    z, dx, dy = _as_grid(topography, dx, dy)
    h = hierarchy if hierarchy is not None else build_hierarchy(z, ocean_level, sinks, dx, dy, edges)
    win = np.broadcast_to(np.asarray(water, dtype=float), z.shape).ravel()
    win = np.where(np.isfinite(win), np.maximum(win, 0.0), 0.0)
    inflow = np.bincount(h.target, weights=win, minlength=h.n_nodes).astype(float)
    W, C, lost = _distribute(h, inflow)
    depth, label = _levels(h, W, C)
    if backend == "wrapper":
        depth = _wrapper_depth(topography, water, ocean_level).ravel()
    coords, dims = _coords(topography, z.shape, dx, dy)
    area = dx * dy
    ds = xr.Dataset(
        {
            "water_depth": (dims, depth.reshape(z.shape),
                            {"units": "m", "long_name": "water depth"}),
            "water_surface": (dims, (h.z + depth).reshape(z.shape),
                              {"units": "m", "long_name": "water-surface (or ice-surface) elevation"}),
            "lake": (dims, label.reshape(z.shape),
                     {"long_name": "lake label (depression id), -1 = dry"}),
            "basin": (dims, h.target.reshape(z.shape),
                      {"long_name": "depression (pit) each cell drains to, 0 = off the grid"}),
            "topography": (dims, h.z.reshape(z.shape), {"units": "m"}),
        },
        coords=coords,
        attrs={"volume_in_m3": float(win.sum() * area),
               "volume_stored_m3": float(depth.sum() * area),
               "volume_lost_m3": float(lost * area),
               "ocean_level": ocean_level, "method": "fill-spill-merge (flexure/fsm.py)"},
    )
    return ds


def _wrapper_depth(topography, water, ocean_level):
    from fillspillmerge import fill_spill_merge   # the compiled wrapper, if installed
    _, hier = fill_spill_merge(topography=topography, ocean_level=ocean_level, return_hierarchy=True)
    return np.asarray(fill_spill_merge(wtd=water, hierarchy=hier))


def lake_at(ds: xr.Dataset, x, y):
    """Label of the lake at (or nearest to) a point."""
    lab = ds.lake.sel({ds.lake.dims[-1]: x, ds.lake.dims[-2]: y}, method="nearest")
    return int(lab)


def lake_table(ds: xr.Dataset) -> "pd.DataFrame":
    """One row per lake: area, volume, max depth and centroid, largest first."""
    import pandas as pd
    lab = ds.lake.values
    d = ds.water_depth.values
    xd, yd = ds.lake.dims[-1], ds.lake.dims[-2]
    X, Y = np.meshgrid(ds[xd].values, ds[yd].values)
    dx = float(abs(ds[xd].values[1] - ds[xd].values[0]))
    dy = float(abs(ds[yd].values[1] - ds[yd].values[0]))
    wet = lab >= 0
    ids, inv = np.unique(lab[wet], return_inverse=True)
    vol = np.bincount(inv, weights=d[wet]) * dx * dy
    area = np.bincount(inv) * dx * dy
    xc = np.bincount(inv, weights=X[wet] * d[wet]) * dx * dy / vol
    yc = np.bincount(inv, weights=Y[wet] * d[wet]) * dx * dy / vol
    dmax = np.zeros(ids.size)
    np.maximum.at(dmax, inv, d[wet])
    t = pd.DataFrame({"lake": ids, "area_km2": area / 1e6, "volume_m3": vol, "max_depth_m": dmax,
                      xd: xc, yd: yc})
    return t.sort_values("volume_m3", ascending=False).reset_index(drop=True)


def catchment_of(h: Hierarchy, region_mask) -> np.ndarray:
    """Cells whose water drains (along D8 paths) into the given region of
    cells (e.g. a lake basin).  Returns a boolean mask (flattened shape)."""
    region = np.asarray(region_mask, bool).ravel()
    n = region.size
    hit = np.zeros(n, np.int8)               # 0 unknown, 1 yes, 2 no
    hit[region] = 1
    rec = h.rec
    for c0 in range(n):
        if hit[c0]:
            continue
        path = []
        c = c0
        while hit[c] == 0 and rec[c] >= 0:
            path.append(c)
            c = rec[c]
        val = hit[c] if hit[c] else 2
        for p in path:
            hit[p] = val
        hit[c0] = val if hit[c0] == 0 else hit[c0]
    return (hit == 1).reshape(h.shape)
