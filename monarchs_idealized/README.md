# MONARCHS in an idealized setting

A minimal, self-contained run of [MONARCHS](https://github.com/monarchs-ice/monarchs)
(MOdel of aNtARctic iCe shelf Hydrology and Stability; Buzzard & Elsey) on an
idealized ice-shelf surface. The goal is to test the melt → lateral drainage →
ponding/refreezing process that drives the meltwater mass redistribution this
repo cares about.

## Setup (`model_setup.py`)

- **Grid:** 15 × 15 cells at 1 km, with 200 vertical firn layers.
- **Surface:** a plane sloping 5 m downslope (left → right) with a 4 m deep
  Gaussian basin (σ = 2 km) two-thirds of the way down. MONARCHS uses
  `firn_depth` as surface elevation, so this is the initial firn thickness
  (37.6–45 m).
- **Firn:** the default empirical density profile with `rho_sfc = 800`
  (dense, melt-affected firn, about 2.5 m of pore space). With the default
  `rho_sfc = 500` (about 9 m of pore space), moderate melt soaks into the firn
  and no lakes form within a season.
- **Forcing:** spatially uniform and constant. There are 50 warm days
  (SW 500, LW 350 W m⁻², T 272 K) then 50 cold days (SW/LW 100 W m⁻², T 250 K).
  The MONARCHS example cases use SW = LW = 800 W m⁻², which melts about 15 m
  of firn and floods the whole domain.
- **Boundaries:** closed domain (`catchment_outflow = False`), so no water
  leaves.

## Running

```bash
python -m venv .venv
.venv/bin/pip install "git+https://github.com/monarchs-ice/monarchs.git@7b5e728"
.venv/bin/monarchs -i model_setup.py   # ~3.5 min on 4 cores incl. Numba compile
.venv/bin/python plot_results.py       # -> idealized_results.png
```

Install from GitHub `main` (pinned above), not PyPI. The PyPI release
`monarchs-ice==1.0.3` still imports `NumbaMinpack` in Numba mode, and that
package needs a Fortran compiler. `main` uses a pure-Numba solver instead.
If pip complains that it can't determine the version (the build uses
hatch-vcs), set `SETUPTOOLS_SCM_PRETEND_VERSION=1.0.3.post0`.

Outputs go to `output/` (gitignored): daily 2-D fields in
`idealized_output.nc`, the final model state in `idealized_dump.nc`, and the
met forcing.

## Result

![results](idealized_results.png)

- Ponding starts in the basin within about 5 days. By day 10 the basin lake is
  about 6 m deep, while the mean over the domain is under 0.5 m.
- Water keeps draining downslope and fills the closed lower half of the domain
  to a roughly flat water level (about 41 m). Peak lake volume is on day 52.
- Lakes melt the firn beneath them, so the firn under the basin thins by about
  5 m.
- Once the cold period starts, frozen lids grow to about 1 m by day 100.
- **Net surface-height change:** about −3 m upslope (melt that drained away)
  and up to +3 m in the basin (ponded or refrozen water). This is the
  redistribution that, combined with BFRNs, would change grounding-line flux.

## Next steps / caveats

- The forcing is constant and has no diurnal cycle. Realistic forcing (ERA5
  or RACMO) is supported via `met_data`.
- A closed domain forces all melt to stay on the grid. Set
  `catchment_outflow = True` to let water leave at the edges.
- To compare with fill-spill-merge (`../python`), apply the same DEM and melt
  map to both models.
