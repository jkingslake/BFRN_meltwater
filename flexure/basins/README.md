# Idealised lake-basin flexure: drainage-system growth and reset

A small, transparent model to test one hypothesis about supraglacial lakes on
Antarctic ice shelves. The hypothesis: **slow, viscous flexure of lake basins
controls whether a lake overtops its sill and extends the drainage system
downstream, and how the system resets after several cool years.**

In high-melt years, water loads a basin and its floor sinks: a little at once
(elastically), and the rest over a relaxation time τ. The water refreezes into
an ice plug that is 9% thicker than the water was. If τ is longer than about a
year, the basin has not finished sinking by the next season. The plug then
fills it, and new water is pushed further downstream. During cool years the
floor keeps sinking, which restores the basin's capacity, and the drainage
system shrinks back.

Start with the notebook [`lake_basin_experiments.ipynb`](lake_basin_experiments.ipynb).

| file | contents |
|---|---|
| [`physics.py`](physics.py) | relaxation times (thin viscous plate, viscous half-space, Maxwell plate), the **exact Fourier-mode solution for a viscous layer floating on sea water**, flexural length, elastic share α_e, Glen's-law viscosity |
| [`basin.py`](basin.py) | one basin: analytic results for one season; one season with `solve_ivp` and an event at the sill; an exact piecewise stepper for many years with refreezing |
| [`cascade.py`](cascade.py) | a chain of basins along a flow line; forcing sequences; the drainage-extent metric |
| [`experiments/`](experiments) | one script per experiment; each writes a NetCDF file and figures to `outputs/` (git-ignored) |
| [`../tests/test_basins.py`](../tests/test_basins.py) | the analytic gap g(t) and overtopping time t*; Π_c for α = 1 and 1.09; mass conservation through spill and refreeze; both τ limits from the exact layer solution; agreement with `viscoelastic_beam.py` |

```bash
cd flexure
python -m basins.experiments.exp1_regime      # also exp2_tau_maps, exp3_width, exp4_multiyear, exp5_cascade
python -m pytest -q tests/test_basins.py
```

Only numpy, scipy, xarray and matplotlib are needed. SI units are used
internally; figures give τ in years and widths in km.

## The model

A basin is a vertical-sided box of area A. Its sill stands D₀ above the
original floor. The state is the water depth h, the ice-plug thickness p and
the viscous part w_v of the floor's deflection:

    load (water-equivalent)   L = h + (ρ_i/ρ_w) p
    floor deflection          w = α_e L + w_v,     dw_v/dt = ((α − α_e) L − w_v) / τ
    free space below sill     C = D₀ + w − p − h

Inflow fills the basin until C = 0. The lake then sits at its sill, and any
inflow the deepening basin cannot take spills downstream. At the end of each
season the water refreezes into the plug. That leaves the load unchanged but
reduces C by 9% of the frozen depth. Between seasons the floor keeps relaxing.

* **One season** (stage 1, α_e = 0, constant inflow Q). The gap
  g = D − h obeys g/D₀ = 1 − αΠ + (α−1)Π t/τ + αΠ e^(−t/τ), with
  Π = τQ/(A D₀). The lake overtops if Π > Π_c, at t* = τ ln[Π/(Π−1)] for
  α = 1. Π_c = 1 for α = 1, and 1/[1 − (α−1) ln(α/(α−1))] ≈ 1.29 for
  α = 1.09. For α < 1 the lake always overtops eventually.
* **Fully relaxed capacity** (the ratchet). C = D₀ − (ρ_w/ρ_i − α) M, where M
  is the water-equivalent depth retained so far. With the thin-plate value
  α = ρ_w/ρ_sw this loses about 0.12 M per cycle; with the half-space value
  α = ρ_w/ρ_i the basin resets completely.
* **Response of a lake of width W** (`physics.basin_response`).
  * `thin_plate`: α = ρ_w/ρ_sw and τ = τ_M + η H³ k⁴ / (3 ρ_sw g).
  * `half_space`: α = ρ_w/ρ_i and τ = τ_M + 2 η k / (ρ_i g).
  * `layer` (default): the dominant mode of the exact viscous layer.

  Here α_e = (ρ_w/ρ_sw) / (1 + (kℓ)⁴), and τ_M = 4η(1−ν²)/E is the Maxwell time.

### What the exact layer solution adds

The exact solution for a viscous layer floating on sea water has **two
modes**, and their amplitudes add up to ρ_w/ρ_i, a fully fluid equilibrium.

* **For kH ≲ 1, the fast mode is bending.** Its τ equals the thin-plate
  formula, and it sinks the floor by ρ_w/ρ_sw ≈ 0.97 per metre of water. The
  slow mode is the ice under the lake thinning by spreading. Its time is
  4η / (ρ_i(1−ρ_i/ρ_sw) g H), which is decades to centuries and independent
  of W, and it adds the remaining 0.12. **Given long enough, that removes the
  thin-plate ratchet**, but the single-mode basin model leaves it out.
* **For kH ≳ 4, one mode carries everything**, with τ equal to the
  half-space formula and α = ρ_w/ρ_i.

## Placeholder assumptions (marked in the code)

* **k = π/W or 2π/W** for a lake of width W. Both are reported as bounds.
* **α_e = (ρ_w/ρ_sw)/(1+(kℓ)⁴).** This is the exact elastic response of a
  single plate mode. With k = 1/W it is the interpolation the brief suggests,
  but it is not valid for kH ≳ 1.
* **Single viscous mode per basin.** The `layer` method keeps the dominant
  mode, drops the slow thickness mode, and adds τ_M as for a viscoelastic
  plate.
* **Basin shape.** A box of area W² (`area=`), with a sill that does not move.
  α is the deflection of the floor relative to the sill; a sill within about ℓ
  of the lake would move with it and reduce α.
* **Refreezing.** All water refreezes each winter (`refreeze=1`). Use
  `refreeze < 1` to leave a buried liquid lake.
* **Forcing.** Inflow is constant through a melt season of 45 days. Cool
  years have 10% of a high year's melt. Stochastic melt is log-normal (a
  stand-in for RACMO statistics).
* **Cascade.** Routing between basins is instantaneous. Widths are log-normal
  and sill heights log-uniform. All melt enters at the head basin.

## Not yet included (candidate extensions)

* Advection of plugs downstream with ice flow (hundreds of m/yr), a
  competing reset mechanism.
* Snow drift preferentially filling depressions in cool years.
* Buried lakes staying liquid for several years. A partial version is
  available through `refreeze < 1`.
* Channel incision lowering the sill, which competes with floor subsidence
  once a lake overtops (link to the Saint-Venant/incision channel model).
* Nonlinear (n = 3) load response. Glen's law is only linearised about the
  background strain rate.
* A hydrofracture threshold for pinned, deepening lakes.
* The slow thickness mode of the layer solution, as a second viscous element.
