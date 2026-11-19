# Cross-verification against independent codes

`viscoelastic_plate.py` (`SpectralPlate`, `FDPlate`) is checked here against two
codes written by other people:

* **GMT `grdflexure`** (Wessel, GMT 6.5): FFT flexure, elastic and Maxwell
  viscoelastic, uniform thickness.
* **gFlex** (Wickert 2016, v1.3.0, GPL-3.0): finite-difference elastic flexure
  with variable thickness and a choice of edge conditions.

Start with [`cross_verification.ipynb`](cross_verification.ipynb). The fast,
coarse versions of the same checks are in
[`../tests/test_cross_verification.py`](../tests/test_cross_verification.py),
and are skipped if GMT or gFlex is missing.

| case | reference | max difference / peak |
|---|---|---|
| A1 broad load (density mapping) | analytic isostasy | 3e-5 |
| A2 elastic lake, uniform H | GMT (FFT) | 3e-7 with GMT's g; 9e-5 with g = 9.81 |
| A3 viscoelastic fill–hold–drain | GMT `-M` | 2e-7 (with ν_v = ν) |
| B1 elastic lake, periodic | gFlex FD | 1e-12 to 5e-9 |
| B2 clamped box edges | gFlex FD | 3e-3 → 8e-4 (first order in dx) |
| B3 rotated grounding line | exact 1-D clamped beam | 1–8% at dx = 200 m; 0.3–2% at 50 m |
| B4 variable thickness 200–600 m | gFlex FD vs 1-D beam | 2e-5 → 1e-6 (second order) |

**What this verifies.** The uniform-thickness viscoelastic physics and its time
integration (to round-off, against GMT), and the finite-difference interior
including variable thickness (against gFlex).

**What it shows to watch.**
* A grounding line is a stair-stepped clamp in `FDPlate`. Its error is first
  order: about 0.2–0.4 · dx/ℓ, which is 2–6% for the 128 m grids in the routing
  notebooks (ℓ ≈ 900 m for H = 400 m). Fixing nodes within half a cell of the
  grounding line (s < dx/2) instead of only those inside it cuts the error
  2–4×.
* Our default ν_v = ½ gives two relaxation times. GMT's Maxwell model has one
  (ν_v = ν). The two differ by up to 0.3% of the peak; that is a model
  difference, not an error.

**Not externally verified.** 2-D viscoelastic flexure with varying H;
viscoelastic flexure with a grounding line; free calving fronts in 2-D.

## Running

```bash
# GMT >= 6.5 (the gmt command line is used; PyGMT is not needed)
sudo apt install gmt gmt-dcw gmt-gshhg   # or: brew install gmt / conda install -c conda-forge gmt
pip install gflex
cd flexure/verification && jupyter nbconvert --to notebook --execute --inplace cross_verification.ipynb
cd .. && python -m pytest -q tests/test_cross_verification.py
```

The notebook takes about two minutes. Versions used: GMT 6.5.0, gFlex 1.3.0.

## `gmt_flexure.py`: notes on driving `grdflexure`

* **Densities.** grdflexure has no "water load on a floating plate" mode, so
  dummy densities are used: `-D3028/2000/2000/1000` (ρ_m = 2000 + ρ_sw, load
  and infill 2000, "water" 2000 − ρ_water) with the lake depth as the load
  topography. This gives exactly D∇⁴w + ρ_sw g w = ρ_water g d. Values below 10
  are read as g/cm³, so 0 and 1 cannot be used directly.
* **Gravity.** GMT uses g = 9.806199203 m/s² (`GMT_G`); use it in our
  `IceShelf` to compare at round-off.
* **Time.** Ages run positive back in time. In GMT 6.5, `-T<file>` and
  filename-template inputs crash; a load list `loads.lis+l` of
  tab-separated `age  grid  tag` records with a regular `-T t0/t1/dt` works.
  Keep load steps off output times: at an output time exactly equal to a step
  time, the step is included or not depending on round-off in the age.
* **FFT.** `-Nf+n+l` (no extension, taper or detrending) makes the domain
  periodic, like `SpectralPlate`. Output is negative down; the wrapper returns
  positive down.
