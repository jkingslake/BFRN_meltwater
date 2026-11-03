# Ice-shelf flexure under meltwater lakes: a 2-D viscoelastic beam model

When a meltwater lake fills on a floating ice shelf, its weight bends the ice.
This folder contains a small, fast, well-tested model of that bending. It
treats the shelf as a thin **Maxwell viscoelastic beam** floating on the ocean,
which acts as a **hydrostatic (Winkler) foundation**. The folder also contains
a step-by-step derivation with diagrams and a narrated explainer video.

![fill-drain](figures/fill_drain_timeseries.png)

| | |
|---|---|
| **Derivation (start here)** | [`derivation.md`](derivation.md): physics → master equation → limits → relaxation spectrum → numerics → verification, with diagrams at every step. Also as a PDF: [`derivation.pdf`](derivation.pdf) (rebuild with `python make_pdf.py`) |
| **Explainer video** | [`video/ice_shelf_flexure_explainer.mp4`](video/ice_shelf_flexure_explainer.mp4): Manim animation with a British-English voice-over and subtitles ([narration script](video/narration_script.md)) |
| **Model** | [`viscoelastic_beam.py`](viscoelastic_beam.py): one file, NumPy and SciPy only |
| **Worked examples** | [`run_examples.py`](run_examples.py): writes the result figures in `figures/` |
| **Diagrams** | [`make_diagrams.py`](make_diagrams.py): writes the schematic figures in `figures/diagram_*.png` |
| **Tests** | [`tests/test_viscoelastic_beam.py`](tests/test_viscoelastic_beam.py): checks against exact solutions |

## The model in one equation

The deflection $w(x,t)$ is positive downwards. Under a lake load
$q = \rho_w g\,d(x,t)$ it obeys

$$
\Big(D\,\frac{\partial^4}{\partial x^4} + k\Big)\frac{\partial w}{\partial t}
= \frac{\partial q}{\partial t} + \frac{q - k\,w}{\tau},
\qquad D = \frac{EH^3}{12(1-\nu^2)},\quad k = \rho_{sw}g,\quad \tau = \frac{4\eta(1-\nu^2)}{E}.
$$

* **On loading, the response is elastic** ($D w'''' + kw = q$). The dip spreads
  over a few flexural lengths $\ell = (D/k)^{1/4}$, which is about 600 m here.
* **Over a few Maxwell times $\tau$** (about 4 days here) the shelf relaxes
  towards **local isostasy**, $w = q/k$.
* **A feature of wavelength $\lambda$ relaxes on the timescale
  $\tau[1 + (2\pi\ell/\lambda)^4]$**, so narrow features relax slowest.
* **Draining a lake that has sat for a while flips the stress.** The surface
  under the old lake goes into tension.

## Quick start

```bash
pip install numpy scipy matplotlib pytest
cd flexure
python viscoelastic_beam.py   # parameters and a timing demo
python run_examples.py        # figures/*.png
python -m pytest -q           # 12 verification tests
```

```python
import numpy as np
from viscoelastic_beam import IceShelf, ViscoelasticBeam, parabolic_lake, SECONDS_PER_DAY

shelf = IceShelf(H=250, E=1e9, eta=1e14)        # SI units
x = np.linspace(0, 30e3, 3001)                  # grounding line at x = 0
beam = ViscoelasticBeam(x, shelf, left="clamped", right="free")
depth = parabolic_lake(x, centre=8e3, half_width=1.5e3, max_depth=4.0)
t = np.linspace(0, 30, 601) * SECONDS_PER_DAY
res = beam.run(t, lambda tt: shelf.lake_load(depth) * min(tt / (10 * SECONDS_PER_DAY), 1))
res.w       # deflection [m], shape (time, x)
res.sigma   # upper-surface bending stress [Pa], + = tension
```

`load` can be any function `q(t)` or an array of shape `(nt, nx)`. `H` can be
an array, for a shelf of varying thickness. Boundary conditions can be
`'free'` (calving front or far field) or `'clamped'` (grounding line).

## Numerics in brief

* **Space.** Second-order finite differences. Two sparse operators do the work:
  $L$ maps deflection to curvature and holds the kinematic boundary conditions;
  $L^*$ maps moment to force and holds the dynamic ones. The bending operator
  is $L^* D L$.
* **Time.** The trapezoidal (Crank–Nicolson) rule is applied to the Maxwell
  law and substituted into the force balance. This gives one constant
  pentadiagonal system, which is factorised once, so each step is an $O(N)$
  back-substitution.
* **Speed.** 3000 nodes × 1200 steps run in about one second.
* **Accuracy.** The scheme is second order in $\Delta x$ and $\Delta t$ and
  unconditionally stable, and the discrete force balance holds to round-off
  at every step.

## Rebuilding the video

```bash
pip install manim kokoro-onnx soundfile       # plus LaTeX and ffmpeg
# Kokoro TTS model files -> ~/kokoro (see video/voiceover.py)
cd flexure/video && ./render.sh               # or: ./render.sh low
```

The voice-over is synthesised offline with the open-weights
[Kokoro](https://github.com/thewh1teagle/kokoro-onnx) TTS model, using its
British English voice `bf_emma` with `en-gb` pronunciation.
