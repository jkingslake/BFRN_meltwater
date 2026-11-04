"""
Manim explainer: "How meltwater lakes bend ice shelves".

Render everything (and stitch it together with the voice-over) with
    ./render.sh            # 1080p
or a single scene for checking with
    manim -ql explainer.py S08Limits

Every scene narrates itself through ``voiceover`` (British English, offline
Kokoro TTS).  The simulations shown are computed live by ../viscoelastic_beam.py.
"""
import os
import sys

import numpy as np
from manim import *

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from viscoelastic_beam import (  # noqa: E402
    SECONDS_PER_DAY as DAY, IceShelf, ViscoelasticBeam, fill_hold_drain, parabolic_lake,
    relaxation_time)
from voiceover import voiceover  # noqa: E402

# ---------------------------------------------------------------------------
# look and feel
# ---------------------------------------------------------------------------
BG = "#0e1621"
INK = "#eceae4"
INK2 = "#b4b2a9"
MUTED = "#6f6d66"
VISCO = "#4c95ec"
ELASTIC = "#f07a45"
ISO = "#2cc28d"
ACCENT = "#a397f2"
TENSION = "#ef6262"
COMP = "#4c95ec"
ICE = "#dbe9f6"
ICE_EDGE = "#8fb3d1"
OCEAN = "#16365a"
OCEAN_LINE = "#4a7bab"
WATER = "#3fa9f5"
BED = "#5e4f3d"

config.background_color = BG
Text.set_default(font="Inter", color=INK)
MathTex.set_default(color=INK)

# The video explains the one-relaxation-time model (elastic and viscous Poisson
# ratios equal), so its simulations use that model.  The library default now has
# incompressible creep (nu_v = 1/2) and two relaxation times; see derivation.md.
SHELF = IceShelf(nu_v=0.3)
RHO = SHELF.rho_i / SHELF.rho_sw


def T(s, size=30, color=INK, **kw):
    return Text(s, font_size=size, color=color, **kw)


def M(s, size=42, color=INK, **kw):
    return MathTex(s, font_size=size, color=color, **kw)


def header(num, title):
    n = T(str(num), 26, ACCENT, weight=BOLD)
    t = T(title, 30, INK2)
    return VGroup(n, t).arrange(RIGHT, buff=0.25).to_corner(UL, buff=0.45)


def polyline(points, color=INK, width=4):
    v = VMobject(stroke_color=color, stroke_width=width)
    v.set_points_as_corners([np.array([p[0], p[1], 0.0]) for p in points])
    return v


def polygon(points, fill, opacity=1.0, stroke=None, width=2):
    pts = [np.array([p[0], p[1], 0.0]) for p in points]
    return Polygon(*pts, fill_color=fill, fill_opacity=opacity,
                   stroke_color=stroke or fill, stroke_width=width if stroke else 0)


def spring(x, y0, y1, n=6, w=0.13, color=ISO, width=3):
    ys = np.linspace(y0, y1, 2 * n + 3)
    xs = np.full_like(ys, x)
    xs[2:-2:2] += w
    xs[3:-2:2] -= w
    return polyline(np.c_[xs, ys], color, width)


def curve_on(ax, x, y, color, width=4):
    pts = ax.c2p(np.asarray(x), np.asarray(y)).T
    v = VMobject(stroke_color=color, stroke_width=width)
    v.set_points_as_corners(pts)
    return v


def plot_axes(x_range, y_range, x_length, y_length):
    """Axes whose y-axis sits on the left edge (not at x = 0)."""
    ax = Axes(x_range=x_range, y_range=y_range, x_length=x_length, y_length=y_length,
              tips=False, axis_config={"color": MUTED, "stroke_width": 2})
    ax.y_axis.set_opacity(0)
    left = Line(ax.c2p(x_range[0], y_range[0]), ax.c2p(x_range[0], y_range[1]), color=MUTED,
                stroke_width=2)
    ax.add(left)
    return ax


def water_on(ax, x, surface, depth, color=None, opacity=0.85):
    """Lake water filling the dip in ``surface``: its top is surface + depth, but never
    above the undisturbed level (0); only where depth > 0."""
    wet = depth > 1e-3
    if wet.sum() < 2:
        return VMobject()
    i0, i1 = np.flatnonzero(wet)[[0, -1]]
    xs, s = x[i0:i1 + 1], surface[i0:i1 + 1]
    top = np.maximum(np.minimum(s + depth[i0:i1 + 1], 0.0), s)
    return fill_on(ax, xs, s, top, color or WATER, opacity)


def fill_on(ax, x, ylo, yhi, color, opacity):
    lo = ax.c2p(np.asarray(x), np.asarray(ylo)).T
    hi = ax.c2p(np.asarray(x), np.asarray(yhi)).T
    pts = np.r_[hi, lo[::-1]]
    return Polygon(*pts, fill_color=color, fill_opacity=opacity, stroke_width=0)


# ---------------------------------------------------------------------------
# a reusable ice-shelf cross-section (screen units, vertical exaggeration)
# ---------------------------------------------------------------------------
class CrossSection:
    """Maps a physical deflection profile w(x) onto a drawn ice shelf."""

    def __init__(self, x_km, sx0, sx1, sea=0.0, H=1.4, wscale=0.15):
        self.x_km = np.asarray(x_km)
        self.sx = np.interp(self.x_km, [self.x_km[0], self.x_km[-1]], [sx0, sx1])
        self.sea, self.H, self.wscale = sea, H, wscale
        self.top0 = sea + H * (1 - RHO)
        self.bot0 = sea - H * RHO

    def ice(self, w):
        dz = -np.asarray(w) * self.wscale
        top, bot = self.top0 + dz, self.bot0 + dz
        pts = np.r_[np.c_[self.sx, top], np.c_[self.sx[::-1], bot[::-1]]]
        return polygon(pts, ICE, 0.92, ICE_EDGE, 2)

    def lake(self, w, depth_m):
        """Water with a flat surface filling the dip, depth taken from depth_m."""
        dz = -np.asarray(w) * self.wscale
        top = self.top0 + dz
        d = np.asarray(depth_m) * self.wscale
        if d.max() <= 1e-6:
            return VMobject()
        level = min(np.max((top + d)[d > 0]), self.top0 - 0.02)
        inside = (top < level) & (d > 0)
        idx = np.flatnonzero(inside)
        if idx.size < 2:
            return VMobject()
        i0, i1 = idx[0], idx[-1]
        xs, tops = self.sx[i0:i1 + 1], np.minimum(top[i0:i1 + 1], level)
        pts = np.r_[np.c_[xs, tops], [[xs[-1], level], [xs[0], level]]]
        return polygon(pts, WATER, 0.95)

    def ocean(self, x0=None, x1=None, depth=2.6):
        x0 = self.sx[0] if x0 is None else x0
        x1 = self.sx[-1] if x1 is None else x1
        r = Rectangle(width=x1 - x0, height=depth, fill_color=OCEAN, fill_opacity=1,
                      stroke_width=0)
        r.move_to([(x0 + x1) / 2, self.sea - depth / 2, 0])
        return r

    def sea_line(self, x0=None, x1=None):
        x0 = self.sx[0] if x0 is None else x0
        x1 = self.sx[-1] if x1 is None else x1
        return DashedLine([x0, self.sea, 0], [x1, self.sea, 0], color=OCEAN_LINE,
                          stroke_width=2, dash_length=0.12)


def lake_solution(x, half_width=1500.0, depth=4.0, centre=0.0, left="free", right="free"):
    beam = ViscoelasticBeam(x, SHELF, left, right)
    d = parabolic_lake(x, centre, half_width, depth)
    q = SHELF.lake_load(d)
    w_el, M_el = beam.elastic(q)
    return beam, d, q, w_el, M_el


# ===========================================================================
class S01Intro(Scene):
    def construct(self):
        title = T("How meltwater lakes bend ice shelves", 48, weight=BOLD)
        sub = T("a 2-D viscoelastic beam model, derived and solved", 28, INK2)
        VGroup(title, sub).arrange(DOWN, buff=0.3).to_edge(UP, buff=1.0)

        x = np.linspace(-9e3, 9e3, 901)
        _, d, _, w_el, _ = lake_solution(x)
        cs = CrossSection(x / 1e3, -7.2, 7.2, sea=-1.2, H=1.6, wscale=0.16)
        ocean, sea = cs.ocean(depth=2.9), cs.sea_line()
        f = ValueTracker(0.0)
        ice = always_redraw(lambda: cs.ice(f.get_value() * w_el))
        lake = always_redraw(lambda: cs.lake(f.get_value() * w_el, f.get_value() * d))

        with voiceover(self, "Every summer, meltwater pools into lakes on the surface of "
                             "Antarctica's floating ice shelves.") as vo:
            self.play(Write(title), run_time=1.6)
            self.play(FadeIn(sub, shift=0.2 * UP), FadeIn(ocean), Create(sea), FadeIn(ice),
                      run_time=1.4)
            self.add(lake)
            self.play(f.animate.set_value(0.35), run_time=max(vo.duration - 3.2, 0.5))
        with voiceover(self, "Those lakes are heavy, and the ice bends beneath them. Bend it "
                             "far enough, or back and forth often enough, and it can crack.") as vo:
            self.play(f.animate.set_value(1.0), run_time=vo.duration * 0.6, rate_func=smooth)
        with voiceover(self, "In this video we'll build a simple model of that bending: a "
                             "two-dimensional viscoelastic beam. We'll derive it step by step, "
                             "and then solve it on a computer in a few lines of code.") as vo:
            note = T("vertical scale exaggerated", 18, MUTED).to_corner(DR, buff=0.3)
            self.play(FadeIn(note), run_time=0.8)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)


# ===========================================================================
class S02Setup(Scene):
    def construct(self):
        hd = header(1, "The set-up")
        x = np.linspace(0, 20e3, 1001)
        _, d, _, w_el, _ = lake_solution(x, centre=10e3, left="clamped")
        cs = CrossSection(x / 1e3, -4.6, 6.6, sea=-0.6, H=1.7, wscale=0.16)
        ocean = cs.ocean(-4.6, 7.1, depth=2.6)
        sea = cs.sea_line(-4.6, 7.1)
        bed = polygon([[-7.2, cs.bot0 - 0.4], [-4.6, cs.bot0], [-3.2, -3.0], [7.2, -3.25],
                       [7.2, -4.1], [-7.2, -4.1]], BED)
        grounded = polygon([[-7.2, cs.top0 + 1.0], [-4.6, cs.top0], [-4.6, cs.bot0],
                            [-7.2, cs.bot0 - 0.4]], ICE, 0.92, ICE_EDGE, 2)
        f = ValueTracker(0.0)
        ice = always_redraw(lambda: cs.ice(f.get_value() * w_el))
        lake = always_redraw(lambda: cs.lake(f.get_value() * w_el, f.get_value() * d))

        gl = DashedLine([-4.6, cs.bot0 - 0.2, 0], [-4.6, cs.top0 + 1.6, 0], color=INK2,
                        stroke_width=2)
        gl_t = T("grounding line", 22, INK2).next_to(gl, UP, buff=0.1)
        cf = DashedLine([6.6, cs.top0 + 0.1, 0], [6.6, cs.top0 + 1.6, 0], color=INK2,
                        stroke_width=2)
        cf_t = T("calving front", 22, INK2).next_to(cf, UP, buff=0.1).align_to([7.0, 0, 0], RIGHT)
        ocean_t = T("ocean", 24, OCEAN_LINE).move_to([4.5, -2.4, 0])
        Hbr = BraceBetweenPoints([4.6, cs.top0 - 0.05, 0], [4.6, cs.bot0 + 0.05, 0],
                                 direction=LEFT, color=BG)
        H_t = M(r"H \approx 250\,\mathrm{m}", 30, BG).next_to(Hbr, LEFT, buff=0.1)

        with voiceover(self, "Take a vertical slice through an ice shelf, from the grounding "
                             "line, where the ice lifts off the bed, out to the calving front, "
                             "where it meets the open ocean.") as vo:
            self.play(FadeIn(hd), run_time=0.6)
            self.play(FadeIn(bed), FadeIn(ocean), Create(sea), FadeIn(grounded), FadeIn(ice),
                      run_time=1.5)
            self.play(Create(gl), FadeIn(gl_t), run_time=1.0)
            self.play(Create(cf), FadeIn(cf_t), FadeIn(ocean_t), run_time=1.0)
        with voiceover(self, "The ice is a few hundred metres thick, and it floats, with "
                             "roughly nine tenths of it below sea level.") as vo:
            self.play(GrowFromCenter(Hbr), Write(H_t), run_time=1.2)
        # the lake
        qarrows = VGroup(*[Arrow([xx, cs.top0 + 1.0, 0], [xx, cs.top0 + 0.15, 0], buff=0,
                                 color=ELASTIC, stroke_width=4, max_tip_length_to_length_ratio=0.3)
                           for xx in np.linspace(cs.sx[400], cs.sx[600], 5)])
        q_t = M(r"q = \rho_w\, g\, d", 34, ELASTIC).next_to(qarrows, UP, buff=0.12)
        i0 = 500
        warrow = always_redraw(lambda: Arrow(
            [cs.sx[i0], cs.bot0, 0], [cs.sx[i0], cs.bot0 - f.get_value() * w_el[i0] * cs.wscale, 0],
            buff=0, color=ACCENT, stroke_width=5, max_tip_length_to_length_ratio=0.5))
        w_t = M(r"w(x,t)", 32, ACCENT).move_to([cs.sx[i0] + 1.0, cs.bot0 - 0.85, 0])
        with voiceover(self, "Now add a lake of depth d. Its weight presses down on the ice "
                             "with a load q, equal to rho w, g, d. The ice is pushed down by a "
                             "distance w, which we'll measure positive downwards.") as vo:
            self.add(lake)
            self.play(f.animate.set_value(1.0), run_time=2.5)
            self.play(LaggedStart(*[GrowArrow(a) for a in qarrows], lag_ratio=0.1),
                      Write(q_t), run_time=1.5)
            self.add(warrow)
            self.play(Write(w_t), run_time=1.0)
        axes = VGroup(Arrow([-6.6, -3.6, 0], [-5.4, -3.6, 0], buff=0, color=INK2, stroke_width=3),
                      Arrow([-6.6, -3.6, 0], [-6.6, -2.4, 0], buff=0, color=INK2, stroke_width=3))
        ax_l = VGroup(M("x", 30, INK2).next_to(axes[0], RIGHT, buff=0.1),
                      M("z", 30, INK2).next_to(axes[1], UP, buff=0.1))
        with voiceover(self, "Our goal is to predict w, as a function of position x and time t, "
                             "and the stresses that come with it.") as vo:
            self.play(Create(axes), FadeIn(ax_l), Indicate(w_t, color=ACCENT), run_time=1.5)
        assumptions = VGroup(
            T("thin beam", 24), T("small deflections", 24),
            T("nothing varies across the page (2-D)", 24),
            T("ocean responds hydrostatically", 24)).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        bullets = VGroup(*[VGroup(Dot(radius=0.05, color=ACCENT), a).arrange(RIGHT, buff=0.18)
                           for a in assumptions]).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        box = SurroundingRectangle(bullets, color=MUTED, buff=0.25, corner_radius=0.12,
                                   fill_color=BG, fill_opacity=0.92)
        card = VGroup(box, bullets).move_to([0.6, 2.35, 0])
        with voiceover(self, "We'll assume the ice is a thin beam, the deflections are small, "
                             "nothing varies across the page, and the ocean responds "
                             "hydrostatically.") as vo:
            self.play(FadeOut(q_t), FadeOut(qarrows), FadeIn(box),
                      LaggedStart(*[FadeIn(b, shift=0.1 * RIGHT) for b in bullets], lag_ratio=0.5),
                      run_time=min(vo.duration, 4))
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)


# ===========================================================================
class S03Buoyancy(Scene):
    def construct(self):
        hd = header(2, "Buoyancy is a bed of springs")
        sea = -0.4
        water = Rectangle(width=5.6, height=3.2, fill_color=OCEAN, fill_opacity=1, stroke_width=0)
        water.move_to([-3.6, sea - 1.6, 0])
        sl = DashedLine([-6.4, sea, 0], [-0.8, sea, 0], color=OCEAN_LINE, stroke_width=2)
        Hc, rho_c = 2.4, 0.66  # cartoon proportions
        col0 = Rectangle(width=1.1, height=Hc, fill_color=ICE, fill_opacity=0.92,
                         stroke_color=ICE_EDGE).move_to([-5.0, sea + Hc * (0.5 - rho_c), 0])
        col = col0.copy().move_to([-2.4, sea + Hc * (0.5 - rho_c), 0])
        free_t = T("floating freely", 22, INK2).next_to(col0, UP, buff=0.3)
        with voiceover(self, "First ingredient: buoyancy. A floating column of ice sinks until "
                             "the sea water it pushes aside weighs as much as the ice.") as vo:
            self.play(FadeIn(hd), FadeIn(water), Create(sl), FadeIn(col0), FadeIn(col),
                      FadeIn(free_t), run_time=2.0)
        wshift = 0.55
        lake = Rectangle(width=1.1, height=0.3, fill_color=WATER, fill_opacity=1,
                         stroke_width=0).next_to(col, UP, buff=0)
        extra = Rectangle(width=1.1, height=wshift, fill_color=ISO, fill_opacity=0.45,
                          stroke_width=0)
        extra.move_to(col.get_bottom() + DOWN * wshift / 2)
        up = VGroup(*[Arrow([xx, col.get_bottom()[1] - wshift - 1.0, 0],
                            [xx, col.get_bottom()[1] - wshift - 0.08, 0], buff=0, color=ISO,
                            stroke_width=5) for xx in (-2.7, -2.1)])
        up_t = M(r"\rho_{sw}\, g\, w", 34, ISO).next_to(up, DOWN, buff=0.1)
        wbr = BraceBetweenPoints(col.get_corner(DR) + DOWN * wshift, col.get_corner(DR),
                                 direction=RIGHT, color=INK2)
        w_t = M("w", 30, INK2).next_to(wbr, RIGHT, buff=0.08)
        with voiceover(self, "Put a lake on top, and the column is pushed down by w. It "
                             "displaces extra sea water, so the ocean pushes back with an extra "
                             "pressure: rho s w, g, w.") as vo:
            self.play(FadeIn(lake, shift=0.3 * DOWN), run_time=0.8)
            self.play(VGroup(col, lake).animate.shift(DOWN * wshift), run_time=1.5)
            self.add(extra)
            self.play(FadeIn(extra), GrowFromCenter(wbr), Write(w_t), run_time=1.0)
            self.play(LaggedStart(*[GrowArrow(a) for a in up], lag_ratio=0.2), Write(up_t),
                      run_time=1.2)
        # springs
        slab = Rectangle(width=5.0, height=0.55, fill_color=ICE, fill_opacity=0.92,
                         stroke_color=ICE_EDGE).move_to([3.7, 0.9, 0])
        ground = Line([1.0, -1.6, 0], [6.4, -1.6, 0], color=INK2, stroke_width=3)
        hatch = VGroup(*[Line([xx, -1.6, 0], [xx - 0.2, -1.85, 0], color=INK2, stroke_width=2)
                         for xx in np.arange(1.2, 6.45, 0.25)])
        springs = VGroup(*[spring(xx, -1.6, slab.get_bottom()[1]) for xx in np.linspace(1.5, 5.9, 9)])
        k_t = M(r"k = \rho_{sw}\, g \approx 10^4\ \mathrm{Pa\,m^{-1}}", 34, ISO).move_to([3.7, -2.4, 0])
        with voiceover(self, "That's exactly like resting the ice on a bed of springs, with "
                             "stiffness k equal to rho s w, g: about ten thousand pascals per "
                             "metre.") as vo:
            self.play(FadeIn(slab), Create(ground), FadeIn(hatch),
                      LaggedStart(*[Create(s) for s in springs], lag_ratio=0.08), run_time=2.2)
            self.play(Write(k_t), run_time=1.3)
        # independent columns -> isostasy
        cols = VGroup(*[Rectangle(width=5.0 / 10, height=0.55, fill_color=ICE, fill_opacity=0.92,
                                  stroke_color=ICE_EDGE, stroke_width=1.5)
                        .move_to([1.2 + 5.0 / 20 + i * 0.5, 0.9, 0]) for i in range(10)])
        dips = np.clip(1 - ((np.arange(10) - 4.5) / 2.6) ** 2, 0, None) * 0.7
        iso = M(r"k\,w = q", 40, ISO).move_to([3.7, 2.4, 0])
        iso_t = T("local isostasy: the dip copies the lake", 24, INK2).next_to(iso, DOWN, buff=0.15)
        with voiceover(self, "If the ice had no stiffness at all, each column would simply sink "
                             "until k w equals q, and the dip would copy the lake exactly. "
                             "That's called local isostasy.") as vo:
            self.remove(slab)
            self.add(cols)
            new_springs = VGroup(*[spring(xx, -1.6, 0.9 - 0.275 - np.interp(xx, cols_x, dips))
                                   for xx, cols_x in zip(np.linspace(1.5, 5.9, 9),
                                                         [np.linspace(1.45, 5.95, 10)] * 9)])
            self.play(*[c.animate.shift(DOWN * dd) for c, dd in zip(cols, dips)],
                      Transform(springs, new_springs), run_time=2.0)
            self.play(Write(iso), FadeIn(iso_t), run_time=1.5)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)


# ===========================================================================
class S04ForceBalance(Scene):
    def construct(self):
        hd = header(3, "Force balance on a slice")
        sl = Rectangle(width=3.0, height=1.4, fill_color=ICE, fill_opacity=0.92,
                       stroke_color=ICE_EDGE).move_to([-3.0, 0.0, 0])
        dx = BraceBetweenPoints(sl.get_corner(DL) + 0.1 * DOWN, sl.get_corner(DR) + 0.1 * DOWN,
                                direction=DOWN, color=INK2)
        dx_t = M("dx", 30, INK2).next_to(dx, DOWN, buff=0.08)
        with voiceover(self, "But ice is stiff, so each column drags on its neighbours. Cut out "
                             "a thin slice of width d x.") as vo:
            self.play(FadeIn(hd), run_time=0.6)
            self.play(DrawBorderThenFill(sl), run_time=1.4)
            self.play(GrowFromCenter(dx), Write(dx_t), run_time=1.0)
        load = VGroup(*[Arrow([xx, 1.8, 0], [xx, 0.72, 0], buff=0, color=ELASTIC, stroke_width=5)
                        for xx in np.linspace(-4.2, -1.8, 5)])
        load_t = M(r"q\,dx", 32, ELASTIC).next_to(load, UP, buff=0.1)
        buoy = VGroup(*[Arrow([xx, -3.0, 0], [xx, -1.75, 0], buff=0, color=ISO, stroke_width=5)
                        for xx in np.linspace(-4.2, -1.8, 5)])
        buoy_t = M(r"k\,w\,dx", 32, ISO).next_to(buoy, DOWN, buff=0.1)
        V1 = Arrow([-4.75, 0.7, 0], [-4.75, -0.7, 0], buff=0, color=INK, stroke_width=5)
        V2 = Arrow([-1.25, -0.7, 0], [-1.25, 0.7, 0], buff=0, color=INK, stroke_width=5)
        V1_t = M("V", 32).next_to(V1, LEFT, buff=0.1)
        V2_t = M("V+dV", 32).next_to(V2, RIGHT, buff=0.1)
        M1 = Arc(radius=1.65, start_angle=PI * 2 / 3, angle=PI * 2 / 3, arc_center=[-4.5, 0, 0],
                 color=ACCENT, stroke_width=5).add_tip(tip_length=0.22)
        M2 = Arc(radius=1.65, start_angle=PI / 3, angle=-PI * 2 / 3, arc_center=[-1.5, 0, 0],
                 color=ACCENT, stroke_width=5).add_tip(tip_length=0.22)
        M1_t = M("M", 32, ACCENT).move_to([-6.0, 1.45, 0])
        M2_t = M("M+dM", 32, ACCENT).move_to([0.55, 1.45, 0])
        with voiceover(self, "The lake pushes down, buoyancy pushes up, and the ice on either "
                             "side exerts shear forces, V, and bending moments, M.") as vo:
            self.play(LaggedStart(*[GrowArrow(a) for a in load], lag_ratio=0.1), Write(load_t),
                      run_time=1.2)
            self.play(LaggedStart(*[GrowArrow(a) for a in buoy], lag_ratio=0.1), Write(buoy_t),
                      run_time=1.2)
            self.play(GrowArrow(V1), GrowArrow(V2), Write(V1_t), Write(V2_t), run_time=1.2)
            self.play(Create(M1), Create(M2), Write(M1_t), Write(M2_t), run_time=1.4)
        e1 = M(r"\frac{dV}{dx} = q - k\,w", 44)
        e2 = M(r"\frac{dM}{dx} = V", 44)
        l1 = T("vertical forces", 22, INK2)
        l2 = T("moments", 22, INK2)
        g1 = VGroup(l1, e1).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        g2 = VGroup(l2, e2).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        eqs = VGroup(g1, g2).arrange(DOWN, aligned_edge=LEFT, buff=0.5).move_to([3.9, 1.0, 0])
        with voiceover(self, "Balancing vertical forces gives: d V by d x equals q minus k w. "
                             "Balancing moments gives: d M by d x equals V.") as vo:
            self.play(FadeIn(l1), Write(e1), run_time=2.0)
            self.wait(max(vo.duration / 2 - 2.0, 0))
            self.play(FadeIn(l2), Write(e2), run_time=1.6)
        key = M(r"\frac{\partial^2 M}{\partial x^2} + k\,w = q", 50)
        key.move_to([3.9, -1.9, 0])
        kbox = SurroundingRectangle(key, color=ACCENT, buff=0.22, corner_radius=0.1)
        with voiceover(self, "Put them together, and we have our first key equation: the second "
                             "derivative of M, plus k w, equals q. Wherever buoyancy isn't "
                             "holding up the lake, bending moments must.") as vo:
            self.play(TransformFromCopy(VGroup(e1, e2), key), run_time=1.8)
            self.play(Create(kbox), run_time=0.8)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)


# ===========================================================================
class S05Bending(Scene):
    def construct(self):
        hd = header(4, "Bending: curvature and moment")
        # a beam that bends into a hump
        bend = ValueTracker(0.0)
        L, h = 5.0, 1.0
        c0 = np.array([-3.3, -0.4, 0])

        def beam_shape():
            b = bend.get_value()
            s = np.linspace(-L / 2, L / 2, 80)
            if b < 1e-4:
                xm, ym = s, np.zeros_like(s)
                nx, ny = np.zeros_like(s), np.ones_like(s)
            else:
                R = 1 / b
                th = s / R
                xm, ym = R * np.sin(th), R * np.cos(th) - R
                nx, ny = np.sin(th), np.cos(th)
            top = np.c_[xm + nx * h / 2, ym + ny * h / 2]
            bot = np.c_[xm - nx * h / 2, ym - ny * h / 2]
            body = polygon(np.r_[top, bot[::-1]] + c0[:2], ICE, 0.92, ICE_EDGE, 2)
            midline = polyline(np.c_[xm, ym] + c0[:2], MUTED, 2)
            sections = VGroup(*[Line([*(bot[i] + c0[:2]), 0], [*(top[i] + c0[:2]), 0],
                                     color=ICE_EDGE, stroke_width=2) for i in (20, 40, 59)])
            return VGroup(body, midline, sections)

        beam = always_redraw(beam_shape)
        with voiceover(self, "Next: how does the bending moment depend on the shape? When a "
                             "beam bends, its cross-sections stay flat, but they tilt.") as vo:
            self.play(FadeIn(hd), FadeIn(beam), run_time=1.0)
            self.play(bend.animate.set_value(0.16), run_time=2.5, rate_func=there_and_back_with_pause)
            self.play(bend.animate.set_value(0.16), run_time=1.2)
        top_t = T("top stretched", 22, TENSION).move_to(c0 + [0, 1.15, 0])
        bot_t = T("bottom squashed", 22, COMP).move_to(c0 + [0, -1.45, 0])
        strain = M(r"\varepsilon = z\,\kappa,\qquad \kappa = \frac{\partial^2 w}{\partial x^2}", 40)
        strain.move_to([-3.3, -2.7, 0])
        with voiceover(self, "So a fibre a height z above the middle is stretched by z times "
                             "kappa, where kappa, the curvature, is the second derivative of "
                             "w.") as vo:
            self.play(FadeIn(top_t), FadeIn(bot_t), run_time=1.0)
            self.play(Write(strain), run_time=2.0)
        # stress profile
        ox, oy, hh = 1.5, 0.4, 2.4
        axis = Line([ox, oy - hh / 2, 0], [ox, oy + hh / 2, 0], color=INK2, stroke_width=3)
        prof = Line([ox - 1.0, oy - hh / 2, 0], [ox + 1.0, oy + hh / 2, 0], color=ACCENT,
                    stroke_width=5)
        arrs = VGroup()
        for zz in np.linspace(-hh / 2 + 0.15, hh / 2 - 0.15, 7):
            xx = ox + zz / (hh / 2)
            if abs(xx - ox) > 0.12:
                arrs.add(Arrow([ox, oy + zz, 0], [xx, oy + zz, 0], buff=0, stroke_width=3,
                               color=TENSION if zz > 0 else COMP,
                               max_tip_length_to_length_ratio=0.35))
        sig = M(r"\sigma = E'\,z\,\kappa", 34, ACCENT).next_to(prof, UP, buff=0.15)
        Mint = M(r"M = \int_{-H/2}^{H/2}\sigma\, z\,dz", 40).move_to([4.8, 1.4, 0])
        Mel = M(r"M = D\,\kappa", 46).move_to([4.8, 0.2, 0])
        Dd = M(r"D = \frac{E H^3}{12(1-\nu^2)}", 40, ELASTIC).move_to([4.8, -1.0, 0])
        with voiceover(self, "For an elastic material, stress is proportional to strain. Add up "
                             "the turning effect of every fibre, and you get: M equals D kappa.") as vo:
            self.play(Create(axis), Create(prof), LaggedStart(*[GrowArrow(a) for a in arrs]),
                      Write(sig), run_time=2.0)
            self.play(Write(Mint), run_time=1.5)
            self.play(Write(Mel), run_time=1.2)
        with voiceover(self, "The flexural rigidity, D, goes as the thickness cubed, so doubling "
                             "the thickness makes the shelf eight times stiffer.") as vo:
            self.play(Write(Dd), run_time=1.5)
            self.play(Indicate(Dd, color=ELASTIC, scale_factor=1.15), run_time=1.2)
        Mv = M(r"M = \Lambda\,\dot\kappa,\qquad \Lambda = \frac{\eta H^3}{3}", 40, VISCO)
        Mv.move_to([4.6, -2.5, 0])
        with voiceover(self, "For a viscous fluid, stress depends instead on how fast the strain "
                             "changes. The same sum gives: M equals capital lambda, times the "
                             "rate of change of kappa, where lambda depends on the viscosity, "
                             "eta.") as vo:
            self.play(Write(Mv), run_time=2.5)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)


# ===========================================================================
class S06Maxwell(Scene):
    def construct(self):
        hd = header(5, "Ice is a Maxwell material")
        ext = ValueTracker(0.0)   # elastic stretch
        flow = ValueTracker(0.0)  # viscous stretch
        y = 1.6

        def element():
            e, v = ext.get_value(), flow.get_value()
            x0 = -5.6
            xs = np.linspace(x0 + 0.6, x0 + 2.4 + e, 13)
            ys = y + np.r_[0, np.tile([0.28, -0.28], 5), 0.0, 0.0][:13]
            ys[0] = ys[-1] = y
            sp = polyline(np.c_[xs, ys], ELASTIC, 5)
            a = x0 + 2.4 + e
            parts = VGroup(
                Line([x0, y, 0], [x0 + 0.6, y, 0], color=INK, stroke_width=4), sp,
                Line([a, y, 0], [a + 0.6, y, 0], color=INK, stroke_width=4),
                # dashpot cylinder
                Line([a + 0.6, y - 0.42, 0], [a + 0.6, y + 0.42, 0], color=VISCO, stroke_width=5),
                Line([a + 0.6, y + 0.42, 0], [a + 2.2, y + 0.42, 0], color=VISCO, stroke_width=5),
                Line([a + 0.6, y - 0.42, 0], [a + 2.2, y - 0.42, 0], color=VISCO, stroke_width=5),
                # piston
                Line([a + 1.3 + v, y - 0.3, 0], [a + 1.3 + v, y + 0.3, 0], color=VISCO,
                     stroke_width=8),
                Line([a + 1.3 + v, y, 0], [a + 3.0 + v, y, 0], color=INK, stroke_width=4))
            return parts

        el = always_redraw(element)
        sp_t = T("spring: elastic", 22, ELASTIC).move_to([-4.1, 2.45, 0])
        dp_t = T("dashpot: viscous", 22, VISCO).move_to([-1.6, 2.45, 0])
        with voiceover(self, "Real ice is both. Over hours it behaves like a spring, but over "
                             "weeks it flows. The simplest material that does both is a Maxwell "
                             "element: a spring and a dashpot, joined in series.") as vo:
            self.play(FadeIn(hd), FadeIn(el), FadeIn(sp_t), FadeIn(dp_t), run_time=1.2)
            self.play(ext.animate.set_value(0.6), run_time=0.4, rate_func=rush_from)
            self.play(flow.animate.set_value(0.8), run_time=max(vo.duration - 2.2, 1.0),
                      rate_func=linear)
        law = MathTex(r"\dot\kappa", "=", r"\frac{\dot M}{D}", "+", r"\frac{M}{\Lambda}",
                      font_size=56)
        law.move_to([-3.5, -0.4, 0])
        law[2].set_color(ELASTIC)
        law[4].set_color(VISCO)
        with voiceover(self, "Their rates of bending simply add.") as vo:
            self.play(Write(law), run_time=min(vo.duration, 2.0))
        # creep and relaxation graphs
        a1 = Axes(x_range=[-0.5, 4, 1], y_range=[0, 5, 1], x_length=2.6, y_length=2.0,
                  tips=False, axis_config={"color": INK2, "stroke_width": 2})
        a2 = a1.copy()
        VGroup(a1, a2).arrange(RIGHT, buff=0.9).move_to([3.9, -0.45, 0])
        t = np.linspace(-0.5, 4, 200)
        creep = curve_on(a1, t, np.where(t < 0, 0.05, 1 + t), VISCO)
        relax = curve_on(a2, t, np.where(t < 0, 0.05, 4.5 * np.exp(-t)), VISCO)
        a1_t = T("hold M fixed", 20, INK2).next_to(a1, UP, buff=0.15)
        a2_t = T("hold κ fixed", 20, INK2).next_to(a2, UP, buff=0.15)
        a1_y = M(r"\kappa", 28, INK2).next_to(a1, LEFT, buff=0.1)
        a2_y = M("M", 28, INK2).next_to(a2, LEFT, buff=0.1)
        a1_x = M(r"t", 26, INK2).next_to(a1, DOWN, buff=0.1)
        a2_x = M(r"t", 26, INK2).next_to(a2, DOWN, buff=0.1)
        c_t = T("creep", 20, VISCO).move_to(a1.c2p(2.0, 4.3))
        r_t = M(r"e^{-t/\tau}", 30, VISCO).move_to(a2.c2p(2.2, 2.4))
        with voiceover(self, "Hold the moment fixed, and the curvature jumps, then creeps "
                             "steadily. Hold the curvature fixed, and the moment relaxes away "
                             "exponentially.") as vo:
            self.play(Create(a1), FadeIn(a1_t), FadeIn(a1_y), FadeIn(a1_x), run_time=0.8)
            self.play(Create(creep), FadeIn(c_t), run_time=2.0)
            self.play(Create(a2), FadeIn(a2_t), FadeIn(a2_y), FadeIn(a2_x), run_time=0.8)
            self.play(Create(relax), FadeIn(r_t), run_time=2.0)
        tau = M(r"\tau = \frac{\Lambda}{D} = \frac{4\eta(1-\nu^2)}{E} \approx 4\ \text{days}", 44)
        tau.move_to([0.0, -2.75, 0])
        tbox = SurroundingRectangle(tau, color=ACCENT, buff=0.2, corner_radius=0.1)
        with voiceover(self, "The timescale is the Maxwell time, tau, equal to capital lambda "
                             "over D. For typical ice-shelf values, tau is about four days.") as vo:
            self.play(Write(tau), run_time=2.2)
            self.play(Create(tbox), run_time=0.8)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)


# ===========================================================================
class S07Master(Scene):
    def construct(self):
        hd = header(6, "One master equation")
        e1 = M(r"\frac{\partial^2 M}{\partial x^2} + k\,w = q", 40)
        e2 = M(r"\dot M = D\,\frac{\partial^2 \dot w}{\partial x^2} - \frac{M}{\tau}", 40)
        l1 = T("force balance", 22, INK2).next_to(e1, UP, buff=0.15)
        l2 = T("Maxwell law", 22, INK2).next_to(e2, UP, buff=0.15)
        g = VGroup(VGroup(l1, e1), VGroup(l2, e2)).arrange(RIGHT, buff=1.6).move_to([0, 2.0, 0])
        master = MathTex(r"\Big(D\,\frac{\partial^4}{\partial x^4} + k\Big)",
                         r"\frac{\partial w}{\partial t}", "=",
                         r"\frac{\partial q}{\partial t}", "+", r"\frac{q - k\,w}{\tau}",
                         font_size=58)
        master.move_to([0, -0.3, 0])
        box = SurroundingRectangle(master, color=ACCENT, buff=0.25, corner_radius=0.12)
        with voiceover(self, "Now we combine the two. Take two x derivatives of the Maxwell law, "
                             "and use the force balance to replace the moments.") as vo:
            self.play(FadeIn(hd), FadeIn(l1), Write(e1), run_time=1.5)
            self.play(FadeIn(l2), Write(e2), run_time=1.5)
        with voiceover(self, "Everything collapses into a single master equation for the "
                             "deflection.") as vo:
            self.play(TransformFromCopy(VGroup(e1, e2), master), run_time=2.0)
            self.play(Create(box), run_time=0.6)
        b1 = Brace(master[0], DOWN, color=ELASTIC)
        t1 = T("elastic beam on springs", 22, ELASTIC).next_to(b1, DOWN, buff=0.1)
        b2 = Brace(master[3], DOWN, color=ELASTIC)
        t2 = T("instant\nelastic response", 22, ELASTIC).next_to(b2, DOWN, buff=0.1)
        b3 = Brace(master[5], DOWN, color=VISCO)
        t3 = T("creep until\nk w = q", 22, VISCO).next_to(b3, DOWN, buff=0.1)
        with voiceover(self, "Read it like this. The bracket on the left is just an elastic beam, "
                             "resting on springs.") as vo:
            self.play(master[0].animate.set_color(ELASTIC), GrowFromCenter(b1), FadeIn(t1),
                      run_time=1.5)
        with voiceover(self, "The first term on the right says that any change in the load gets "
                             "an instant, elastic response.") as vo:
            self.play(master[3].animate.set_color(ELASTIC), GrowFromCenter(b2), FadeIn(t2),
                      run_time=1.5)
        with voiceover(self, "And the second says that, wherever buoyancy isn't yet carrying "
                             "the lake, the ice keeps creeping, at a rate set by tau, until k w "
                             "equals q.") as vo:
            self.play(master[5].animate.set_color(VISCO), GrowFromCenter(b3), FadeIn(t3),
                      run_time=1.5)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)


# ===========================================================================
class S08Limits(Scene):
    def construct(self):
        hd = header(7, "Elastic at first, isostatic at last")
        el = M(r"t = 0^+:\quad D\,\frac{\partial^4 w}{\partial x^4} + k\,w = q", 38, ELASTIC)
        ell = M(r"\ell = \Big(\frac{D}{k}\Big)^{1/4} \approx 600\ \mathrm{m}", 38, ELASTIC)
        iso = M(r"t \to \infty:\quad k\,w = q", 38, ISO)
        VGroup(el, ell).arrange(RIGHT, buff=0.8).move_to([0, 2.6, 0])
        iso.next_to(VGroup(el, ell), DOWN, buff=0.3).align_to(el, LEFT)
        with voiceover(self, "That gives two limits. At the instant the lake appears, the "
                             "response is purely elastic. Its natural length scale is the "
                             "flexural length, ell: the fourth root of D over k, about six "
                             "hundred metres here.") as vo:
            self.play(FadeIn(hd), Write(el), run_time=2.0)
            self.play(Write(ell), run_time=2.0)
        with voiceover(self, "A long time later, the ice relaxes to local isostasy, and the dip "
                             "simply copies the lake.") as vo:
            self.play(Write(iso), run_time=1.6)

        # --- simulation -------------------------------------------------------
        x = np.linspace(-15e3, 15e3, 1501)
        beam, d, q, w_el, _ = lake_solution(x)
        tau = SHELF.tau
        t = np.linspace(0, 12 * tau, 481)
        res = beam.run(t, lambda _: q)
        sel = np.abs(x) <= 6e3
        xk = x[sel] / 1e3
        top = plot_axes([-6, 6, 2], [-4.5, 1.0, 1], 11.2, 2.3)
        bot = plot_axes([-6, 6, 2], [-400, 250, 200], 11.2, 1.35)
        VGroup(top, bot).arrange(DOWN, buff=0.45).move_to([0.5, -1.25, 0])
        ylab1 = T("surface (m)", 18, INK2).rotate(PI / 2).next_to(top, LEFT, buff=0.5)
        ylab2 = T("stress (kPa)", 18, INK2).rotate(PI / 2).next_to(bot, LEFT, buff=0.55)
        xlab = T("distance (km)", 18, INK2).next_to(bot.c2p(6, -400), DOWN, buff=0.1).align_to(bot, RIGHT)
        ticks = VGroup(*[T(f"{v:g}", 16, MUTED).next_to(bot.c2p(v, -400), DOWN, buff=0.08)
                         for v in range(-4, 5, 2)])
        ticks.add(*[T(f"{v:g}", 16, MUTED).next_to(top.c2p(-6, v), LEFT, buff=0.08)
                    for v in (-4, -2, 0)])
        ticks.add(*[T(f"{v:g}", 16, MUTED).next_to(bot.c2p(-6, v), LEFT, buff=0.08)
                    for v in (-400, 0, 200)])
        k_tr = ValueTracker(0.0)

        def idx():
            return min(int(round(k_tr.get_value())), t.size - 1)

        def surface():
            i = idx()
            s = -res.w[i][sel]
            ice = fill_on(top, xk, np.full_like(xk, -4.5), s, ICE, 0.12)
            water = water_on(top, xk, s, d[sel])
            return VGroup(ice, water, curve_on(top, xk, s, INK, 4))

        def stress():
            i = idx()
            sg = res.sigma[i][sel] / 1e3
            return VGroup(fill_on(bot, xk, np.zeros_like(xk), np.clip(sg, 0, None), TENSION, 0.75),
                          fill_on(bot, xk, np.clip(sg, None, 0), np.zeros_like(xk), COMP, 0.75),
                          curve_on(bot, xk, sg, INK, 2))

        el_curve = DashedVMobject(curve_on(top, xk, -w_el[sel], ELASTIC, 4), num_dashes=60)
        iso_curve = DashedVMobject(curve_on(top, xk, -q[sel] / SHELF.k, ISO, 4), num_dashes=60)
        surf = always_redraw(surface)
        strs = always_redraw(stress)
        clock = always_redraw(lambda: T(f"t = {t[idx()] / DAY:4.1f} days", 24, INK).move_to(
            top.c2p(4.2, -3.6)))
        lg = VGroup(
            VGroup(Line(ORIGIN, 0.5 * RIGHT, color=ELASTIC, stroke_width=4), T("elastic", 18, ELASTIC)).arrange(RIGHT, buff=0.12),
            VGroup(Line(ORIGIN, 0.5 * RIGHT, color=ISO, stroke_width=4), T("isostatic", 18, ISO)).arrange(RIGHT, buff=0.12),
            VGroup(Square(0.22, fill_color=TENSION, fill_opacity=0.8, stroke_width=0), T("tension", 18, INK2)).arrange(RIGHT, buff=0.12),
            VGroup(Square(0.22, fill_color=COMP, fill_opacity=0.8, stroke_width=0), T("compression", 18, INK2)).arrange(RIGHT, buff=0.12),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([5.7, 2.25, 0])
        with voiceover(self, "Here's the model running, for a lake three kilometres wide and four "
                             "metres deep. The orange curve is the instant elastic response: a "
                             "broad dish, with small bulges on either side.") as vo:
            self.play(Create(top), Create(bot), FadeIn(ylab1), FadeIn(ylab2), FadeIn(xlab),
                      FadeIn(ticks), run_time=1.2)
            self.add(surf, strs, clock)
            self.play(Create(el_curve), FadeIn(lg), run_time=1.5)
            self.play(Create(iso_curve), run_time=1.5)
        with voiceover(self, "Over a few Maxwell times, the ice creeps towards the dashed "
                             "isostatic shape, and the bending stresses, shown below, slowly fade "
                             "away. Notice that the sharp edges are the last to relax.") as vo:
            self.play(k_tr.animate.set_value(t.size - 1), run_time=vo.duration,
                      rate_func=lambda u: u ** 1.6)
        self.wait(0.5)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)


# ===========================================================================
class S09Spectrum(Scene):
    def construct(self):
        hd = header(8, "Which shapes relax first?")
        form = M(r"\tau_\lambda = \tau\,\Big[1 + \Big(\frac{2\pi\ell}{\lambda}\Big)^4\Big]", 48)
        form.move_to([0, 2.4, 0])
        ax = Axes(x_range=[0, 12, 2], y_range=[0, 1.05, 0.5], x_length=7.0, y_length=3.6,
                  tips=False, axis_config={"color": MUTED, "stroke_width": 2}).move_to([-2.2, -1.1, 0])
        xl = M(r"t/\tau", 30, INK2).next_to(ax, DOWN, buff=0.15)
        yl = T("amplitude / isostatic", 20, INK2).rotate(PI / 2).next_to(ax, LEFT, buff=0.55)
        ticks = VGroup(*[T(f"{v}", 16, MUTED).next_to(ax.c2p(v, 0), DOWN, buff=0.08)
                         for v in (0, 4, 8, 12)])
        ticks.add(*[T(f"{v:g}", 16, MUTED).next_to(ax.c2p(0, v), LEFT, buff=0.08) for v in (0, 0.5, 1)])
        tt = np.linspace(0, 12, 300)
        ell = SHELF.ell
        curves, labels = VGroup(), VGroup()
        cols = ["#e0ecfb", "#9cc3f5", "#5a9ff0", "#3a7ad8"]
        for r, c in zip([20, 6, 3, 1.5], cols):
            k = 2 * np.pi / (r * ell)
            e = 1 / (1 + (k * ell) ** 4)
            amp = 1 + (e - 1) * np.exp(-tt * SHELF.tau / relaxation_time(k, SHELF))
            cv = curve_on(ax, tt, amp, c, 5)
            curves.add(cv)
            lab = M(rf"\lambda = {r:g}\,\ell", 28, c)
            labels.add(lab)
        labels.arrange(DOWN, aligned_edge=LEFT, buff=0.18).next_to(ax, RIGHT, buff=0.5).shift(0.5 * UP)
        with voiceover(self, "Why do some parts relax faster than others? Split the load into "
                             "waves. Each wavelength jumps elastically, then relaxes "
                             "exponentially, on its own timescale.") as vo:
            self.play(FadeIn(hd), Write(form), run_time=2.0)
            self.play(Create(ax), FadeIn(xl), FadeIn(yl), FadeIn(ticks), run_time=1.0)
            self.play(LaggedStart(*[Create(c) for c in curves], lag_ratio=0.3),
                      LaggedStart(*[FadeIn(l) for l in labels], lag_ratio=0.3),
                      run_time=max(vo.duration - 3.2, 1.0))
        note1 = T("broad: about one Maxwell time", 22, cols[0])
        note2 = T("narrow: months, even years", 22, cols[3])
        notes = VGroup(note1, note2).arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(labels, DOWN, buff=0.5, aligned_edge=LEFT)
        with voiceover(self, "Broad features relax in about one Maxwell time. But narrow "
                             "features, less than a few flexural lengths across, are held up by "
                             "the ice's stiffness, and can take months, or even years.") as vo:
            self.play(Indicate(curves[0], color=cols[0]), FadeIn(note1), run_time=1.6)
            self.play(Indicate(curves[3], color=cols[3]), FadeIn(note2), run_time=1.6)
        with voiceover(self, "Handily, nothing ever relaxes faster than tau itself, so the "
                             "equations aren't stiff, and the numerics can be kept simple.") as vo:
            self.play(Circumscribe(form, color=ACCENT), run_time=1.5)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)


# ===========================================================================
class S10Numerics(Scene):
    def construct(self):
        hd = header(9, "Numerics: fast and simple")
        n = 9
        xs = np.linspace(-4.6, 4.6, n)
        y0 = 1.4
        rod = Line([xs[0], y0, 0], [xs[-1], y0, 0], color=ICE_EDGE, stroke_width=10,
                   stroke_opacity=0.5)
        nodes = VGroup(*[Dot([x, y0, 0], radius=0.11, color=VISCO) for x in xs])
        ghosts = VGroup(*[Circle(radius=0.11, color=MUTED, stroke_width=3).move_to([x, y0, 0])
                          for x in (xs[0] - 1.15, xs[-1] + 1.15)])
        dxb = BraceBetweenPoints([xs[3], y0 - 0.2, 0], [xs[4], y0 - 0.2, 0], DOWN, color=INK2)
        dxt = M(r"\Delta x", 28, INK2).next_to(dxb, DOWN, buff=0.06)
        labs = VGroup(M("w_j,\\ M_j", 28, VISCO).next_to(nodes[4], UP, buff=0.18))
        with voiceover(self, "On to the numerics. Place nodes along the beam, a distance delta x "
                             "apart, and store the deflection and the bending moment at each "
                             "one.") as vo:
            self.play(FadeIn(hd), Create(rod), run_time=0.8)
            self.play(LaggedStart(*[GrowFromCenter(d) for d in nodes], lag_ratio=0.1), run_time=1.5)
            self.play(GrowFromCenter(dxb), Write(dxt), Write(labs), run_time=1.2)
        # operator chain
        y = -0.1
        names = [("deflection", "w", VISCO), ("curvature", r"\kappa", INK2),
                 ("moment", "M", ACCENT), ("net force", "q - k w", ELASTIC)]
        boxes = VGroup()
        for name, sym, c in names:
            g = VGroup(T(name, 22, c), M(sym, 32, c)).arrange(DOWN, buff=0.08)
            boxes.add(VGroup(SurroundingRectangle(g, color=c, buff=0.18, corner_radius=0.1), g))
        boxes.arrange(RIGHT, buff=1.3).move_to([0, y, 0])
        arrows = VGroup(*[Arrow(boxes[i].get_right(), boxes[i + 1].get_left(), buff=0.1,
                                color=INK, stroke_width=4) for i in range(3)])
        alabs = VGroup(M("L", 34).next_to(arrows[0], UP, buff=0.1),
                       T("Maxwell", 20, INK2).next_to(arrows[1], UP, buff=0.12),
                       M("L^*", 34).next_to(arrows[2], UP, buff=0.1))
        stencil = M(r"L,\ L^* \;\sim\; \frac{[\,1,\ -2,\ 1\,]}{\Delta x^2}", 38).move_to([0, -1.6, 0])
        with voiceover(self, "Two small matrices do all the work. L turns deflection into "
                             "curvature, and L star turns moments into net force. Both are the "
                             "familiar one, minus two, one stencil.") as vo:
            self.play(FadeIn(boxes[0]), run_time=0.5)
            for i in range(3):
                self.play(GrowArrow(arrows[i]), FadeIn(alabs[i]), FadeIn(boxes[i + 1]), run_time=0.9)
            self.play(Write(stencil), run_time=1.5)
        clamp = VGroup(Rectangle(width=0.35, height=1.3, color=INK2).move_to([xs[0] - 0.3, y0, 0]))
        clamp.add(*[Line([xs[0] - 0.47, y0 - 0.6 + i * 0.2, 0], [xs[0] - 0.13, y0 - 0.45 + i * 0.2, 0],
                         color=INK2, stroke_width=2) for i in range(6)])
        bc1 = VGroup(T("clamped grounding line", 20, INK2),
                     M(r"w_0 = 0,\ \ w_{-1} = w_1\ \ \rightarrow L", 28)).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        bc1.next_to(clamp, UP, buff=0.25).align_to(clamp, LEFT).shift(LEFT * 0.4)
        bc2 = VGroup(T("free calving front", 20, INK2),
                     M(r"M_N = 0,\ \ M_{N+1} = M_{N-1}\ \ \rightarrow L^*", 28)).arrange(DOWN, aligned_edge=RIGHT, buff=0.08)
        bc2.next_to(ghosts[1], UP, buff=0.3).align_to(ghosts[1], RIGHT)
        with voiceover(self, "They differ only in their end rows, which is where the boundary "
                             "conditions live: a clamped grounding line goes into L, and a free "
                             "calving front goes into L star.") as vo:
            self.play(FadeIn(ghosts), FadeIn(clamp), FadeOut(labs), run_time=1.0)
            self.play(FadeIn(bc1, shift=0.1 * DOWN), run_time=1.2)
            self.play(FadeIn(bc2, shift=0.1 * DOWN), run_time=1.2)
        # time stepping
        step1 = M(r"M^{n+1} = c\,\Big[\big(1-\tfrac{a}{2}\big)M^n + D\,L\,(w^{n+1}-w^n)\Big]", 34, ACCENT)
        step2 = M(r"\big(L^* c D\,L + k\big)\,w^{n+1} = q^{n+1} + L^* cD\,L\,w^n - L^*c\big(1-\tfrac{a}{2}\big)M^n", 34, VISCO)
        defs = M(r"a = \Delta t/\tau,\quad c = 1/(1 + a/2)", 28, INK2)
        steps = VGroup(step1, step2, defs).arrange(DOWN, buff=0.35).move_to([0, 1.6, 0])
        with voiceover(self, "To step forward in time, apply the trapezoidal rule to the Maxwell "
                             "law, and substitute it into the force balance. That leaves a "
                             "single linear system for the new deflection.") as vo:
            self.play(FadeOut(VGroup(boxes, arrows, alabs, stencil, rod, nodes, ghosts, dxb, dxt,
                                     clamp, bc1, bc2)), run_time=0.6)
            self.play(Write(step1), run_time=2.0)
            self.play(Write(step2), FadeIn(defs), run_time=2.5)
        code_lines = [
            "rhs = q1 + S @ w0 - Lstar @ (c * (1 - a/2) * M0)",
            "w1  = lu.solve(rhs)            # one banded solve",
            "M1  = c * ((1 - a/2) * M0 + D * (L @ (w1 - w0)))",
        ]
        code = VGroup(*[Text(s, font="DejaVu Sans Mono", font_size=22, color=INK) for s in code_lines])
        code.arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        cbox = SurroundingRectangle(code, color=MUTED, buff=0.25, corner_radius=0.1,
                                    fill_color="#0a1018", fill_opacity=1)
        cg = VGroup(cbox, code).move_to([0, -1.1, 0])
        speed = T("3000 nodes × 1200 steps ≈ 1 second", 26, ISO).next_to(cg, DOWN, buff=0.4)
        with voiceover(self, "The matrix never changes, so it's factorised once, and each time "
                             "step is just one banded solve. Three thousand nodes and twelve "
                             "hundred time steps take about a second.") as vo:
            self.play(FadeIn(cg), run_time=1.0)
            self.play(LaggedStart(*[AddTextLetterByLetter(c) for c in code], lag_ratio=0.5),
                      run_time=3.0)
            self.play(FadeIn(speed), run_time=0.8)
        img = ImageMobject(os.path.join(HERE, "..", "figures", "verification.png"))
        img.scale_to_fit_width(12.5).move_to([0, -0.4, 0])
        frame = SurroundingRectangle(img, color=MUTED, buff=0.05, corner_radius=0.05)
        with voiceover(self, "And it checks out. It matches the exact elastic solutions and the "
                             "exact Fourier solution, and it converges at second order in both "
                             "space and time.") as vo:
            self.play(FadeOut(VGroup(steps, cg, speed)), run_time=0.6)
            self.play(FadeIn(img), Create(frame), run_time=1.2)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)


# ===========================================================================
class S11FillDrain(Scene):
    def construct(self):
        hd = header(10, "A lake fills, sits, and drains")
        x = np.linspace(-15e3, 15e3, 1501)
        beam, d, q, w_el, _ = lake_solution(x)
        tf, th, tdr = 10 * DAY, 15 * DAY, 0.25 * DAY
        frac = lambda tt: fill_hold_drain(tt, tf, th, tdr)
        t = np.unique(np.r_[np.linspace(0, 45 * DAY, 901), np.linspace(25 * DAY, 25.25 * DAY, 13)])
        res = beam.run(t, lambda tt: frac(tt) * q)
        fr = frac(t)
        td = t / DAY
        sel = np.abs(x) <= 6e3
        xk = x[sel] / 1e3

        top = plot_axes([-6, 6, 2], [-4.5, 2.5, 1], 11.0, 2.5)
        bot = plot_axes([-6, 6, 2], [-400, 400, 200], 11.0, 1.7)
        tl = plot_axes([0, 45, 5], [0, 4.5, 2], 11.0, 0.75)
        VGroup(top, bot, tl).arrange(DOWN, buff=0.45).move_to([0.55, -0.6, 0])
        lab = VGroup(T("surface (m)", 17, INK2).rotate(PI / 2).next_to(top, LEFT, buff=0.45),
                     T("stress (kPa)", 17, INK2).rotate(PI / 2).next_to(bot, LEFT, buff=0.6),
                     T("lake", 17, INK2).next_to(tl, LEFT, buff=0.45),
                     T("distance (km)", 17, INK2).next_to(bot.c2p(6, -400), DOWN, buff=0.08).align_to(bot, RIGHT),
                     T("time (days)", 17, INK2).next_to(tl.c2p(45, 0), DOWN, buff=0.08).align_to(tl, RIGHT))
        ticks = VGroup(*[T(f"{v}", 15, MUTED).next_to(tl.c2p(v, 0), DOWN, buff=0.06)
                         for v in (0, 10, 20, 30)])
        ticks.add(*[T(f"{v:g}", 15, MUTED).next_to(bot.c2p(v, -400), DOWN, buff=0.06)
                    for v in (-4, -2, 0, 2, 4)])
        ticks.add(*[T(f"{v:g}", 15, MUTED).next_to(top.c2p(-6, v), LEFT, buff=0.06) for v in (-4, -2, 0, 2)])
        ticks.add(*[T(f"{v:g}", 15, MUTED).next_to(bot.c2p(-6, v), LEFT, buff=0.06)
                    for v in (-400, 0, 400)])
        zero = DashedLine(bot.c2p(-6, 0), bot.c2p(6, 0), color=MUTED, stroke_width=1.5)
        lake_hist = VGroup(fill_on(tl, td, np.zeros_like(td), 4 * fr, WATER, 0.5),
                           curve_on(tl, td, 4 * fr, WATER, 3))
        day = ValueTracker(0.0)

        def i_now():
            return int(np.clip(np.searchsorted(td, day.get_value()), 0, td.size - 1))

        def surface():
            i = i_now()
            s = -res.w[i][sel]
            dep = fr[i] * d[sel]
            g = VGroup(fill_on(top, xk, np.full_like(xk, -4.5), s, ICE, 0.12))
            g.add(water_on(top, xk, s, dep))
            g.add(curve_on(top, xk, s, INK, 4))
            return g

        def stress():
            i = i_now()
            sg = np.clip(res.sigma[i][sel] / 1e3, -400, 400)
            return VGroup(fill_on(bot, xk, np.zeros_like(xk), np.clip(sg, 0, None), TENSION, 0.75),
                          fill_on(bot, xk, np.clip(sg, None, 0), np.zeros_like(xk), COMP, 0.75),
                          curve_on(bot, xk, sg, INK, 2))

        marker = always_redraw(lambda: VGroup(
            Line(tl.c2p(day.get_value(), 0), tl.c2p(day.get_value(), 4.5), color=INK, stroke_width=3),
            Dot(tl.c2p(day.get_value(), 4 * float(frac(day.get_value() * DAY))), color=INK, radius=0.06)))
        clock = always_redraw(lambda: T(f"day {day.get_value():4.1f}", 24, INK).move_to(top.c2p(4.8, 1.7)))
        surf, strs = always_redraw(surface), always_redraw(stress)
        legend = VGroup(
            VGroup(Square(0.2, fill_color=TENSION, fill_opacity=0.8, stroke_width=0), T("tension", 17, INK2)).arrange(RIGHT, buff=0.1),
            VGroup(Square(0.2, fill_color=COMP, fill_opacity=0.8, stroke_width=0), T("compression", 17, INK2)).arrange(RIGHT, buff=0.1),
        ).arrange(RIGHT, buff=0.3).move_to(bot.c2p(3.8, 330))
        with voiceover(self, "Now let's use it. A lake fills over ten days, sits for fifteen, "
                             "then drains in just six hours, perhaps through a new fracture.") as vo:
            self.play(FadeIn(hd), Create(top), Create(bot), Create(tl), FadeIn(lab), FadeIn(ticks),
                      Create(zero), run_time=1.5)
            self.play(FadeIn(lake_hist), FadeIn(legend), run_time=1.2)
            self.add(surf, strs, marker, clock)
        with voiceover(self, "As the lake fills, the ice sags, and a ring of tension, of one to "
                             "two hundred kilopascals, forms just outside the shoreline. That's "
                             "enough to open crevasses.") as vo:
            self.play(day.animate.set_value(10.0), run_time=vo.duration * 0.8, rate_func=linear)
        with voiceover(self, "While the lake sits, the stresses relax. The ice gets used to being "
                             "bent.") as vo:
            self.play(day.animate.set_value(24.9), run_time=vo.duration, rate_func=linear)
        with voiceover(self, "Then the lake drains. The ice springs back elastically, but it's now "
                             "bent the other way: the surface beneath the old lake is pulled into "
                             "tension, of over three hundred kilopascals, and a dip lingers for "
                             "weeks.") as vo:
            self.play(day.animate.set_value(25.3), run_time=1.5, rate_func=smooth)
            self.wait(vo.duration * 0.35)
            self.play(day.animate.set_value(45.0), run_time=vo.duration * 0.45, rate_func=linear)
        with voiceover(self, "So filling isn't the only danger. Every fill and drain cycle flips "
                             "the stress, like bending a paperclip back and forth.") as vo:
            self.play(day.animate.set_value(25.3), run_time=1.0)
            self.play(Circumscribe(bot, color=TENSION), run_time=2.0)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)


# ===========================================================================
class S12Summary(Scene):
    def construct(self):
        hd = header(11, "Summary")
        rows = VGroup(
            VGroup(T("buoyancy", 26, ISO), T("a bed of springs", 24, INK2), M(r"k = \rho_{sw}g", 34, ISO)),
            VGroup(T("bending", 26, ELASTIC), T("an elastic beam", 24, INK2), M(r"M = D\,\kappa", 34, ELASTIC)),
            VGroup(T("creep", 26, VISCO), T("a Maxwell material", 24, INK2), M(r"\dot\kappa = \dot M/D + M/\Lambda", 34, VISCO)),
        )
        for i, r in enumerate(rows):
            yy = 2.5 - 0.75 * i
            r[0].move_to([0, yy, 0]).align_to([-5.6, 0, 0], LEFT)
            r[1].move_to([0, yy, 0]).align_to([-3.2, 0, 0], LEFT)
            r[2].move_to([0, yy, 0]).align_to([0.9, 0, 0], LEFT)
        master = M(r"\Big(D\,\frac{\partial^4}{\partial x^4} + k\Big)\frac{\partial w}{\partial t}"
                   r" = \frac{\partial q}{\partial t} + \frac{q - k\,w}{\tau}", 50)
        master.move_to([0, -0.9, 0])
        box = SurroundingRectangle(master, color=ACCENT, buff=0.25, corner_radius=0.12)
        with voiceover(self, "To recap. Buoyancy acts like a bed of springs, ice bends like a "
                             "beam, and a Maxwell material lets it flow.") as vo:
            self.play(FadeIn(hd), run_time=0.5)
            for r in rows:
                self.play(FadeIn(r, shift=0.15 * RIGHT), run_time=1.1)
        sub = T("elastic at first  ·  isostatic at last  ·  narrow features relax slowest", 24, INK2)
        sub.next_to(box, DOWN, buff=0.35)
        with voiceover(self, "Together they give one master equation: elastic at first, "
                             "isostatic in the long run, with narrow features relaxing slowest.") as vo:
            self.play(Write(master), Create(box), run_time=2.5)
            self.play(FadeIn(sub), run_time=1.0)
        where = T("code · derivation · tests:   BFRN_meltwater / flexure", 26, ACCENT)
        where.to_edge(DOWN, buff=0.6)
        with voiceover(self, "The code, the full derivation with diagrams, and the tests are all "
                             "in the flexure folder of this repository. Thanks for watching.") as vo:
            self.play(FadeIn(where, shift=0.2 * UP), run_time=1.2)
        self.wait(1.0)
        self.play(FadeOut(Group(*self.mobjects)), run_time=1.0)


SCENES = ["S01Intro", "S02Setup", "S03Buoyancy", "S04ForceBalance", "S05Bending",
          "S06Maxwell", "S07Master", "S08Limits", "S09Spectrum", "S10Numerics",
          "S11FillDrain", "S12Summary"]
