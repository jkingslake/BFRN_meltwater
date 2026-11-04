# Narration script

Generated from the voice-over lines in `explainer.py`.


## Intro

> Every summer, meltwater pools into lakes on the surface of Antarctica's floating ice shelves.

> Those lakes are heavy, and the ice bends beneath them. Bend it far enough, or back and forth often enough, and it can crack.

> In this video we'll build a simple model of that bending: a two-dimensional viscoelastic beam. We'll derive it step by step, and then solve it on a computer in a few lines of code.


## Setup

> Take a vertical slice through an ice shelf, from the grounding line, where the ice lifts off the bed, out to the calving front, where it meets the open ocean.

> The ice is a few hundred metres thick, and it floats, with roughly nine tenths of it below sea level.

> Now add a lake of depth d. Its weight presses down on the ice with a load q, equal to rho w, g, d. The ice is pushed down by a distance w, which we'll measure positive downwards.

> Our goal is to predict w, as a function of position x and time t, and the stresses that come with it.

> We'll assume the ice is a thin beam, the deflections are small, nothing varies across the page, and the ocean responds hydrostatically.


## Buoyancy

> First ingredient: buoyancy. A floating column of ice sinks until the sea water it pushes aside weighs as much as the ice.

> Put a lake on top, and the column is pushed down by w. It displaces extra sea water, so the ocean pushes back with an extra pressure: rho s w, g, w.

> That's exactly like resting the ice on a bed of springs, with stiffness k equal to rho s w, g: about ten thousand pascals per metre.

> If the ice had no stiffness at all, each column would simply sink until k w equals q, and the dip would copy the lake exactly. That's called local isostasy.


## ForceBalance

> But ice is stiff, so each column drags on its neighbours. Cut out a thin slice of width d x.

> The lake pushes down, buoyancy pushes up, and the ice on either side exerts shear forces, V, and bending moments, M.

> Balancing vertical forces gives: d V by d x equals q minus k w. Balancing moments gives: d M by d x equals V.

> Put them together, and we have our first key equation: the second derivative of M, plus k w, equals q. Wherever buoyancy isn't holding up the lake, bending moments must.


## Bending

> Next: how does the bending moment depend on the shape? When a beam bends, its cross-sections stay flat, but they tilt.

> So a fibre a height z above the middle is stretched by z times kappa, where kappa, the curvature, is the second derivative of w.

> For an elastic material, stress is proportional to strain. Add up the turning effect of every fibre, and you get: M equals D kappa.

> The flexural rigidity, D, goes as the thickness cubed, so doubling the thickness makes the shelf eight times stiffer.

> For a viscous fluid, stress depends instead on how fast the strain changes. The same sum gives: M equals capital lambda, times the rate of change of kappa, where lambda depends on the viscosity, eta.


## Maxwell

> Real ice is both. Over hours it behaves like a spring, but over weeks it flows. The simplest material that does both is a Maxwell element: a spring and a dashpot, joined in series.

> Their rates of bending simply add.

> Hold the moment fixed, and the curvature jumps, then creeps steadily. Hold the curvature fixed, and the moment relaxes away exponentially.

> The timescale is the Maxwell time, tau, equal to capital lambda over D. For typical ice-shelf values, tau is about four days.


## Master

> Now we combine the two. Take two x derivatives of the Maxwell law, and use the force balance to replace the moments.

> Everything collapses into a single master equation for the deflection.

> Read it like this. The bracket on the left is just an elastic beam, resting on springs.

> The first term on the right says that any change in the load gets an instant, elastic response.

> And the second says that, wherever buoyancy isn't yet carrying the lake, the ice keeps creeping, at a rate set by tau, until k w equals q.


## Limits

> That gives two limits. At the instant the lake appears, the response is purely elastic. Its natural length scale is the flexural length, ell: the fourth root of D over k, about six hundred metres here.

> A long time later, the ice relaxes to local isostasy, and the dip simply copies the lake.

> Here's the model running, for a lake three kilometres wide and four metres deep. The orange curve is the instant elastic response: a broad dish, with small bulges on either side.

> Over a few Maxwell times, the ice creeps towards the dashed isostatic shape, and the bending stresses, shown below, slowly fade away. Notice that the sharp edges are the last to relax.


## Spectrum

> Why do some parts relax faster than others? Split the load into waves. Each wavelength jumps elastically, then relaxes exponentially, on its own timescale.

> Broad features relax in about one Maxwell time. But narrow features, less than a few flexural lengths across, are held up by the ice's stiffness, and can take months, or even years.

> Handily, nothing ever relaxes faster than tau itself, so the equations aren't stiff, and the numerics can be kept simple.


## Numerics

> On to the numerics. Place nodes along the beam, a distance delta x apart, and store the deflection and the bending moment at each one.

> Two small matrices do all the work. L turns deflection into curvature, and L star turns moments into net force. Both are the familiar one, minus two, one stencil.

> They differ only in their end rows, which is where the boundary conditions live: a clamped grounding line goes into L, and a free calving front goes into L star.

> To step forward in time, apply the trapezoidal rule to the Maxwell law, and substitute it into the force balance. That leaves a single linear system for the new deflection.

> The matrix never changes, so it's factorised once, and each time step is just one banded solve. Three thousand nodes and twelve hundred time steps take about a second.

> And it checks out. It matches the exact elastic solutions and the exact Fourier solution, and it converges at second order in both space and time.


## FillDrain

> Now let's use it. A lake fills over ten days, sits for fifteen, then drains in just six hours, perhaps through a new fracture.

> As the lake fills, the ice sags, and a ring of tension, of one to two hundred kilopascals, forms just outside the shoreline. That's enough to open crevasses.

> While the lake sits, the stresses relax. The ice gets used to being bent.

> Then the lake drains. The ice springs back elastically, but it's now bent the other way: the surface beneath the old lake is pulled into tension, of over three hundred kilopascals, and a dip lingers for weeks.

> So filling isn't the only danger. Every fill and drain cycle flips the stress, like bending a paperclip back and forth.


## Summary

> To recap. Buoyancy acts like a bed of springs, ice bends like a beam, and a Maxwell material lets it flow.

> Together they give one master equation: elastic at first, isostatic in the long run, with narrow features relaxing slowest.

> The code, the full derivation with diagrams, and the tests are all in the flexure folder of this repository. Thanks for watching.
