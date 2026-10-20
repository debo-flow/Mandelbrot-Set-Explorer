# Mathematical Background

## The Complex Plane
The Mandelbrot Set exists in the complex plane ($\mathbb{C}$). A complex number is written as $c = a + bi$, where $a$ is the real part, $b$ is the imaginary part, and $i = \sqrt{-1}$. We map these to our coordinate boundaries: Real Axis (X) and Imaginary Axis (Y).

## The Iteration
The set is defined by iterating the quadratic polynomial:
$$z_{n+1} = z_n^2 + c$$
Starting with $z_0 = 0$. 

## Boundedness and the Escape Radius
The Mandelbrot Set is the set of all $c$ for which the sequence $\{z_n\}$ does *not* escape to infinity. Mathematically, if the magnitude $\vert{}z_n\vert{} > 2$ at any point, it is guaranteed that the sequence will diverge to infinity. Therefore, an **escape radius of 2** is sufficient to conclusively prove a point is *not* in the Mandelbrot Set.

## Escaped vs Unresolved Points
Because computers cannot calculate infinity, we compute up to a finite `max_iter`. 
- **Escaped Points:** Points where $\vert{}z_n\vert{} > 2$ within `max_iter`. We record *when* they escaped to color them.
- **Unresolved Points:** Points that remain bounded ($\vert{}z_n\vert{} \le 2$) after `max_iter`. We mark them as unresolved (typically colored black). It is crucial to note that finite computation *cannot mathematically guarantee* membership in the Mandelbrot Set; it merely proves that a point hasn't escaped *yet*.

## Aspect Ratio Correctness
A critical requirement of scientific visualization is preventing spatial distortion. If the requested pixel dimensions have a ratio of $H_{px} / W_{px}$, the complex plane interval must strictly obey:
$$\frac{Im_{max} - Im_{min}}{Re_{max} - Re_{min}} = \frac{H_{px}}{W_{px}}$$
The application rigorously enforces this *before* computation by holding the real-axis width constant and adjusting the imaginary-axis height to perfectly match the pixel grid.

## Continuous Escape Math (Smooth Coloring)
Standard integer iteration creates distinct visible "bands" of color where the iteration count increments. To achieve smooth, continuous gradients, we approximate the fractional iteration count a point "would have taken" to reach infinity. 

For the iteration $z_{n+1} = z_n^2 + c$, the relationship between successive magnitudes approaches $\vert{}z_{n+1}\vert{} \approx \vert{}z_n\vert{}^2$ as $z$ gets large. Utilizing the mathematically established potential function, the continuous escape value $\mu$ is:
$$\mu = n + 1 - \frac{\ln(\ln(\vert{}z_n\vert{}))}{\ln(2)}$$

The term $\ln(\vert{}z_n\vert{})$ is strictly positive and mathematically safe because the application guarantees the escape logic only triggers when $\vert{}z_n\vert{} > 2$. Consequently, $\ln(\vert{}z_n\vert{}) > \ln(2) \approx 0.693 > 0$, preventing both `log(negative)` and `log(0)` domain errors.

## Logarithmic Visibility Mode
In regions of extreme chaos, linear smooth escape times can span vast numerical ranges too quickly for a standard linear colormap. The Logarithmic mode safely applies $\ln(1 + \mu)$ to compress output dynamics, visually surfacing deep structural details near the boundary.

## Floating-Point Precision Limitations
This CPU implementation uses standard 64-bit IEEE-754 floats (`float64`). At extreme magnifications (deep zooms beyond $\approx 10^{-13}$), floating-point precision fails, leading to blocky "pixelation". Resolving this requires arbitrary-precision arithmetic, which is outside the scope of v3.0.0.
