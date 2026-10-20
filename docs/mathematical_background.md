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

## Floating-Point Precision Limitations
This CPU implementation uses standard 64-bit IEEE-754 floats (`float64`). At extreme magnifications (deep zooms beyond $\approx 10^{-13}$), floating-point precision fails, leading to blocky "pixelation". Resolving this requires arbitrary-precision arithmetic, which is outside the scope of v1.0.0.
