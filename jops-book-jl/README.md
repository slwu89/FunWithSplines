# What the hell is going on in here?

The files:
    * `jops.jl`: contains source code, mostly translated from the `JOPS` R package for P-Splines in Julia.
    * `f-air-wind.jl`: From Fig 2.1 in Section 2.1 of the JOPS book, basically nothing noteworthy here.
    * `f-mot-bsp.jl`: From Fig 2.5 in Section 2.2 of the book, fits cubic B-Splines to some data, using the `bbase` function.
    * `f-d1pen.jl`: From Fig 2.9 in Section 2.3 of the book, shows P-Splines at different smoothing values of lambda.
    * `f-extrapol1.jl`: From Fig 2.11 in Section 2.4 of the book, shows how the P-Splines handle interpolation and extrapolation nicely.
    * `f-se.jl`: From Fig 2.14 in Section 2.7 of the book, uses `psNormal` to show SE on a fit to the motorcycle data.