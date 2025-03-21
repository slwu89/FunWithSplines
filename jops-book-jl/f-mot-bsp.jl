using RDatasets, DataFrames
using BSplines

mcycle = dataset("MASS", "mcycle")
x = mcycle.Times
y = mcycle.Accel

# the B-spline basis
deg = 3
xlo = minimum(x)
xhi = maximum(x)
ndx = 5

dx = (xhi - xlo)/ndx
knots = collect(range(start = xlo - deg * dx, step = dx, stop = xhi + deg * dx))

B = BSplineBasis(deg+1, knots)
# Spline(B, ones(14))