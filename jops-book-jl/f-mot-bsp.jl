using RDatasets, DataFrames
using BSplines
using LinearAlgebra
using Plots

mcycle = dataset("MASS", "mcycle")
x = mcycle.Times
y = mcycle.Accel

# check that this works for bdeg != 3 (for ex, 2),
# check against R

# in R we give to bbase :
#   x: vector of points to evaluate the spline
#   xl: lower support, usu min(x)
#   xr: upper support, usu max(x)
#   nseg: number of segments on support
#   bdeg: degree of splines

# for Julia we would:
#   1. from the xl, xr, bdeg, nseg args compute the vector of knots (actually breakpoints,
#      knot are computed breakpoints plus duplicating first/last points k, order times).
#   2. make the BSplineBasis from the knots and degree
#   3. from x and the basis, make the basis matrix B
function bbase(x, xl, xr, nseg = 10, bdeg = 3)
    # compute breakpoints
    dx = (xr - xl)/nseg
    knots = collect(range(start = xl - bdeg * dx, step = dx, stop = xr + bdeg * dx))

    k = bdeg + 1
    bbasis = BSplines.BSplineBasis(k, knots)

    # the basis matrix evaluated at points x
    B = basismatrix(bbasis, x)
    B = B[:, bdeg+1:end-bdeg]
    return B
end

# Boundary for the subdomain
thr = 5
sel = x .> thr
xsel = x[sel]
ysel = y[sel]

# the B-spline basis
deg = 3
xlo = minimum(x)
xhi = maximum(x)
ndx = 5

B = bbase(x, xlo, xhi, ndx, deg)

# Basis for fit on grid
ng = 1000
xg = range(minimum(x), maximum(x), ng) |> collect
Bg = bbase(xg, xlo, xhi, ndx, deg)

# Use 0/1 weight to select the subdomain
W = 1 .* Diagonal(x .> thr)

# Estimate the coefficients and compute fit on the grid
a = (transpose(B) * B) \ (transpose(B) * y)
z1 = Bg * a

asel = (transpose(B) * W * B) \ (transpose(B) * W * y)
zsel = Bg * asel

# Create data frames for ggplot
Zf1 = DataFrame(x = xg, y = z1)
Zf2 = DataFrame(x = xg[xg .> thr], y = zsel[xg .> thr])

p1 = scatter(
    x, y, legend=false, color="black",
    xlabel="Time (ms)", ylabel="Acceleration (g)",
    title="Motorcycle helmet impact data"  
)
plot!(p1, Zf1.x, Zf1.y, legend=false, linecolor="blue")
plot!(p1, Zf2.x, Zf2.y, legend=false, linecolor="red", linestyle=:dash)


p1 = scatter(
    Dat.x, Dat.y, legend=false, color="black",
    xlabel="Wind speed (mph)", ylabel="Ozone concentration (ppb)",
    title="New York air quality"
)
plot!(p1, DatSort.x, predict(linear_mod, DatSort), legend=false, linestyle=:dash, linecolor="blue")
plot!(p1, DatSort.x, predict(poly_mod, DatSort), legend=false, linecolor="red")