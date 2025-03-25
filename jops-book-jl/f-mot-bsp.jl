using RDatasets, DataFrames
using BSplines
using LinearAlgebra
using Plots

include("./jops.jl")

mcycle = dataset("MASS", "mcycle")
x = mcycle.Times
y = mcycle.Accel

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

# Create data frames for plotting
Zf1 = DataFrame(x = xg, y = z1)
Zf2 = DataFrame(x = xg[xg .> thr], y = zsel[xg .> thr])

p1 = scatter(
    x, y, legend=false, color="black",
    xlabel="Time (ms)", ylabel="Acceleration (g)",
    title="Motorcycle helmet impact data"  
)
plot!(p1, Zf1.x, Zf1.y, legend=false, linecolor="blue")
plot!(p1, Zf2.x, Zf2.y, legend=false, linecolor="red", linestyle=:dash)