using RDatasets, DataFrames
include("./jops.jl")

mcycle = dataset("MASS", "mcycle")

x = mcycle[!, :Times]
y = mcycle[!, :Accel]

xl = minimum(x)
xr = maximum(x)

nseg = 20
bdeg = 3
pord = 2
lambda = 0.8

xgrid = 100

m = length(x)
B = bbase(x, xl, xr, nseg, bdeg)

# why is bbase not the right size

# compute breakpoints
dx = (xr - xl)/nseg
knots = collect(range(start = xl - bdeg * dx, step = dx, stop = xr + bdeg * dx))

((xr + bdeg * dx) - (xl - bdeg * dx)) / dx

# BSplineBasis duplicates low/upper breakpoints such that they appear k-1 times
k = bdeg + 1
bbasis = BSplines.BSplineBasis(k, knots)

# the basis matrix evaluated at points x
B = basismatrix(bbasis, x)
B = B[:, bdeg+1:end-bdeg]


# # Construct penalty stuff
# n = size(B)[2]
# P <- sqrt(lambda) * diff(diag(n), diff = pord)
# nix <- rep(0, n - pord)
