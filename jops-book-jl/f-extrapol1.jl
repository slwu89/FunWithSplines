using Distributions
using DataFrames
using Plots
include("./jops.jl")

# Simulate data
m = 50
x = sort(rand(m))
y = sin.(2.5 .* x) .+ rand(Normal(), m) .* 0.05 .+ 0.2
f = findall(0.2 .< x .&& x .< 0.4 .|| 0.6 .< x .&& x .< 0.8)
x = x[f]
y = y[f]
Data = DataFrame(x=x,y=y)

# Make a matrix containing the B-spline basis
ndx = 25
deg = 3
B = bbase(x, 0, 1, ndx, deg)
nb = size(B,2)

# Basis for fit on grid
ng = 500
xg = range(0, 1, ng)
Bg = bbase(xg, 0, 1, ndx, deg)

# Fit
D = diffmat(nb, 1)
P = transpose(D) * D
lambda = 1
a = (transpose(B) * B .+ lambda .* P) \ (transpose(B) * y)
z = Bg * a

knots = ((1:nb) .- 2)/ndx |> collect
Fa = DataFrame(x = knots, y = a)

# Create data frame for plots
Zf = DataFrame(x = xg, y = z)

plt1 = plot(
    Zf.x, Zf.y, linewidth=0.6, color="blue", title="First differences", legend=false
)
scatter!(plt1, Data.x, Data.y, color="darkgrey", markeralpha=0.8)
scatter!(plt1, Fa.x, Fa.y, color="red")

# Fit
D = diffmat(nb, 2)
P = transpose(D) * D
lambda = 1
a = (transpose(B) * B .+ lambda .* P) \ (transpose(B) * y)
z = Bg *  a

Zf = DataFrame(x = xg, y = z)
Fa = DataFrame(x = knots, y = a)

plt2 = plot(
    Zf.x, Zf.y, linewidth=0.6, color="blue", title="Second differences", legend=false
)
scatter!(plt2, Data.x, Data.y, color="darkgrey", markeralpha=0.8)
scatter!(plt2, Fa.x, Fa.y, color="red")

plot(plt1, plt2, layout = 2)