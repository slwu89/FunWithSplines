using Turing
using Distributions
using DataFrames
using Plots

include("./jops.jl")

n = 100
x = collect(range(0, 1, length=n))
f_true = @. (sin(2*π*x^3))^3
y = f_true + rand(Normal(0, 0.2), n)

p1 = scatter(x, y, legend=false, markeralpha=0.5, color="black")
plot!(p1, x, f_true, legend=false, color="black", linestyle=:dash)

B = bbase(x, 0, 1, 20, 3)

# these observations are equally spaced