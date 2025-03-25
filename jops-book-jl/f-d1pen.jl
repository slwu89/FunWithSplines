using Distributions
using Plots
using DataFrames
include("./jops.jl")

# Simulate data
m = 50
x = sort(rand(m))
y = sin.(2.5 .* x) .+ rand(Normal(), m) .* 0.1 .+ 0.2

# Make basis and penalty
nu = 200
u = range(0, 1, nu)
nseg = 20
Bu = bbase(u, minimum(u), maximum(u), nseg, 3)
B = bbase(x, minimum(x), maximum(x), nseg, 3)
nb = size(B,2)
knots = collect(((1:nb) .- 2) ./ nseg)
n = size(B,2)
D = diffmat(n, 1)
P = transpose(D) * D
BtB = transpose(B) * B
Bty = transpose(B) * y

# Compute coefficients (A), residuals (Mus), fitted splines (Z)
lambdas = [0.1, 1, 10, 100] .* 2
A = Vector{Vector{Float64}}(undef, length(lambdas))
Mu = Vector{Vector{Float64}}(undef, length(lambdas))
Z = Vector{Vector{Float64}}(undef, length(lambdas))
for (i, lambda) in enumerate(lambdas)
    a = (BtB .+ lambda * P) \ Bty
    A[i] = a
    Z[i] = Bu * a
    Mu[i] = B * a
end

# generate the plots
plts = Vector{Any}(undef, length(lambdas))
for j in eachindex(lambdas)
    # Compute roughness and std dev of residuals
    aj = A[j]
    da = diff(aj)
    r = sqrt(sum(da .^ 2) ./ (n - 1))
    r = round(r, digits=2)
    s = sqrt(mean((y .- Mu[j]).^2))
    s = round(s, digits=2)

    # scaled basis
    Bsc = B * Diagonal(aj)
    # Create data frames for plots
    Zf = DataFrame(x = u, y = Z[j])
    titl = "λ=$(lambdas[j]) | s=$(s) | r=$(r)"

    Fa = DataFrame(x = knots, y = aj)

    # build the graphs
    plt1 = scatter(
        x, y, legend=false, color="black",
        title=titl, titlefont=font(10)
    )
    plot!(plt1, Zf.x, Zf.y, legend=false, linecolor="blue", linewidth=3, linealpha=0.8)
    scatter!(plt1, Fa.x, Fa.y, legend=false, color="red", markeralpha=0.5)
    plts[j] = plt1
end

plot(plts..., layout = 4)