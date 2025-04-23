# urchins model: random effects, MLE by direct Laplace approximation
using DataFrames, CSV
using Distributions
using LinearAlgebra

using Optim
using LineSearches
using Plots, StatsPlots
using DifferentiationInterface
import ForwardDiff

# for sparse AD
using SparseConnectivityTracer
using SparseMatrixColorings

ad_sys = AutoForwardDiff()

# load the urchin data
const urchin = CSV.read("./MLE/urchin.csv", DataFrame)
select!(urchin, Not(:id))
urchin.age = convert(Vector{Float64}, urchin.age)

# index into a single vector for b (random effects)
const log_g_ix = 1:nrow(urchin)
const log_p_ix = range(start=nrow(urchin)+1, length=nrow(urchin))

# index into a single vector for θ (fixed effects)
const log_ω_ix = 1
const μ_g_ix = 2
const log_σ_g_ix = 3
const μ_p_ix = 4
const log_σ_p_ix = 5
const log_σ_ix = 6

"""
The biological model for a single urchin; `g` and `p`
    are the individual level random effects
"""
function model_urchin_vol(ω, g, p, a)
    aₘ = log(p / (g*ω))/g
    return a < aₘ ? ω*exp(g*a) : p/g + p*(a-aₘ)
end

"""
Negative log density (likelihood) of y and b hat
"""
function nlyfb_urchin(b, θ, urchin)
    # extract fixed effects
    σ = exp(θ[log_σ_ix])
    σ_g = exp(θ[log_σ_g_ix])
    σ_p = exp(θ[log_σ_p_ix])
    ω = exp(θ[log_ω_ix])
    # extract random effects
    log_g = b[log_g_ix]
    log_p = b[log_p_ix]
    # calculate the loglikelihood of y and b
    logll = 0.0
    for i in axes(urchin,1)
        v = model_urchin_vol(ω, exp(log_g[i]), exp(log_p[i]), urchin[i, :age])
        logll += logpdf(Normal(sqrt(v), σ), sqrt(urchin[i, :vol]))
        logll += logpdf(Normal(θ[μ_g_ix], σ_g), log_g[i])
        logll += logpdf(Normal(θ[μ_p_ix], σ_p), log_p[i])
    end
    return -logll
end

θ_init = [
    -4.0,
    -0.2, 
    log(0.1),
    0.2,
    log(0.1),
    log(0.5)
]

b_init = [
    fill(θ_init[μ_g_ix], nrow(urchin)); fill(θ_init[μ_p_ix], nrow(urchin))
]

prep_g_urchin = prepare_gradient(nlyfb_urchin, ad_sys, zero(b_init), Constant(θ_init), Constant(urchin))
g_urchin!(G, b) = gradient!(nlyfb_urchin, G, prep_g_urchin, ad_sys, b, Constant(θ_init), Constant(urchin))

# test optimizing for b
result_urchin = optimize(
    b -> nlyfb_urchin(b, θ_init, urchin), 
    g_urchin!,
    b_init, 
    LBFGS(;alphaguess=InitialStatic(scaled=true), linesearch=BackTracking())
)

b_hat = Optim.minimizer(result_urchin)
Optim.minimum(result_urchin)

# --------------------------------------------------
# Hessian

# 1. dense
hessian(nlyfb_urchin, ad_sys, b_hat, Constant(θ_init), Constant(urchin))

# 2. sparse w/TracerLocalSparsityDetector
# the Hessian is very sparse, lets see if we can use a sparse-aware AD method
sparse_ad_sys = AutoSparse(
    ad_sys;
    sparsity_detector=TracerLocalSparsityDetector(),
    coloring_algorithm=GreedyColoringAlgorithm(),
)

hessian(nlyfb_urchin, sparse_ad_sys, b_hat, Constant(θ_init), Constant(urchin))

# 3. sparse preparation: nice & fast
prep_h_nlyfb = prepare_hessian(nlyfb_urchin, sparse_ad_sys, zero(b_init), Constant(θ_init), Constant(urchin))
hessian(nlyfb_urchin, prep_h_nlyfb, sparse_ad_sys, b_hat, Constant(θ_init), Constant(urchin))