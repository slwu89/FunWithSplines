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
using SparseMatrixColorings
import Symbolics

const ad_sys = AutoForwardDiff()

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
    # return a < aₘ ? ω*exp(g*a) : p/g + p*(a-aₘ)
    return ifelse(a < aₘ, ω*exp(g*a), p/g + p*(a-aₘ))
end

# lyfb in the R code, but need to do the grad/hessian seperately
"""
Log density (likelihood) of y and b hat, \\log{f(y,b)}
"""
function lfyb_urchin(b, θ, urchin)
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
    return logll
end




# tests
# th = zeros(6)
th = [
    -3.0,
    -0.3, 
    -1.5,
    0.15,
    -1.5,
    -1.37
]
n = nrow(urchin)
b = [fill(th[2],n); fill(th[4],n)]

prep_g_lfyb = prepare_gradient(lfyb_urchin, ad_sys, zero(b), Constant(th), Constant(urchin))

sp_ad_sys = AutoSparse(
    ad_sys;
    sparsity_detector=Symbolics.SymbolicsSparsityDetector(),
    coloring_algorithm=GreedyColoringAlgorithm(),
)
prep_sp_h_lfyb = prepare_hessian(lfyb_urchin, sp_ad_sys, rand(length(b)), Constant(th), Constant(urchin))

# log density and grad/hess
lfyb_urchin(b, th, urchin)
gradient(lfyb_urchin, prep_g_lfyb, ad_sys, b, Constant(th), Constant(urchin))
hessian(lfyb_urchin, prep_sp_h_lfyb, sp_ad_sys, b, Constant(th), Constant(urchin))    

# lfybs and grad/hess
## evaluate s log f(y,b;th) + log f(y,b;thp)
function lfybs_urchin(b, θ, θ′, s, urchin)
    lf = lfyb_urchin(b, θ′, urchin)
    if s != 0.0
        lfs = lfyb_urchin(b, θ, urchin)
        lf += s * lfs
    end
    return lf
end

thp = copy(th)
th = thp + (rand(6)*0.1)
lfybs_urchin(b, th, thp, 0.12, urchin)

prep_g_lfybs = prepare_gradient(lfybs_urchin, ad_sys, rand(length(b)), Constant(th), Constant(thp), Constant(0.0), Constant(urchin))
gradient(lfybs_urchin, prep_g_lfybs, ad_sys, b, Constant(th), Constant(thp), Constant(0.12), Constant(urchin))

prep_sp_h_lfybs = prepare_hessian(lfybs_urchin, sp_ad_sys, rand(length(b)), Constant(th), Constant(thp), Constant(0.0), Constant(urchin))
hessian(lfybs_urchin, prep_sp_h_lfybs, sp_ad_sys, b, Constant(th), Constant(thp), Constant(0.12), Constant(urchin))


# """
# The marginal log-likelihood of the fixed effects L(Θ) = \\int f_{\\Theta}(y,b) db
# """
# function marg_nll(Θ)
#     # the `llu` function from Wood's R code
#     nb = length(b_cache)

#     f_yb_mle = optimize(
#         b -> nlyfb_urchin(b, Θ, urchin), 
#         g_nlyfb!,
#         b_cache, 
#         LBFGS(;alphaguess=InitialStatic(scaled=true), linesearch=BackTracking())
#     )
#     b_hat = Optim.minimizer(f_yb_mle)
#     b_cache .= b_hat # updated cached value for next iteration
#     f_yb = Optim.minimum(f_yb_mle)

#     H = hessian(nlyfb_urchin, prep_sp_h_nlyfb, sp_ad_sys, b_hat, Constant(Θ), Constant(urchin))    
#     return f_yb - 0.5 * (log((2π)^nb) - logdet(H))
# end