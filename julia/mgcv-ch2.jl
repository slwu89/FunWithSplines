using RDatasets, DataFrames
using LinearAlgebra
using CSV
using StatsModels
using DifferentiationInterface
import ForwardDiff
using Optim, LineSearches

llm = function(theta,X,Z,y)
    # untransform variance parameters (random effects and iid error)
    sigma_b = exp(theta[1])
    sigma = exp(theta[2])
    # extract dimensions
    n = length(y)
    pr = size(Z,2) # num random eff
    pf = size(X,2) # num fixed eff
    # obtain \hat \beta, \hat b...
    X1 = hcat(X,Z)
    ipsi = [zeros(pf); fill(1/sigma_b^2, pr)]
    b1 = ((transpose(X1) * X1) ./ sigma^2 .+ diagm(ipsi)) \ (transpose(X1) * y ./ (sigma^2))
    # compute log|Z'Z/sigma^2 + I/sigma.b^2|...
    ldet = logdet((transpose(Z) * Z) ./ (sigma^2) + diagm(ipsi[pf+1:end]))
    # compute log profile likelihood...
    l = (-sum((y - (X1 * b1)).^2)/sigma^2 - sum(b1.^2 .* ipsi) - n*log(sigma^2) - pr*log(sigma_b^2) - ldet - n*log(2π))/2
    return (-l, b1)
end

Rail = CSV.read("../data/Rail.csv", DataFrame)
Z = ModelFrame(
    @formula(travel ~ 0 + Rail),
    Rail,
    contrasts = Dict(
        :Rail => DummyCoding(base=1)
    )
)
Z = Int.(modelmatrix(Z))
X = ones(18, 1)
y = Rail.travel
theta = [0.0,0.0]


rail_mod = optimize(llm_value, theta, NelderMead(); inplace = false)

# AD stuff

ad_backend = AutoForwardDiff()
llm_first = x -> first(llm(x, X, Z, y))
ad_prep = prepare_gradient(llm_first, ad_backend, similar(theta))
llm_grad(x) = gradient(llm_first, ad_prep, ad_backend, x)

llm_grad([0.0,0.0])

llm_value(x) = first(llm(x, X, Z, y))

llm_value([0.0,0.0])

# fit the model
rail_mod1 = optimize(llm_value, llm_grad, theta, LBFGS(;alphaguess=InitialStatic(scaled=true), linesearch=BackTracking()); inplace = false)

exp.(Optim.minimizer(rail_mod))
exp.(Optim.minimizer(rail_mod1))

inv(hessian(llm_value, ad_backend, Optim.minimizer(rail_mod1)))

last(llm(Optim.minimizer(rail_mod1), X, Z, y))