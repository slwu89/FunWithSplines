using RDatasets, DataFrames
using DifferentiationInterface
using Enzyme
using ForwardDiff
using GLM
using LinearAlgebra

trees = dataset("datasets", "trees")

tree_mod_expectation = function(beta, data)
    return @. beta[1] * (data[!, :Girth]^beta[2]) * (data[!, :Height]^beta[3])
end

beta = [.002, 2, 1]

f = tree_mod_expectation

# DifferentiationInterface.jacobian(f, AutoEnzyme(), beta, Constant(trees))
DifferentiationInterface.jacobian(f, AutoForwardDiff(), beta, Constant(trees))
jac_prep = prepare_jacobian(f, AutoForwardDiff(), zeros(length(beta)), Constant(trees))

# iterative least squares
beta_old = @. (beta*100)+100 # need to make it ridiculous so algo does not stop immediately
while sum(abs.(beta - beta_old)) > 1E-7*sum(abs.(beta_old))
    mu, J = value_and_jacobian(f, jac_prep, AutoForwardDiff(), beta, Constant(trees))
    z = (trees[!, :Volume] - mu) + (J * beta) # pseudodata
    beta_old = beta
    beta = coef(lm(J, z))
end

# sig2 estimator in eqn 1.8: \hat{\sigma}^{2} = \left \lVert r \right \rVert^{2} \ (n-p)
# r is residuals
# p is num of cols in X (design/model matrix)
mu, J = value_and_jacobian(f, jac_prep, AutoForwardDiff(), beta, Constant(trees))
sig2 = sum((trees[!, :Volume] - mu).^2) / (nrow(trees) - length(beta))
Vb = inv(transpose(J) * J) .* sig2
se = sqrt.(diag(Vb))
