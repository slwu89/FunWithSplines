using RDatasets, DataFrames
using DifferentiationInterface
using Enzyme
using ForwardDiff

trees = dataset("datasets", "trees")

tree_mod_expectation = function(beta, data)
    return @. beta[1] * (data[!, :Girth]^beta[2]) * (data[!, :Height]^beta[3])
end

beta = [.002, 2, 1]

f = tree_mod_expectation

DifferentiationInterface.jacobian(f, AutoEnzyme(), beta, Constant(trees))
DifferentiationInterface.jacobian(f, AutoForwardDiff(), beta, Constant(trees))

jac_prep = prepare_jacobian(f, AutoForwardDiff(), zeros(length(beta)), Constant(trees))

f_e, f_J = value_and_jacobian(f, jac_prep, AutoForwardDiff(), beta, Constant(trees))