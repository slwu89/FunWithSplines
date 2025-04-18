# examples from section 5.1.1 Newton's Method
# in Wood's Core Statistics

using DataFrames
using Distributions
using Optim
using LineSearches
using DifferentiationInterface
import ForwardDiff

# use ForwardDiff because it's simple
ad_sys = AutoForwardDiff()

const cell = DataFrame(t=2:14, y=[35,33,33,39,24,25,18,20,23,13,14,20,18])

function model_cell(delta, cell)
    # expected values of counts
    mu = @. 50*exp(-delta*cell.t)
    # neg log likelihood
    -sum([logpdf(Poisson(mu[i]), cell[i, :y]) for i in axes(cell,1)])
end

# a starting value
delta_0 = [5.0]

# one argument closure; not the most efficient, see https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterface/dev/tutorials/advanced/#Contexts
model_cell_f = x -> model_cell(x, cell)

prep_grad_cell = prepare_gradient(model_cell_f, ad_sys, zero(delta_0))

# gradient(model_cell_f, prep_grad_cell, ad_sys, delta_0)

grad_cell!(G, x) = gradient!(model_cell_f, G, prep_grad_cell, ad_sys, x)

prep_hess_cell = prepare_hessian(model_cell_f, ad_sys, zero(delta_0))

# hessian(model_cell_f, prep_hess_cell, ad_sys, delta_0)

hess_cell!(H, x) = hessian!(model_cell_f, H, prep_hess_cell, ad_sys, x)

# default HagerZhang line search is too aggressive
result_cell = optimize(model_cell_f, grad_cell!, hess_cell!, delta_0, Newton(;alphaguess=InitialStatic(scaled=true), linesearch=BackTracking()))
Optim.minimizer(result_cell)