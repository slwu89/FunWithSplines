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

@model function bayesian_spline_rw1(y, B)
    # prior on RW(1) variance for coefficients
    sigma_f ~ truncated(Cauchy(0, 1); lower=0)
    # prior on error variance
    sigma_e ~ truncated(Cauchy(0, 1); lower=0)

    # RW(1) prior on coefficients
    alpha = zeros(size(B, 2))
    alpha[1] ~ Normal(0, sigma_f)
    for i in 2:size(B, 2)
        alpha[i] ~ Normal(alpha[i-1], sigma_f)
    end

    # evaluated spline
    mu = B * alpha

    # likelihood
    for i in eachindex(y)
        y[i] ~ mu[i] + Normal(0, sigma_e)
    end
end

model1 = bayesian_spline_rw1(y, B)

n_samples_post = 1_000
chain1 = sample(model1, NUTS(), n_samples_post)
chain1 = DataFrame(chain1)[:, Cols(:sigma_f, :sigma_e, r"alpha")]

function plot_samples(chain, x, y, B, f_true, title)
    p1 = scatter(x, y, legend=false, markeralpha=0.5, color="black", title=title)
    for i in 1:n_samples_post
        plot!(
            p1, x, B * Vector(chain[i, r"alpha"]), 
            legend=false, color="red", linealpha=0.05
        )
    end
    plot!(p1, x, f_true, legend=false, color="black", linestyle=:dash)
    p1
end

function plot_quantiles(chain, x, y, B, f_true, title)
    q_05 = mapcols(x -> quantile(x, 0.5), chain)
    q_025 = mapcols(x -> quantile(x, 0.025), chain)
    q_975 = mapcols(x -> quantile(x, 0.975), chain)
    p1 = scatter(x, y, legend=false, markeralpha=0.5, color="black", title=title)
    mid_q = B * Vector(q_05[1, r"alpha"])
    upper_q = B * Vector(q_975[1, r"alpha"])
    lower_q = B * Vector(q_025[1, r"alpha"])
    plot!(p1, x, ribbon = (mid_q .- lower_q, upper_q .- mid_q), mid_q, legend=false, color="red", linealpha=0.8)
    plot!(p1, x, f_true, legend=false, color="black", linestyle=:dash)
    p1
end

plot(
    plot_samples(chain1, x, y, B, f_true, "RW(1) prior with samples"),
    plot_quantiles(chain1, x, y, B, f_true, "RW(1) prior with quantiles"),
    size=(1000,600)
)

@model function bayesian_spline_rw2(y, B)
    # prior on RW(2) variance for coefficients
    sigma_f ~ truncated(Cauchy(0, 1); lower=0)
    # prior on error variance
    sigma_e ~ truncated(Cauchy(0, 1); lower=0)

    # RW(2) prior on coefficients
    alpha = zeros(size(B, 2))
    alpha[1] ~ Normal(0, sigma_f)
    alpha[2] ~ Normal(0, sigma_f)
    for i in 3:size(B, 2)
        alpha[i] ~ Normal(2*alpha[i-1] - alpha[i-2], sigma_f)
    end

    # evaluated spline
    mu = B * alpha

    # likelihood
    for i in eachindex(y)
        y[i] ~ mu[i] + Normal(0, sigma_e)
    end
end

model2 = bayesian_spline_rw2(y, B)

n_samples_post = 1_000
chain2 = sample(model2, NUTS(), n_samples_post)
chain2 = DataFrame(chain2)[:, Cols(:sigma_f, :sigma_e, r"alpha")]

plot(
    plot_samples(chain2, x, y, B, f_true, "RW(2) prior with samples"),
    plot_quantiles(chain2, x, y, B, f_true, "RW(2) prior with quantiles"),
    size=(1000,600)
)