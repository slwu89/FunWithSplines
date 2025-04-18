# getting a handle on the RW(1) and RW(2) priors
# from this https://inlatools.netlify.app/articles/rwprior
using Distributions
using Plots

function simulate_rw1(n, sigma)
    rw_path = zeros(n)
    rw_path[1] = rand(Normal(0, sigma))
    for i in 2:length(rw_path)
        rw_path[i] = rand(Normal(rw_path[i-1], sigma))
    end
    return rw_path
end

function simulate_rw2(n, sigma)
    rw_path = zeros(n)
    rw_path[1:2] .= rand(Normal(0, sigma), 2)
    for i in 3:length(rw_path)
        rw_path[i] = rand(Normal(2*rw_path[i-1] - rw_path[i-2], sigma))
    end
    return rw_path
end

rw1_trajs = zeros(20, 1000)
for i in axes(rw1_trajs, 2)
    rw1_trajs[:, i] .= simulate_rw1(20, 0.05)
end

rw2_trajs = zeros(20, 1000)
for i in axes(rw2_trajs, 2)
    rw2_trajs[:, i] .= simulate_rw2(20, 0.05)
end

plot(rw1_trajs, linealpha=0.05, legend=false, color="black")
Plots.abline!(0, 0, color="red", linestyle=:dash, linewidth=2)

plot(rw2_trajs, linealpha=0.05, legend=false, color="black")
Plots.abline!(0, 0, color="red", linestyle=:dash, linewidth=2)