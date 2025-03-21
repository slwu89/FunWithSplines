using RDatasets, DataFrames
using GLM

mcycle = dataset("MASS", "mcycle")
times = mcycle.Times
x = deepcopy(times)
accel = mcycle.Accel
y = deepcopy(accel)
Data = DataFrame(x=x,y=y)

function make_grid(x, n=100)
    return range(minimum(x), maximum(x), length=n)
end

mc1 = subset(mcycle, :Times => (x -> x .> 5))
lm1 = fit(LinearModel, @formula(Accel ~ Times + Times^2 + Times^3 + Times^4 + Times^5 + Times^6 + Times^7 + Times^8 + Times^9), mc1)

lm = fit(LinearModel, @formula(Accel ~ Times + Times^2 + Times^3 + Times^4 + Times^5 + Times^6 + Times^7 + Times^8 + Times^9), mcycle)
