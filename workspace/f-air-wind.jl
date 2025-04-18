using RDatasets, DataFrames
using GLM
using Plots

airquality = dataset("datasets", "airquality")
Dat = DataFrame(x=airquality.Wind, y=airquality.Ozone)
dropmissing!(Dat)

poly_mod = fit(LinearModel, @formula(y ~ 1 + x + x^2), Dat)
linear_mod = fit(LinearModel, @formula(y ~ 1 + x), Dat)
DatSort = sort(Dat, :x)

p1 = scatter(
    Dat.x, Dat.y, legend=false, color="black",
    xlabel="Wind speed (mph)", ylabel="Ozone concentration (ppb)",
    title="New York air quality"
)
plot!(p1, DatSort.x, predict(linear_mod, DatSort), legend=false, linestyle=:dash, linecolor="blue")
plot!(p1, DatSort.x, predict(poly_mod, DatSort), legend=false, linecolor="red")