# stuff on mixed models in ch2
using DataFrames, CSV
using MixedModels
using StatsBase

Rail = CSV.read("../data/Rail.csv", DataFrame)
rail_mod = lmm(@formula(travel ~ 1 + (1|Rail)), Rail, REML=true)

# fixed effects table
coeftable(rail_mod)
# random effects table
DataFrame(only(raneftables(rail_mod)))


Loblolly = CSV.read("../data/Loblolly.csv", DataFrame)

# center the age variable
transform!(Loblolly, :age => (x -> x .- mean(x)) => :age)