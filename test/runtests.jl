using IntervalSySCoRe
using Test

@testset verbose = true "IntervalSySCoRe.jl" begin
    @testset verbose = true "systems" include("systems/systems.jl")
end
