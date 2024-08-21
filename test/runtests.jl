using IntervalSySCoRe
using Test

@testset verbose = true "IntervalSySCoRe.jl" begin
    @testset verbose = true "dynamics" include("dynamics/dynamics.jl")
    @testset verbose = true "abstractions" include("abstractions/abstractions.jl")
end
