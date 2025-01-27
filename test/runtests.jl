using IntervalMDPAbstractions
using Test

@testset verbose = true "IntervalMDPAbstractions.jl" begin
    @testset verbose = true "dynamics" include("dynamics/dynamics.jl")
    @testset verbose = true "abstractions" include("abstractions/abstractions.jl")
    @testset verbose = true "specifications" include("specifications.jl")
end
