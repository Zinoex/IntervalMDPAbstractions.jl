using Revise, Test
using LinearAlgebra, LazySets
using IntervalMDP, IntervalMDPAbstractions

@testset "decoupling" begin
    include("example_systems.jl")

    sys, spec = modified_running_example_sys()
    prob = AbstractionProblem(sys, spec)
    new_prob, Tx = decouple(prob)
    @test prob == new_prob
    @test Tx.T == I
    @test Tx.Tinv == I

    sys, spec = modified_running_example_sys(; noise=:nondiagonal)
    prob = AbstractionProblem(sys, spec)
    new_prob, Tx = decouple(prob)
    @test prob != new_prob
    @test Tx.T != I
    @test Tx.Tinv != I
end

test_files = ["direct.jl", "decoupled.jl", "compare_abstractions.jl"]
for f in test_files
    @testset "abstractions/additive_noise/$f" include(f)
end
