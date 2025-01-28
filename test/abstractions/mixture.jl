using Revise, Test
using LinearAlgebra, LazySets
using IntervalMDP, IntervalMDPAbstractions


A1 = [
    0.1 0.9
    0.8 0.2
]
B1 = [
    0.0
    0.0
][:, :]
w1_stddev = [0.3, 0.2]
mode1 = AffineAdditiveNoiseDynamics(A1, B1, AdditiveDiagonalGaussianNoise(w1_stddev))

A2 = [
    0.8 0.2
    0.1 0.9
]
B2 = [
    0.0
    0.0
][:, :]
w2_stddev = [0.2, 0.1]
mode2 = AffineAdditiveNoiseDynamics(A2, B2, AdditiveDiagonalGaussianNoise(w2_stddev))

dyn = StochasticSwitchedDynamics([mode1, mode2], [0.7, 0.3])
initial_region = EmptySet(2)
sys = System(dyn, initial_region)

horizon = 10
reach_region = Hyperrectangle(; low = [-1.0, -1.0], high = [0.0, 1.0])
avoid_region = Hyperrectangle(; low = [1.0, 0.0], high = [2.0, 1.0])
prop = FiniteTimeRegionReachAvoid(reach_region, avoid_region, horizon)
spec = Specification(prop, Pessimistic, Maximize)

prob = AbstractionProblem(sys, spec)

X = Hyperrectangle(; low = [-2.0, -2.0], high = [2.0, 2.0])
state_split = (20, 20)
state_abs = StateUniformGridSplit(X, state_split)

input_abs = InputDiscrete([Singleton([0.0])])

@testset "mixture vs direct" begin
    mdp_mixture, abstract_spec_mixture =
        abstraction(prob, state_abs, input_abs, MixtureIMDPTarget())

    @test num_states(mdp_mixture) == 21 * 21
    @test length(stateptr(mdp_mixture)) == 20 * 20 + 1  # 20 * 20 non-sink states
    @test stateptr(mdp_mixture)[end] == 20 * 20 + 1  # No control actions

    prob_mixture = Problem(mdp_mixture, abstract_spec_mixture)

    V_mixture, k, res = value_iteration(prob_mixture)
    @test k == 10

    mdp_direct, abstract_spec_direct = abstraction(prob, state_abs, input_abs, IMDPTarget())

    @test num_states(mdp_direct) == 20 * 20 + 1
    @test length(stateptr(mdp_direct)) == 20 * 20 + 1  # 20 * 20 non-sink states
    @test stateptr(mdp_direct)[end] == 20 * 20 + 1  # No control actions

    prob_direct = Problem(mdp_direct, abstract_spec_direct)

    V_direct, k, res = value_iteration(prob_direct)
    @test k == 10
    @test all(V_mixture[1:end-1, 1:end-1] .≥ reshape(V_direct[1:end-1], state_split))
end
