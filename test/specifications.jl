
# TODO: Test Reachability, ReachAvoid, and Safety properties (with EmptySet for avoid regions)
# TODO: Test reach/avoid states in Pessimistic vs Optimistic satisfaction_modes
# TODO: Test specifications for direct vs decoupled abstractions

using Revise, Test
using LinearAlgebra, LazySets
using IntervalMDP, IntervalMDPAbstractions

function sys_2d_ra()
    A = 0.9I(2)
    B = 0.7I(2)
    w_stddev = [1.0, 1.0]

    dyn = AffineAdditiveNoiseDynamics(A, B, AdditiveDiagonalGaussianNoise(w_stddev))

    initial_region = EmptySet(2)

    sys = System(dyn, initial_region)

    reach_region = Hyperrectangle(; low = [4.0, -6.0], high = [10.0, -2.0])
    avoid_region = EmptySet(2)

    return sys, reach_region, avoid_region
end

@testset "preserve modes" begin
    sys, reach_region, avoid_region = sys_2d_ra()

    X = Hyperrectangle(; low = [-10.0, -10.0], high = [10.0, 10.0])
    state_abs = StateUniformGridSplit(X, (10, 10))

    U = Hyperrectangle(; low = [-1.0, -1.0], high = [1.0, 1.0])
    input_abs = InputGridSplit(U, [3, 3])

    target_model = IMDPTarget()

    time_horizon = 10
    prop = FiniteTimeRegionReachAvoid(reach_region, avoid_region, time_horizon)
    spec = Specification(prop, Pessimistic, Maximize)

    prob = AbstractionProblem(sys, spec)
    mdp, abstract_spec = abstraction(prob, state_abs, input_abs, target_model)

    @test satisfaction_mode(spec) == satisfaction_mode(abstract_spec)
    @test strategy_mode(spec) == strategy_mode(abstract_spec)
end


@testset "preserve termination criteria" begin
    sys, reach_region, avoid_region = sys_2d_ra()

    X = Hyperrectangle(; low = [-10.0, -10.0], high = [10.0, 10.0])
    state_abs = StateUniformGridSplit(X, (10, 10))

    U = Hyperrectangle(; low = [-1.0, -1.0], high = [1.0, 1.0])
    input_abs = InputGridSplit(U, [3, 3])

    target_model = IMDPTarget()

    @testset "finite time" begin
        horizon = 10

        @testset "reachability" begin
            prop = FiniteTimeRegionReachability(reach_region, horizon)
            spec = Specification(prop, Pessimistic, Maximize)
            prop = system_property(spec)

            prob = AbstractionProblem(sys, spec)
            mdp, abstract_spec = abstraction(prob, state_abs, input_abs, target_model)
            abstract_prop = system_property(abstract_spec)

            @test isfinitetime(prop) == isfinitetime(abstract_prop)
            @test isfinitetime(prop) == true
            @test time_horizon(prop) == time_horizon(abstract_prop)
            @test time_horizon(prop) == horizon
        end

        @testset "reach-avoid" begin
            prop = FiniteTimeRegionReachAvoid(reach_region, avoid_region, horizon)
            spec = Specification(prop, Pessimistic, Maximize)
            prop = system_property(spec)

            prob = AbstractionProblem(sys, spec)
            mdp, abstract_spec = abstraction(prob, state_abs, input_abs, target_model)
            abstract_prop = system_property(abstract_spec)

            @test isfinitetime(prop) == isfinitetime(abstract_prop)
            @test isfinitetime(prop) == true
            @test time_horizon(prop) == time_horizon(abstract_prop)
            @test time_horizon(prop) == horizon
        end

        @testset "safety" begin
            prop = FiniteTimeRegionSafety(avoid_region, horizon)
            spec = Specification(prop, Pessimistic, Maximize)
            prop = system_property(spec)

            prob = AbstractionProblem(sys, spec)
            mdp, abstract_spec = abstraction(prob, state_abs, input_abs, target_model)
            abstract_prop = system_property(abstract_spec)

            @test isfinitetime(prop) == isfinitetime(abstract_prop)
            @test isfinitetime(prop) == true
            @test time_horizon(prop) == time_horizon(abstract_prop)
            @test time_horizon(prop) == horizon
        end
    end

    @testset "infinite time" begin
        eps = 1e-6

        @testset "reachability" begin
            prop = InfiniteTimeRegionReachability(reach_region, eps)
            spec = Specification(prop, Pessimistic, Maximize)
            prop = system_property(spec)

            prob = AbstractionProblem(sys, spec)
            mdp, abstract_spec = abstraction(prob, state_abs, input_abs, target_model)
            abstract_prop = system_property(abstract_spec)

            @test isfinitetime(prop) == isfinitetime(abstract_prop)
            @test isfinitetime(prop) == false
            @test convergence_eps(prop) == convergence_eps(abstract_prop)
            @test convergence_eps(prop) == eps
        end

        @testset "reach-avoid" begin
            prop = InfiniteTimeRegionReachAvoid(reach_region, avoid_region, eps)
            spec = Specification(prop, Pessimistic, Maximize)
            prop = system_property(spec)

            prob = AbstractionProblem(sys, spec)
            mdp, abstract_spec = abstraction(prob, state_abs, input_abs, target_model)
            abstract_prop = system_property(abstract_spec)

            @test isfinitetime(prop) == isfinitetime(abstract_prop)
            @test isfinitetime(prop) == false
            @test convergence_eps(prop) == convergence_eps(abstract_prop)
            @test convergence_eps(prop) == eps
        end

        @testset "safety" begin
            prop = InfiniteTimeRegionSafety(avoid_region, eps)
            spec = Specification(prop, Pessimistic, Maximize)
            prop = system_property(spec)

            prob = AbstractionProblem(sys, spec)
            mdp, abstract_spec = abstraction(prob, state_abs, input_abs, target_model)
            abstract_prop = system_property(abstract_spec)

            @test isfinitetime(prop) == isfinitetime(abstract_prop)
            @test isfinitetime(prop) == false
            @test convergence_eps(prop) == convergence_eps(abstract_prop)
            @test convergence_eps(prop) == eps
        end
    end
end