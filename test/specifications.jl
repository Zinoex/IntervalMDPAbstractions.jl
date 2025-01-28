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

    reach_region = Hyperrectangle(; low = [5.0, -6.0], high = [10.0, -2.0])
    avoid_region = Hyperrectangle(; low = [5.0, 2.0], high = [10.0, 6.0])

    return sys, reach_region, avoid_region
end

@testset "preserve modes" begin
    sys, reach_region, avoid_region = sys_2d_ra()

    X = Hyperrectangle(; low = [-10.0, -10.0], high = [10.0, 10.0])
    state_abs = StateUniformGridSplit(X, (10, 10))

    U = Hyperrectangle(; low = [-1.0, -1.0], high = [1.0, 1.0])
    input_abs = InputGridSplit(U, [3, 3])

    target_model = IMDPTarget()

    horizon = 10
    prop = FiniteTimeRegionReachAvoid(reach_region, avoid_region, horizon)
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

decoupled_avoid_outside = [
    CartesianIndex(11, 1),
    CartesianIndex(11, 2),
    CartesianIndex(11, 3),
    CartesianIndex(11, 4),
    CartesianIndex(11, 5),
    CartesianIndex(11, 6),
    CartesianIndex(11, 7),
    CartesianIndex(11, 8),
    CartesianIndex(11, 9),
    CartesianIndex(11, 10),
    CartesianIndex(1, 11),
    CartesianIndex(2, 11),
    CartesianIndex(3, 11),
    CartesianIndex(4, 11),
    CartesianIndex(5, 11),
    CartesianIndex(6, 11),
    CartesianIndex(7, 11),
    CartesianIndex(8, 11),
    CartesianIndex(9, 11),
    CartesianIndex(10, 11),
    CartesianIndex(11, 11),
]

@testset "reachability" begin
    sys, reach_region, avoid_region = sys_2d_ra()

    X = Hyperrectangle(; low = [-10.0, -10.0], high = [10.0, 10.0])
    state_abs = StateUniformGridSplit(X, (10, 10))

    U = Hyperrectangle(; low = [-1.0, -1.0], high = [1.0, 1.0])
    input_abs = InputGridSplit(U, [3, 3])

    horizon = 10
    prop = FiniteTimeRegionReachability(reach_region, horizon)
    spec = Specification(prop, Pessimistic, Maximize)
    prob = AbstractionProblem(sys, spec)

    inverted_spec = Specification(prop, Optimistic, Minimize)
    inverted_prob = AbstractionProblem(sys, inverted_spec)

    @testset "direct" begin
        target_model = IMDPTarget()

        mdp, abstract_spec = abstraction(prob, state_abs, input_abs, target_model)

        abstract_prop = system_property(abstract_spec)
        @test IntervalMDP.avoid(abstract_prop) == [CartesianIndex(101)]
        @test all(
            IntervalMDP.reach(abstract_prop) .== [
                CartesianIndex(29),
                CartesianIndex(30),
                CartesianIndex(39),
                CartesianIndex(40),
            ],
        )

        mdp, inverted_abstract_spec =
            abstraction(inverted_prob, state_abs, input_abs, target_model)

        abstract_prop = system_property(inverted_abstract_spec)
        @test IntervalMDP.avoid(abstract_prop) == [CartesianIndex(101)]
        @test all(
            IntervalMDP.reach(abstract_prop) .== [
                CartesianIndex(28),
                CartesianIndex(29),
                CartesianIndex(30),
                CartesianIndex(38),
                CartesianIndex(39),
                CartesianIndex(40),
            ],
        )
    end

    @testset "decoupled" begin
        target_model = OrthogonalIMDPTarget()

        mdp, abstract_spec = abstraction(prob, state_abs, input_abs, target_model)

        abstract_prop = system_property(abstract_spec)
        @test IntervalMDP.avoid(abstract_prop) == decoupled_avoid_outside
        @test all(
            IntervalMDP.reach(abstract_prop) .== [
                CartesianIndex(9, 3),
                CartesianIndex(10, 3),
                CartesianIndex(9, 4),
                CartesianIndex(10, 4),
            ],
        )

        mdp, inverted_abstract_spec =
            abstraction(inverted_prob, state_abs, input_abs, target_model)

        abstract_prop = system_property(inverted_abstract_spec)
        @test IntervalMDP.avoid(abstract_prop) == decoupled_avoid_outside
        @test all(
            IntervalMDP.reach(abstract_prop) .== [
                CartesianIndex(8, 3),
                CartesianIndex(9, 3),
                CartesianIndex(10, 3),
                CartesianIndex(8, 4),
                CartesianIndex(9, 4),
                CartesianIndex(10, 4),
            ],
        )
    end
end

@testset "reach-avoid" begin
    sys, reach_region, avoid_region = sys_2d_ra()

    X = Hyperrectangle(; low = [-10.0, -10.0], high = [10.0, 10.0])
    state_abs = StateUniformGridSplit(X, (10, 10))

    U = Hyperrectangle(; low = [-1.0, -1.0], high = [1.0, 1.0])
    input_abs = InputGridSplit(U, [3, 3])

    horizon = 10
    prop = FiniteTimeRegionReachAvoid(reach_region, avoid_region, horizon)
    spec = Specification(prop, Pessimistic, Maximize)
    prob = AbstractionProblem(sys, spec)

    inverted_spec = Specification(prop, Optimistic, Minimize)
    inverted_prob = AbstractionProblem(sys, inverted_spec)

    @testset "direct" begin
        target_model = IMDPTarget()

        mdp, abstract_spec = abstraction(prob, state_abs, input_abs, target_model)

        abstract_prop = system_property(abstract_spec)
        @test IntervalMDP.avoid(abstract_prop) == [
            CartesianIndex(101),
            CartesianIndex(68),
            CartesianIndex(69),
            CartesianIndex(70),
            CartesianIndex(78),
            CartesianIndex(79),
            CartesianIndex(80),
        ]
        @test all(
            IntervalMDP.reach(abstract_prop) .== [
                CartesianIndex(29),
                CartesianIndex(30),
                CartesianIndex(39),
                CartesianIndex(40),
            ],
        )

        mdp, inverted_abstract_spec =
            abstraction(inverted_prob, state_abs, input_abs, target_model)

        abstract_prop = system_property(inverted_abstract_spec)
        @test IntervalMDP.avoid(abstract_prop) == [
            CartesianIndex(101),
            CartesianIndex(69),
            CartesianIndex(70),
            CartesianIndex(79),
            CartesianIndex(80),
        ]
        @test all(
            IntervalMDP.reach(abstract_prop) .== [
                CartesianIndex(28),
                CartesianIndex(29),
                CartesianIndex(30),
                CartesianIndex(38),
                CartesianIndex(39),
                CartesianIndex(40),
            ],
        )
    end

    @testset "decoupled" begin
        target_model = OrthogonalIMDPTarget()

        mdp, abstract_spec = abstraction(prob, state_abs, input_abs, target_model)

        abstract_prop = system_property(abstract_spec)
        @test IntervalMDP.avoid(abstract_prop) == [
            decoupled_avoid_outside
            [
                CartesianIndex(8, 7),
                CartesianIndex(9, 7),
                CartesianIndex(10, 7),
                CartesianIndex(8, 8),
                CartesianIndex(9, 8),
                CartesianIndex(10, 8),
            ]
        ]
        @test all(
            IntervalMDP.reach(abstract_prop) .== [
                CartesianIndex(9, 3),
                CartesianIndex(10, 3),
                CartesianIndex(9, 4),
                CartesianIndex(10, 4),
            ],
        )

        mdp, inverted_abstract_spec =
            abstraction(inverted_prob, state_abs, input_abs, target_model)

        abstract_prop = system_property(inverted_abstract_spec)
        @test IntervalMDP.avoid(abstract_prop) == [
            decoupled_avoid_outside
            [
                CartesianIndex(9, 7),
                CartesianIndex(10, 7),
                CartesianIndex(9, 8),
                CartesianIndex(10, 8),
            ]
        ]
        @test all(
            IntervalMDP.reach(abstract_prop) .== [
                CartesianIndex(8, 3),
                CartesianIndex(9, 3),
                CartesianIndex(10, 3),
                CartesianIndex(8, 4),
                CartesianIndex(9, 4),
                CartesianIndex(10, 4),
            ],
        )
    end
end

@testset "safety" begin
    sys, reach_region, avoid_region = sys_2d_ra()

    X = Hyperrectangle(; low = [-10.0, -10.0], high = [10.0, 10.0])
    state_abs = StateUniformGridSplit(X, (10, 10))

    U = Hyperrectangle(; low = [-1.0, -1.0], high = [1.0, 1.0])
    input_abs = InputGridSplit(U, [3, 3])

    horizon = 10
    prop = FiniteTimeRegionSafety(avoid_region, horizon)
    spec = Specification(prop, Pessimistic, Maximize)
    prob = AbstractionProblem(sys, spec)

    inverted_spec = Specification(prop, Optimistic, Minimize)
    inverted_prob = AbstractionProblem(sys, inverted_spec)

    @testset "direct" begin
        target_model = IMDPTarget()

        mdp, abstract_spec = abstraction(prob, state_abs, input_abs, target_model)

        abstract_prop = system_property(abstract_spec)
        @test IntervalMDP.avoid(abstract_prop) == [
            CartesianIndex(101),
            CartesianIndex(68),
            CartesianIndex(69),
            CartesianIndex(70),
            CartesianIndex(78),
            CartesianIndex(79),
            CartesianIndex(80),
        ]

        mdp, inverted_abstract_spec =
            abstraction(inverted_prob, state_abs, input_abs, target_model)

        abstract_prop = system_property(inverted_abstract_spec)
        @test IntervalMDP.avoid(abstract_prop) == [
            CartesianIndex(101),
            CartesianIndex(69),
            CartesianIndex(70),
            CartesianIndex(79),
            CartesianIndex(80),
        ]
    end

    @testset "decoupled" begin
        target_model = OrthogonalIMDPTarget()

        mdp, abstract_spec = abstraction(prob, state_abs, input_abs, target_model)

        abstract_prop = system_property(abstract_spec)
        @test IntervalMDP.avoid(abstract_prop) == [
            decoupled_avoid_outside
            [
                CartesianIndex(8, 7),
                CartesianIndex(9, 7),
                CartesianIndex(10, 7),
                CartesianIndex(8, 8),
                CartesianIndex(9, 8),
                CartesianIndex(10, 8),
            ]
        ]

        mdp, inverted_abstract_spec =
            abstraction(inverted_prob, state_abs, input_abs, target_model)

        abstract_prop = system_property(inverted_abstract_spec)
        @test IntervalMDP.avoid(abstract_prop) == [
            decoupled_avoid_outside
            [
                CartesianIndex(9, 7),
                CartesianIndex(10, 7),
                CartesianIndex(9, 8),
                CartesianIndex(10, 8),
            ]
        ]
    end
end

@testset "safety emptyset/stay in roi" begin
    sys, reach_region, avoid_region = sys_2d_ra()

    # Region of interest
    X = Hyperrectangle(; low = [-10.0, -10.0], high = [10.0, 10.0])
    state_abs = StateUniformGridSplit(X, (10, 10))

    U = Hyperrectangle(; low = [-1.0, -1.0], high = [1.0, 1.0])
    input_abs = InputGridSplit(U, [3, 3])

    horizon = 10
    avoid_region = EmptySet(2)
    prop = FiniteTimeRegionSafety(avoid_region, horizon)
    spec = Specification(prop, Pessimistic, Maximize)
    prob = AbstractionProblem(sys, spec)

    inverted_spec = Specification(prop, Optimistic, Minimize)
    inverted_prob = AbstractionProblem(sys, inverted_spec)

    @testset "direct" begin
        target_model = IMDPTarget()

        mdp, abstract_spec = abstraction(prob, state_abs, input_abs, target_model)

        abstract_prop = system_property(abstract_spec)
        @test IntervalMDP.avoid(abstract_prop) == [CartesianIndex(101)]

        mdp, inverted_abstract_spec =
            abstraction(inverted_prob, state_abs, input_abs, target_model)

        abstract_prop = system_property(inverted_abstract_spec)
        @test IntervalMDP.avoid(abstract_prop) == [CartesianIndex(101)]
    end

    @testset "decoupled" begin
        target_model = OrthogonalIMDPTarget()

        mdp, abstract_spec = abstraction(prob, state_abs, input_abs, target_model)

        abstract_prop = system_property(abstract_spec)
        @test IntervalMDP.avoid(abstract_prop) == decoupled_avoid_outside

        mdp, inverted_abstract_spec =
            abstraction(inverted_prob, state_abs, input_abs, target_model)

        abstract_prop = system_property(inverted_abstract_spec)
        @test IntervalMDP.avoid(abstract_prop) == decoupled_avoid_outside
    end
end
