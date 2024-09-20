using LinearAlgebra, LazySets
using IntervalMDP, IntervalSySCoRe


function building_automation_system_7d()
    A = [
        0.9678 0.0    0.0036 0.0    0.0036 0.0    0.0036;
        0.0    0.9682 0.0    0.0034 0.0    0.0034 0.0034;
        0.0106 0.0    0.9494 0.0    0.0    0.0    0.0;
        0.0    0.0097 0.0    0.9523 0.0    0.0    0.0;
        0.0106 0.0    0.0    0.0    0.9494 0.0    0.0;
        0.0    0.0097 0.0    0.0    0.0    0.9523 0.0;
        0.0106 0.0097 0.0    0.0    0.0    0.0    0.9794;
    ]

    B = zero(Float64, 7, 1)

    w_variance = [1/51.2821, 1/50.0, 1/21.7865, 1/23.5294, 1/25.1889, 1/26.5252, 1/91.7431]
    w_stddev = sqrt.(w_variance)

    dyn = AffineAdditiveNoiseDynamics(A, B, AdditiveDiagonalGaussianNoise(w_stddev))

    initial_region = EmptySet(7)
    reach_region = EmptySet(7)
    avoid_region = EmptySet(7)

    sys = System(dyn, initial_region, reach_region, avoid_region)

    return sys
end

function building_automation_system_7d_decoupled(; sparse=false, state_split=(21, 21, 3, 3, 3, 3, 3))
    sys = building_automation_system_7d()

    X = Hyperrectangle(; low=[-0.525, -0.525, -0.75, -0.75, -0.75, -0.75, -0.75], high=[0.525, 0.525, 0.75, 0.75, 0.75, 0.75, 0.75])
    state_abs = StateUniformGridSplit(X, state_split)

    input_abs = InputDiscrete([Singleton(zero(7))])

    if sparse
        target_model = SparseOrthogonalIMDPTarget()
    else
        target_model = OrthogonalIMDPTarget()
    end

    mdp, reach, avoid = abstraction(sys, state_abs, input_abs, target_model)

    return mdp, reach, avoid
end

function building_automation_system_7d_direct(; sparse=false, state_split=(21, 21, 3, 3, 3, 3, 3))
    sys = building_automation_system_7d()

    X = Hyperrectangle(; low=[-0.525, -0.525, -0.75, -0.75, -0.75, -0.75, -0.75], high=[0.525, 0.525, 0.75, 0.75, 0.75, 0.75, 0.75])
    state_abs = StateUniformGridSplit(X, state_split)

    input_abs = InputDiscrete([Singleton(zero(7))])

    if sparse
        target_model = SparseIMDPTarget()
    else
        target_model = IMDPTarget()
    end

    mdp, reach, avoid = abstraction(sys, state_abs, input_abs, target_model)

    return mdp, reach, avoid
end

function main()
    @time "abstraction" mdp, _, avoid = building_automation_system_7d_decoupled(; state_split=(21, 21, 3, 3, 3, 3, 3))

    prop = FiniteTimeReachability(avoid, 10)
    spec = Specification(prop, Optimistic, Minimize)
    prob = Problem(mdp, spec)

    @time "value iteration" V_unsafety, k, res = value_iteration(prob)
    V_safety = 1.0 .- V_unsafety
end