using LinearAlgebra, LazySets
using IntervalMDP, IntervalSySCoRe


function building_automation_system_7d(time_horizon)
    A = [
        0.9678 0.0    0.0036 0.0    0.0036 0.0    0.0036;
        0.0    0.9682 0.0    0.0034 0.0    0.0034 0.0034;
        0.0106 0.0    0.9494 0.0    0.0    0.0    0.0;
        0.0    0.0097 0.0    0.9523 0.0    0.0    0.0;
        0.0106 0.0    0.0    0.0    0.9494 0.0    0.0;
        0.0    0.0097 0.0    0.0    0.0    0.9523 0.0;
        0.0106 0.0097 0.0    0.0    0.0    0.0    0.9794;
    ]

    B = zeros(Float64, 7, 1)

    w_variance = [1/51.2821, 1/50.0, 1/21.7865, 1/23.5294, 1/25.1889, 1/26.5252, 1/91.7431]
    w_stddev = sqrt.(w_variance)

    dyn = AffineAdditiveNoiseDynamics(A, B, AdditiveDiagonalGaussianNoise(w_stddev))

    initial_region = EmptySet(7)
    sys = System(dyn, initial_region)

    avoid_region = EmptySet(7)
    prop = FiniteTimeRegionSafety(avoid_region, time_horizon)
    spec = Specification(prop, Pessimistic, Maximize)

    return sys, spec
end

function building_automation_system_7d_decoupled(time_horizon=10; sparse=false, state_split=(21, 21, 3, 3, 3, 3, 3))
    sys, spec = building_automation_system_7d(time_horizon)

    X = Hyperrectangle(; low=[-0.525, -0.525, -0.75, -0.75, -0.75, -0.75, -0.75], high=[0.525, 0.525, 0.75, 0.75, 0.75, 0.75, 0.75])
    state_abs = StateUniformGridSplit(X, state_split)

    input_abs = InputDiscrete([Singleton([0.0])])

    if sparse
        target_model = SparseOrthogonalIMDPTarget()
    else
        target_model = OrthogonalIMDPTarget()
    end

    prob = AbstractionProblem(sys, spec)
    mdp, abstract_spec = abstraction(prob, state_abs, input_abs, target_model)

    upper_bound_spec = Specification(system_property(spec), !satisfaction_mode(spec))
    upper_bound_spec = convert_specification(upper_bound_spec, state_abs, target_model)

    return mdp, abstract_spec, upper_bound_spec
end

function building_automation_system_7d_direct(time_horizon=10; sparse=false, state_split=(21, 21, 3, 3, 3, 3, 3))
    sys, spec = building_automation_system_7d(time_horizon)

    X = Hyperrectangle(; low=[-0.525, -0.525, -0.75, -0.75, -0.75, -0.75, -0.75], high=[0.525, 0.525, 0.75, 0.75, 0.75, 0.75, 0.75])
    state_abs = StateUniformGridSplit(X, state_split)

    input_abs = InputDiscrete([Singleton([0.0])])

    if sparse
        target_model = SparseIMDPTarget()
    else
        target_model = IMDPTarget()
    end

    prob = AbstractionProblem(sys, spec)
    mdp, abstract_spec = abstraction(prob, state_abs, input_abs, target_model)

    upper_bound_spec = Specification(system_property(spec), !satisfaction_mode(spec))
    upper_bound_spec = convert_specification(upper_bound_spec, state_abs, target_model)

    return mdp, abstract_spec, upper_bound_spec
end

function main()
    @time "abstraction" mdp, spec, _ = building_automation_system_7d_decoupled(; state_split=(21, 21, 3, 3, 3, 3, 3))
    prob = Problem(mdp, spec)

    @time "value iteration" V_safety, k, res = value_iteration(prob)

    # Remove the first state from each axis (the avoid state, whose value is always 0).
    V_safety = V_safety[(2:d for d in size(V_safety))...]

    return V_safety
end