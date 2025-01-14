using LinearAlgebra, LazySets
using IntervalMDP, IntervalMDPAbstractions


function building_automation_system_4d(time_horizon)
    A = [
        0.6682 0.0 0.02632 0.0
        0.0 0.683 0.0 0.02096
        1.0005 0.0 -0.000499 0.0
        0.0 0.8004 0.0 0.1996
    ]

    B = [
        0.1320
        0.1402
        0.0
        0.0
    ][:, :]

    C = [3.4378, 2.9272, 13.0207, 10.4166]

    w_variance = [1 / 12.9199, 1 / 12.9199, 1 / 2.5826, 1 / 3.2276]
    w_stddev = sqrt.(w_variance)

    dyn = AffineAdditiveNoiseDynamics(A, B, C, AdditiveDiagonalGaussianNoise(w_stddev))

    initial_region = EmptySet(4)

    sys = System(dyn, initial_region)

    avoid_region = EmptySet(4)
    prop = FiniteTimeRegionSafety(avoid_region, time_horizon)
    spec = Specification(prop, Pessimistic, Maximize)

    return sys, spec
end

function building_automation_system_4d_decoupled(
    time_horizon = 10;
    sparse = false,
    state_split = (5, 5, 7, 7),
    input_split = 4,
)
    sys, spec = building_automation_system_4d(time_horizon)

    X = Hyperrectangle(;
        low = [18.75, 18.75, 29.5, 29.5],
        high = [21.25, 21.25, 36.5, 36.5],
    )
    state_abs = StateUniformGridSplit(X, state_split)

    U = Hyperrectangle(; low = [17.0], high = [20.0])
    input_abs = InputLinRange(U, input_split)

    if sparse
        target_model = SparseOrthogonalIMDPTarget()
    else
        target_model = OrthogonalIMDPTarget()
    end

    prob = AbstractionProblem(sys, spec)
    mdp, abstract_spec = abstraction(prob, state_abs, input_abs, target_model)

    upper_bound_spec = Specification(system_property(spec), !satisfaction_mode(spec))
    upper_bound_spec =
        IntervalMDPAbstractions.convert_specification(upper_bound_spec, state_abs, target_model)

    return mdp, abstract_spec, upper_bound_spec
end

function building_automation_system_4d_direct(
    time_horizon = 10;
    sparse = false,
    state_split = (5, 5, 7, 7),
    input_split = 4,
)
    sys, spec = building_automation_system_4d(time_horizon)

    X = Hyperrectangle(;
        low = [18.75, 18.75, 29.5, 29.5],
        high = [21.25, 21.25, 36.5, 36.5],
    )
    state_abs = StateUniformGridSplit(X, state_split)

    U = Hyperrectangle(; low = [17.0], high = [20.0])
    input_abs = InputLinRange(U, input_split)

    if sparse
        target_model = SparseIMDPTarget()
    else
        target_model = IMDPTarget()
    end

    prob = AbstractionProblem(sys, spec)
    mdp, abstract_spec = abstraction(prob, state_abs, input_abs, target_model)

    upper_bound_spec = Specification(system_property(spec), !satisfaction_mode(spec))
    upper_bound_spec =
        IntervalMDPAbstractions.convert_specification(upper_bound_spec, state_abs, target_model)

    return mdp, abstract_spec, upper_bound_spec
end

function main()
    @time "abstraction" mdp, spec, _ = building_automation_system_4d_decoupled(;
        state_split = (5, 5, 7, 7),
        input_split = 4,
    )
    prob = Problem(mdp, spec)

    @time "value iteration" V_safety, k, res = value_iteration(prob)

    # Remove the first state from each axis (the avoid state, whose value is always 0).
    V_safety = V_safety[(1:d-1 for d in size(V_safety))...]

    return V_safety
end
