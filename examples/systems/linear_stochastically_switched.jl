using LinearAlgebra, LazySets
using IntervalMDP, IntervalSySCoRe


function linear_stochastically_switched_sys(time_horizon)
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

    reach_region = Hyperrectangle(; low = [-1.0, -1.0], high = [0.0, 1.0])
    avoid_region = Hyperrectangle(; low = [1.0, 0.0], high = [2.0, 1.0])
    prop = FiniteTimeRegionReachAvoid(reach_region, avoid_region, time_horizon)
    spec = Specification(prop, Pessimistic, Maximize)

    return sys, spec
end

function linear_stochastically_switched_direct(
    time_horizon = 10;
    sparse = false,
    state_split = (40, 40),
)
    sys, spec = linear_stochastically_switched_sys(time_horizon)

    X = Hyperrectangle(; low = [-2.0, -2.0], high = [2.0, 2.0])
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
    upper_bound_spec =
        IntervalSySCoRe.convert_specification(upper_bound_spec, state_abs, target_model)

    return mdp, abstract_spec, upper_bound_spec
end

function linear_stochastically_switched_mixture(
    time_horizon = 10;
    sparse = false,
    state_split = (40, 40),
)
    sys, spec = linear_stochastically_switched_sys(time_horizon)

    X = Hyperrectangle(; low = [-2.0, -2.0], high = [2.0, 2.0])
    state_abs = StateUniformGridSplit(X, state_split)

    input_abs = InputDiscrete([Singleton([0.0])])

    if sparse
        target_model = SparseMixtureIMDPTarget()
    else
        target_model = MixtureIMDPTarget()
    end

    prob = AbstractionProblem(sys, spec)
    mdp, abstract_spec = abstraction(prob, state_abs, input_abs, target_model)

    upper_bound_spec = Specification(system_property(spec), !satisfaction_mode(spec))
    upper_bound_spec =
        IntervalSySCoRe.convert_specification(upper_bound_spec, state_abs, target_model)

    return mdp, abstract_spec, upper_bound_spec
end

function main()
    @time "abstraction" mdp, spec, upper_bound_spec =
        linear_stochastically_switched_mixture(; state_split = (40, 40))
    prob = Problem(mdp, spec)

    @time "control synthesis" strategy, V_lower, k, res = control_synthesis(prob)

    upper_bound_prob = Problem(mdp, upper_bound_spec, strategy)
    @time "upper bound" V_upper, _, _ = value_iteration(upper_bound_prob)

    # Remove the first state from each axis (the avoid state, whose value is always 0).
    V_lower = V_lower[(2:d for d in size(V_lower))...]
    V_upper = V_upper[(2:d for d in size(V_upper))...]

    return V_lower, V_upper
end
