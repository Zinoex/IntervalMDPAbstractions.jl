using LinearAlgebra, LazySets
using IntervalMDP, IntervalSySCoRe


function car_parking_sys(time_horizon)
    A = 0.9I(2)
    B = 0.7I(2)
    w_stddev = [1.0, 1.0]

    dyn = AffineAdditiveNoiseDynamics(A, B, AdditiveDiagonalGaussianNoise(w_stddev))

    initial_region = EmptySet(2)
    sys = System(dyn, initial_region)

    reach_region = Hyperrectangle(; low=[4.0, -4.0], high=[10.0, 0.0])
    avoid_region = Hyperrectangle(; low=[4.0, 0.0], high=[10.0, 4.0])
    prop = FiniteTimeRegionReachAvoid(reach_region, avoid_region, time_horizon)
    spec = Specification(prop, Pessimistic, Maximize)

    return sys, spec
end


function car_parking_decoupled(time_horizon=10; sparse=false, range_vs_grid=:grid, state_split=(20, 20), input_split=(3, 3))
    sys, spec = car_parking_sys(time_horizon)

    X = Hyperrectangle(; low=[-10.0, -10.0], high=[10.0, 10.0])
    state_abs = StateUniformGridSplit(X, state_split)

    U = Hyperrectangle(; low=[-1.0, -1.0], high=[1.0, 1.0])
    if range_vs_grid == :range
        input_abs = InputLinRange(U, input_split)
    elseif range_vs_grid == :grid
        input_abs = InputGridSplit(U, input_split)
    else
        throw(ArgumentError("Invalid range_vs_grid argument"))
    end

    if sparse
        target_model = SparseOrthogonalIMDPTarget()
    else
        target_model = OrthogonalIMDPTarget()
    end

    prob = AbstractionProblem(sys, spec)
    mdp, abstract_spec = abstraction(prob, state_abs, input_abs, target_model)

    upper_bound_spec = Specification(system_property(spec), !satisfaction_mode(spec))
    upper_bound_spec = IntervalSySCoRe.convert_specification(upper_bound_spec, state_abs, target_model)

    return mdp, abstract_spec, upper_bound_spec
end

function car_parking_direct(time_horizon=10; sparse=false, range_vs_grid=:grid, state_split=(20, 20), input_split=(3, 3))
    sys, spec = car_parking_sys(time_horizon)

    X = Hyperrectangle(; low=[-10.0, -10.0], high=[10.0, 10.0])
    state_abs = StateUniformGridSplit(X, state_split)

    U = Hyperrectangle(; low=[-1.0, -1.0], high=[1.0, 1.0])
    if range_vs_grid == :range
        input_abs = InputLinRange(U, input_split)
    elseif range_vs_grid == :grid
        input_abs = InputGridSplit(U, input_split)
    else
        throw(ArgumentError("Invalid range_vs_grid argument"))
    end

    if sparse
        target_model = SparseIMDPTarget()
    else
        target_model = IMDPTarget()
    end

    prob = AbstractionProblem(sys, spec)
    mdp, abstract_spec = abstraction(prob, state_abs, input_abs, target_model)

    upper_bound_spec = Specification(system_property(spec), !satisfaction_mode(spec))
    upper_bound_spec = IntervalSySCoRe.convert_specification(upper_bound_spec, state_abs, target_model)

    return mdp, abstract_spec, upper_bound_spec
end

function main()
    # Direct
    @time "abstraction direct" mdp_direct, spec_direct, _ = car_parking_direct()
    prob_direct = Problem(mdp_direct, spec_direct)

    @time "value iteration direct" V_direct, k_direct, res_direct = value_iteration(prob_direct)

    # Decoupled
    @time "abstraction decoupled" mdp_decoupled, spec_decoupled, _ = car_parking_decoupled()
    prob_decoupled = Problem(mdp_decoupled, spec_decoupled)

    @time "value iteration decoupled" V_decoupled, k_decoupled, res_decoupled = value_iteration(prob_decoupled)

    V_diff = V_decoupled[2:end, 2:end] - reshape(V_direct[2:end], 20, 20)

    return V_diff, V_decoupled[2:end, 2:end], reshape(V_direct[2:end], 20, 20)
end