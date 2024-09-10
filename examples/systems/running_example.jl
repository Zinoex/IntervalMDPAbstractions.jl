using LinearAlgebra, LazySets
using IntervalMDP, IntervalSySCoRe


function running_example_sys()
    A = 0.9I(2)
    B = 0.7I(2)
    w_stddev = [1.0, 1.0]

    dyn = AffineAdditiveNoiseDynamics(A, B, AdditiveDiagonalGaussianNoise(w_stddev))

    initial_region = EmptySet(2)
    reach_region = Hyperrectangle(; low=[4.0, -4.0], high=[10.0, 0.0])
    avoid_region = Hyperrectangle(; low=[4.0, 0.0], high=[10.0, 4.0])

    sys = System(dyn, initial_region, reach_region, avoid_region)

    return sys
end


function running_example_decoupled(;range_vs_grid=:grid, state_split=(20, 20), input_split=(3, 3))
    sys = running_example_sys()

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

    target_model = DecoupledIMDP()

    mdp, reach, avoid = abstraction(sys, state_abs, input_abs, target_model)

    return mdp, reach, avoid
end

function running_example_direct(; sparse=false, range_vs_grid=:grid, state_split=(20, 20), input_split=(3, 3))
    sys = running_example_sys()

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
        target_model = SparseDirectIMDP()
    else
        target_model = DirectIMDP()
    end

    mdp, reach, avoid = abstraction(sys, state_abs, input_abs, target_model)

    return mdp, reach, avoid
end

function main()
    # Direct
    @time "abstraction direct" mdp_direct, reach_direct, avoid_direct = running_example_direct()
    prop_direct = FiniteTimeReachAvoid(reach_direct, avoid_direct, 10)
    spec_direct = Specification(prop_direct, Pessimistic, Maximize)
    prob_direct = Problem(mdp_direct, spec_direct)

    @time "value iteration direct" V_direct, k_direct, res_direct = value_iteration(prob_direct)

    # Decoupled
    @time "abstraction decoupled" mdp_decoupled, reach_decoupled, avoid_decoupled = running_example_decoupled()
    prop_decoupled = FiniteTimeReachAvoid(reach_decoupled, avoid_decoupled, 10)
    spec_decoupled = Specification(prop_decoupled, Pessimistic, Maximize)
    prob_decoupled = Problem(mdp_decoupled, spec_decoupled)

    @time "value iteration decoupled" V_decoupled, k_decoupled, res_decoupled = value_iteration(prob_decoupled)

    V_diff = V_decoupled[2:end, 2:end] - reshape(V_direct[2:end], 20, 20)

    return V_diff, V_decoupled[2:end, 2:end], reshape(V_direct[2:end], 20, 20)
end