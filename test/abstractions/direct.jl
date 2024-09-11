using Revise, Test
using LinearAlgebra, LazySets
using IntervalMDP, IntervalSySCoRe

include("example_systems.jl")

function simple_1d_direct(; sparse=false)
    sys = simple_1d_sys()

    X = Hyperrectangle(; low=[-2.5], high=[2.5])
    state_abs = StateUniformGridSplit(X, (10,))

    input_abs = InputDiscrete([Singleton([0.0])])

    if sparse
        target_model = SparseIMDPTarget()
    else
        target_model = IMDPTarget()
    end

    mdp, reach, avoid = abstraction(sys, state_abs, input_abs, target_model)

    return mdp, reach, avoid
end

# Dense
mdp_direct, reach_direct, avoid_direct = simple_1d_direct()
@test num_states(mdp_direct) == 11
@test stateptr(mdp_direct)[end] == 12

prop_direct = FiniteTimeReachAvoid(reach_direct, avoid_direct, 10)
spec_direct = Specification(prop_direct, Pessimistic, Maximize)
prob_direct = Problem(mdp_direct, spec_direct)

V_dense_grid, k, res = value_iteration(prob_direct)
@test k == 10

# Sparse
mdp_direct, reach_direct, avoid_direct = simple_1d_direct(; sparse=true)
@test num_states(mdp_direct) == 11
@test stateptr(mdp_direct)[end] == 12

prop_direct = FiniteTimeReachAvoid(reach_direct, avoid_direct, 10)
spec_direct = Specification(prop_direct, Pessimistic, Maximize)
prob_direct = Problem(mdp_direct, spec_direct)

V_sparse_grid, k, res = value_iteration(prob_direct)
@test k == 10
@test all(V_dense_grid .≥ V_sparse_grid)

function modified_running_example_direct(; sparse=false, range_vs_grid=:grid)
    sys = modified_running_example_sys()

    X = Hyperrectangle(; low=[-10.0, -10.0], high=[10.0, 10.0])
    state_abs = StateUniformGridSplit(X, (10, 10))

    U = Hyperrectangle(; low=[-1.0, -1.0], high=[1.0, 1.0])
    if range_vs_grid == :range
        input_abs = InputLinRange(U, [3, 3])
    elseif range_vs_grid == :grid
        input_abs = InputGridSplit(U, [3, 3])
    else
        throw(ArgumentError("Invalid range_vs_grid argument"))
    end

    if sparse
        target_model = SparseIMDPTarget()
    else
        target_model = IMDPTarget()
    end

    mdp, reach, avoid = abstraction(sys, state_abs, input_abs, target_model)

    return mdp, reach, avoid
end

# Dense, input grid
mdp_direct, reach_direct, avoid_direct = modified_running_example_direct()
@test num_states(mdp_direct) == 101
@test stateptr(mdp_direct)[end] == 10 * 10 * 9 + 2

prop_direct = FiniteTimeReachAvoid(reach_direct, avoid_direct, 10)
spec_direct = Specification(prop_direct, Pessimistic, Maximize)
prob_direct = Problem(mdp_direct, spec_direct)

V_dense_grid, k, res = value_iteration(prob_direct)
@test k == 10

# Sparse, input grid
mdp_direct, reach_direct, avoid_direct = modified_running_example_direct(; sparse=true)
@test num_states(mdp_direct) == 101
@test stateptr(mdp_direct)[end] == 10 * 10 * 9 + 2

prop_direct = FiniteTimeReachAvoid(reach_direct, avoid_direct, 10)
spec_direct = Specification(prop_direct, Pessimistic, Maximize)
prob_direct = Problem(mdp_direct, spec_direct)

V_sparse_grid, k, res = value_iteration(prob_direct)
@test k == 10
@test all(V_dense_grid .≥ V_sparse_grid)

# Dense, input range
mdp_direct, reach_direct, avoid_direct = modified_running_example_direct(; range_vs_grid=:range)
@test num_states(mdp_direct) == 101
@test stateptr(mdp_direct)[end] == 10 * 10 * 9 + 2

prop_direct = FiniteTimeReachAvoid(reach_direct, avoid_direct, 10)
spec_direct = Specification(prop_direct, Pessimistic, Maximize)
prob_direct = Problem(mdp_direct, spec_direct)

V_dense_range, k, res = value_iteration(prob_direct)
@test k == 10
@test all(V_dense_range .≥ V_dense_grid)
