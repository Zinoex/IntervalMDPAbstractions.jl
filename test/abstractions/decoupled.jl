using Revise, Test
using LinearAlgebra, LazySets
using IntervalMDP, IntervalSySCoRe

include("example_systems.jl")

function simple_1d_decoupled()
    sys = simple_1d_sys()

    X = Hyperrectangle(; low=[-2.5], high=[2.5])
    state_abs = StateUniformGridSplit(X, (10,))
    input_abs = InputDiscrete([Singleton([0.0])])
    target_model = DecoupledIMDP()

    mdp, reach, avoid = abstraction(sys, state_abs, input_abs, target_model)

    return mdp, reach, avoid
end

mdp_decoupled, reach_decoupled, avoid_decoupled = simple_1d_decoupled()
@test num_states(mdp_decoupled) == 11
@test stateptr(mdp_decoupled)[end] == 12

prop_decoupled = FiniteTimeReachAvoid(reach_decoupled, avoid_decoupled, 10)
spec_decoupled = Specification(prop_decoupled, Pessimistic, Maximize)
prob_decoupled = Problem(mdp_decoupled, spec_decoupled)

V_dense_grid, k, res = value_iteration(prob_decoupled)
@test k == 10

function modified_running_example_decoupled(; range_vs_grid=:grid)
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

    target_model = DecoupledIMDP()

    mdp, reach, avoid = abstraction(sys, state_abs, input_abs, target_model)

    return mdp, reach, avoid
end

# Input grid
mdp_decoupled, reach_decoupled, avoid_decoupled = modified_running_example_decoupled()
@test num_states(mdp_decoupled) == 121
@test stateptr(mdp_decoupled)[end] == 11 * 11 + 10 * 10 * 8 + 1

prop_decoupled = FiniteTimeReachAvoid(reach_decoupled, avoid_decoupled, 10)
spec_decoupled = Specification(prop_decoupled, Pessimistic, Maximize)
prob_decoupled = Problem(mdp_decoupled, spec_decoupled)

V_grid, k, res = value_iteration(prob_decoupled)
@test k == 10

# Input range
mdp_decoupled, reach_decoupled, avoid_decoupled = modified_running_example_decoupled(; range_vs_grid=:range)
@test num_states(mdp_decoupled) == 121
@test stateptr(mdp_decoupled)[end] == 11 * 11 + 10 * 10 * 8 + 1

prop_decoupled = FiniteTimeReachAvoid(reach_decoupled, avoid_decoupled, 10)
spec_decoupled = Specification(prop_decoupled, Pessimistic, Maximize)
prob_decoupled = Problem(mdp_decoupled, spec_decoupled)

V_range, k, res = value_iteration(prob_decoupled)
@test k == 10
@test all(V_range .≥ V_grid)
