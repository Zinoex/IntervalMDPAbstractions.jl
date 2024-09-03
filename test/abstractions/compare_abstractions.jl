using Revise, Test
using LinearAlgebra, LazySets
using IntervalMDP, IntervalSySCoRe

include("example_systems.jl")

function modified_running_example_compare()
    sys = modified_running_example_sys()

    X = Hyperrectangle(; low=[-10.0, -10.0], high=[10.0, 10.0])
    state_abs = StateUniformGridSplit(X, (10, 10))

    U = Hyperrectangle(; low=[-1.0, -1.0], high=[1.0, 1.0])
    input_abs = InputGridSplit(U, [3, 3])

    target_model_direct = DirectIMDP()
    mdp_direct, reach_direct, avoid_direct = abstraction(sys, state_abs, input_abs, target_model_direct)

    target_model_decoupled = DecoupledIMDP()
    mdp_decoupled, reach_decoupled, avoid_decoupled = abstraction(sys, state_abs, input_abs, target_model_decoupled)

    return mdp_direct, reach_direct, avoid_direct, mdp_decoupled, reach_decoupled, avoid_decoupled
end

mdp_direct, reach_direct, avoid_direct, mdp_decoupled, reach_decoupled, avoid_decoupled = modified_running_example_compare()

# Value iteration
prop_direct = FiniteTimeReachAvoid(reach_direct, avoid_direct, 10)
spec_direct = Specification(prop_direct, Pessimistic, Maximize)
prob_direct = Problem(mdp_direct, spec_direct)

V_direct, k, res = value_iteration(prob_direct)
@test k == 10

prop_decoupled = FiniteTimeReachAvoid(reach_decoupled, avoid_decoupled, 10)
spec_decoupled = Specification(prop_decoupled, Pessimistic, Maximize)
prob_decoupled = Problem(mdp_decoupled, spec_decoupled)

V_decoupled, k, res = value_iteration(prob_decoupled)
@test k == 10
@test all(V_decoupled[2:end, 2:end] .≥ reshape(V_direct[2:end], 10, 10))
