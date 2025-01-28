using Revise, Test
using LinearAlgebra, LazySets
using IntervalMDP, IntervalMDPAbstractions

include("example_systems.jl")

function modified_running_example_compare()
    sys, spec = modified_running_example_sys()
    prob = AbstractionProblem(sys, spec)

    X = Hyperrectangle(; low = [-10.0, -10.0], high = [10.0, 10.0])
    state_abs = StateUniformGridSplit(X, (10, 10))

    U = Hyperrectangle(; low = [-1.0, -1.0], high = [1.0, 1.0])
    input_abs = InputGridSplit(U, [3, 3])

    target_model_direct = IMDPTarget()
    mdp_direct, abstract_spec_direct =
        abstraction(prob, state_abs, input_abs, target_model_direct)

    target_model_decoupled = OrthogonalIMDPTarget()
    mdp_decoupled, abstract_spec_decoupled =
        abstraction(prob, state_abs, input_abs, target_model_decoupled)

    return mdp_direct, abstract_spec_direct, mdp_decoupled, abstract_spec_decoupled
end

mdp_direct, spec_direct, mdp_decoupled, spec_decoupled = modified_running_example_compare()

# Value iteration
prob_direct = Problem(mdp_direct, spec_direct)

V_direct, k, res = value_iteration(prob_direct)
@test k == 10

prob_decoupled = Problem(mdp_decoupled, spec_decoupled)

V_decoupled, k, res = value_iteration(prob_decoupled)
@test k == 10
@test all(V_decoupled[1:end-1, 1:end-1] .≥ reshape(V_direct[1:end-1], 10, 10))
