using LinearAlgebra, LazySets
using IntervalMDP, IntervalSySCoRe


function building_automation_system()
    # TODO: Add parameters and fill in A and B, w_stddev, and check that the dynamics are of the correct form.

    A = zeros(Float64, 7, 7)
    A[1, 1] = 1.0
    A[2, 2] = 1.0

    B = zeros(Float64, 7, 7)
    B[1, 1] = 10.0
    B[2, 2] = 10.0

    w_variance = [0.75, 0.75]
    w_stddev = sqrt.(w_variance)

    dyn = LinearAdditiveNoiseDynamics(A, B, AdditiveDiagonalGaussianNoise(w_stddev))

    initial_region = EmptySet(7)
    reach_region = EmptySet(7)
    avoid_region = EmptySet(7)

    sys = System(dyn, initial_region, reach_region, avoid_region)

    return sys
end

function building_automation_system_decoupled(; state_split=(10, 10, 10, 10, 10, 10, 10), input_split=20)
    sys = c()

    X = Hyperrectangle(; low=[19.5, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0], high=[20.5, 20.0, 30.0, 30.0, 30.0, 30.0, 30.0])
    state_abs = StateUniformGridSplit(X, state_split)

    U = Hyperrectangle(; low=[15.0], high=[30.0])
    input_abs = Input(U, input_split)

    target_model = DecoupledIMDP()

    mdp, reach, avoid = abstraction(sys, state_abs, input_abs, target_model)

    return mdp, reach, avoid
end

function main()
    mdp, _, avoid = building_automation_system_decoupled(; state_split=(10, 10, 10, 10, 10, 10, 10), input_split=20)

    prop = FiniteTimeReachability(avoid, 6)
    spec = Specification(prop, Optimistic, Minimize)
    prob = Problem(mdp, spec)

    V_unsafety, k, res = value_iteration(prob)
    V_safety = 1.0 .- V_unsafety
end