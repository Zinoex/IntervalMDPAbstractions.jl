using LinearAlgebra, LazySets
using IntervalMDP, IntervalSySCoRe


function building_automation_system_4d()
    A = [
        0.6682 0.0    0.02632   0.0;
        0.0    0.683  0.0       0.02096;
        1.0005 0.0    -0.000499 0.0;
        0.0    0.8004 0.0       0.1996
    ]

    B = [
        0.1320;
        0.1402;
        0.0;
        0.0
    ]
    
    C = [3.4378, 2.9272, 13.0207, 10.4166]

    w_variance = [1/12.9199, 1/12.9199, 1/2.5826, 1/3.2276]
    w_stddev = sqrt.(w_variance)

    dyn = AffineAdditiveNoiseDynamics(A, B, C, AdditiveDiagonalGaussianNoise(w_stddev))

    initial_region = EmptySet(4)
    reach_region = EmptySet(4)
    avoid_region = EmptySet(4)

    sys = System(dyn, initial_region, reach_region, avoid_region)

    return sys
end

function building_automation_system_4d_decoupled(; sparse=false, state_split=(4, 4, 6, 6), input_split=4)
    sys = c()

    X = Hyperrectangle(; low=[19.0, 19.0, 30.0, 30.0], high=[21.0, 21.0, 36.0, 36.0])
    state_abs = StateUniformGridSplit(X, state_split)

    U = Hyperrectangle(; low=[17.0], high=[20.0])
    input_abs = InputLinRange(U, input_split)

    if sparse
        target_model = SparseOrthogonalIMDPTarget()
    else
        target_model = OrthogonalIMDPTarget()
    end

    mdp, reach, avoid = abstraction(sys, state_abs, input_abs, target_model)

    return mdp, reach, avoid
end

function main()
    @time "abstraction" mdp, _, avoid = building_automation_system_4d_decoupled(; state_split=(4, 4, 6, 6), input_split=4)

    prop = FiniteTimeReachability(avoid, 10)
    spec = Specification(prop, Optimistic, Minimize)
    prob = Problem(mdp, spec)

    @time "value iteration" V_unsafety, k, res = value_iteration(prob)
    V_safety = 1.0 .- V_unsafety
end