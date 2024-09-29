using LinearAlgebra, LazySets
using IntervalMDP, IntervalSySCoRe


function almost_identity_sys(num_dims::Int)
    A = zeros(Float64, num_dims, num_dims)
    B = zeros(Float64, num_dims, 1)

    for i in 1:num_dims
        A[i, i] = 0.7
        A[i, mod1(i + 1, num_dims)] = -0.1

        # B[i, i] = 1.0
    end

    w_variance = [0.01 for _ in 1:num_dims]
    w_stddev = sqrt.(w_variance)

    dyn = AffineAdditiveNoiseDynamics(A, B, AdditiveDiagonalGaussianNoise(w_stddev))

    initial_region = EmptySet(num_dims)
    reach_region = EmptySet(num_dims)
    avoid_region = EmptySet(num_dims)

    sys = System(dyn, initial_region, reach_region, avoid_region)

    return sys
end

function almost_identity_decoupled(num_dims::Int; sparse=false, state_split_per_dim=8)
    sys = almost_identity_sys(num_dims)

    X = Hyperrectangle(; low=[-1.0 for _ in 1:num_dims], high=[1.0 for _ in 1:num_dims])
    state_abs = StateUniformGridSplit(X, ntuple(i -> state_split_per_dim, num_dims))

    input_abs = InputDiscrete([Singleton([0.0])])

    if sparse
        target_model = SparseOrthogonalIMDPTarget(1e-3)
    else
        target_model = OrthogonalIMDPTarget()
    end

    mdp, reach, avoid = abstraction(sys, state_abs, input_abs, target_model)

    return mdp, reach, avoid
end

function main(n)
    @time "abstraction" mdp, reach, avoid = almost_identity_decoupled(n; sparse=true)

    println("Memory usage: $(Base.summarysize(mdp) / 1000^2) MB")

    prop = FiniteTimeReachability(avoid, 10)
    spec = Specification(prop, Optimistic, Minimize)
    prob = Problem(mdp, spec)

    @time "value iteration" V, k, res = value_iteration(prob)

    # Remove the first state from each axis (the avoid state, whose value is always 0).
    V = V[(2:s for s in size(V))...]

    return 1.0 .- V
end