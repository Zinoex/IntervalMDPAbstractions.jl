using LinearAlgebra, LazySets
using IntervalMDP, IntervalSySCoRe

include("exact_time.jl")

function integrator_chain_sys(num_dims::Int; sampling_time=0.1)
    A = zeros(Float64, num_dims, num_dims)
    B = zeros(Float64, num_dims, 1)

    for i in 1:num_dims
        for j in i:num_dims
            n = j - i
            A[i, j] = sampling_time^n / factorial(n)
        end

        n = num_dims - i + 1
        B[i, 1] = sampling_time^n / factorial(n)
    end

    w_variance = [0.01 for _ in 1:num_dims]
    w_stddev = sqrt.(w_variance)

    dyn = AffineAdditiveNoiseDynamics(A, B, AdditiveDiagonalGaussianNoise(w_stddev))

    initial_region = EmptySet(num_dims)
    reach_region = Hyperrectangle(; low=[-8.0 for _ in 1:num_dims], high=[8.0 for _ in 1:num_dims])
    avoid_region = EmptySet(num_dims)

    sys = System(dyn, initial_region, reach_region, avoid_region)

    return sys
end

function integrator_chain_decoupled(num_dims::Int; state_split_per_dim=50, input_split=11)
    sys = integrator_chain_sys(num_dims)

    X = Hyperrectangle(; low=[-10.0 for _ in 1:num_dims], high=[10.0 for _ in 1:num_dims])
    state_abs = StateUniformGridSplit(X, Tuple(state_split_per_dim for _ in 1:num_dims))

    U = Hyperrectangle(; low=[-1.0], high=[1.0])
    input_abs = InputLinRange(U, input_split)

    target_model = DecoupledIMDP()

    mdp, reach, avoid = abstraction(sys, state_abs, input_abs, target_model)

    return mdp, reach, avoid
end

function main(n)
    @time "abstraction" mdp, reach, avoid = integrator_chain_decoupled(n)

    prop = ExactTimeReachAvoid(reach, avoid, 5)
    spec = Specification(prop, Pessimistic, Maximize)
    prob = Problem(mdp, spec)

    @time "value iteration" V, k, res = value_iteration(prob)

    # Remove the first state from each axis (the avoid state, whose value is always 0).
    V = V[(2:s for s in size(V))...]

    return V
end