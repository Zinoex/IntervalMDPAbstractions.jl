using LinearAlgebra, LazySets
using IntervalMDP, IntervalMDPAbstractions


function big_sys(num_dims::Int, time_horizon)
    A = zeros(Float64, num_dims, num_dims)
    B = zeros(Float64, num_dims, 1)

    for i = 1:num_dims
        A[i, i] = 0.8
    end

    w_variance = [0.2 for _ = 1:num_dims]
    w_stddev = sqrt.(w_variance)

    dyn = AffineAdditiveNoiseDynamics(A, B, AdditiveDiagonalGaussianNoise(w_stddev))

    initial_region = EmptySet(num_dims)
    sys = System(dyn, initial_region)

    avoid_region = EmptySet(num_dims)
    prop = FiniteTimeRegionSafety(avoid_region, time_horizon)
    spec = Specification(prop, Pessimistic, Maximize)

    return sys, spec
end

function big_decoupled(
    num_dims::Int,
    time_horizon = 10;
    sparse = false,
    state_split_per_dim = 2,
)
    sys, spec = big_sys(num_dims, time_horizon)

    X = Hyperrectangle(; low = [-1.0 for _ = 1:num_dims], high = [1.0 for _ = 1:num_dims])
    state_abs = StateUniformGridSplit(X, ntuple(i -> state_split_per_dim, num_dims))

    input_abs = InputDiscrete([Singleton([0.0])])

    if sparse
        target_model = SparseOrthogonalIMDPTarget(1e-3)
    else
        target_model = OrthogonalIMDPTarget()
    end

    prob = AbstractionProblem(sys, spec)
    mdp, abstract_spec = abstraction(prob, state_abs, input_abs, target_model)

    upper_bound_spec = Specification(system_property(spec), !satisfaction_mode(spec))
    upper_bound_spec = IntervalMDPAbstractions.convert_specification(
        upper_bound_spec,
        state_abs,
        target_model,
    )

    return mdp, abstract_spec, upper_bound_spec
end

function big_direct(
    num_dims::Int,
    time_horizon = 10;
    sparse = false,
    state_split_per_dim = 2,
)
    sys, spec = big_sys(num_dims, time_horizon)

    X = Hyperrectangle(; low = [-1.0 for _ = 1:num_dims], high = [1.0 for _ = 1:num_dims])
    state_abs = StateUniformGridSplit(X, ntuple(i -> state_split_per_dim, num_dims))

    input_abs = InputDiscrete([Singleton([0.0])])

    if sparse
        target_model = SparseIMDPTarget(1e-9)
    else
        target_model = IMDPTarget()
    end

    prob = AbstractionProblem(sys, spec)
    mdp, abstract_spec = abstraction(prob, state_abs, input_abs, target_model)

    upper_bound_spec = Specification(system_property(spec), !satisfaction_mode(spec))
    upper_bound_spec = IntervalMDPAbstractions.convert_specification(
        upper_bound_spec,
        state_abs,
        target_model,
    )

    return mdp, abstract_spec, upper_bound_spec
end

function small_sys(time_horizon)
    A = zeros(Float64, 1, 1)
    B = zeros(Float64, 1, 1)

    A[1, 1] = 0.8

    w_variance = [0.2]
    w_stddev = sqrt.(w_variance)

    dyn = AffineAdditiveNoiseDynamics(A, B, AdditiveDiagonalGaussianNoise(w_stddev))

    initial_region = EmptySet(1)
    sys = System(dyn, initial_region)

    avoid_region = EmptySet(1)
    prop = FiniteTimeRegionSafety(avoid_region, time_horizon)
    spec = Specification(prop, Pessimistic, Maximize)

    return sys, spec
end

function small_direct(time_horizon = 10; sparse = false, state_split_per_dim = 2)
    sys, spec = small_sys(time_horizon)

    X = Hyperrectangle(; low = [-1.0], high = [1.0])
    state_abs = StateUniformGridSplit(X, (state_split_per_dim,))

    input_abs = InputDiscrete([Singleton([0.0])])

    if sparse
        target_model = SparseIMDPTarget(1e-9)
    else
        target_model = IMDPTarget()
    end

    prob = AbstractionProblem(sys, spec)
    mdp, abstract_spec = abstraction(prob, state_abs, input_abs, target_model)

    upper_bound_spec = Specification(system_property(spec), !satisfaction_mode(spec))
    upper_bound_spec = IntervalMDPAbstractions.convert_specification(
        upper_bound_spec,
        state_abs,
        target_model,
    )

    return mdp, abstract_spec, upper_bound_spec
end

function main(n, time_horizon = 10)
    @time "abstraction" mdp, spec, _ = big_decoupled(n, time_horizon; sparse = false)

    println("Memory usage: $(Base.summarysize(mdp) / 1000^2) MB")

    prob = Problem(mdp, spec)

    @time "value iteration" V, k, res = value_iteration(prob)

    # Remove the first state from each axis (the avoid state, whose value is always 0).
    V = V[(1:d-1 for d in size(V))...]

    return V
end
