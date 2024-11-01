using LinearAlgebra, LazySets
using IntervalMDP, IntervalSySCoRe

using NPZ

function load_mode(base_path, mode, regions)
    mean = npzread(joinpath(base_path, "mean_data_$(mode)_0.npy"))
    variance = npzread(joinpath(base_path, "sig_data_$(mode)_0.npy"))

    dyn = Vector{AbstractedGaussianProcessRegion{Float64, Vector{Float64}, eltype(regions)}}(undef, length(regions))

    for (i, region) in enumerate(regions)
        dyn[i] = AbstractedGaussianProcessRegion(
            region,
            convert.(Float64, mean[i, :, 1]),
            convert.(Float64, mean[i, :, 2]),
            convert.(Float64, variance[i, :, 1]),
            convert.(Float64, variance[i, :, 2])
        )
    end
end

function load_npy_dynamics(base_path, num_modes)
    region_extents = npzread(joinpath(base_path, "extents_0.npy"))

    regions = [
        Hyperrectangle(low=convert.(Float64, extent[:, 1]), high=convert.(Float64, extent[:, 2])) for extent
        in eachrow(region_extents)
    ]

    dyn = Vector{Vector{AbstractedGaussianProcessRegion{Float64, Vector{Float64}, eltype(regions)}}}(undef, num_modes)

    for action in 1:num_modes
        dyn[action] = load_mode(base_path, action, regions)
    end

    return dyn
end

function load_npy_system(system_name::String, num_modes)
    base_path = joinpath(@__DIR__, "gp_data/$(system_name)")
    dynamics = load_npy_dynamics(base_path, num_modes)

    return dynamics
end

function dubins_car_sys(time_horizon)
    pwa_dyn = load_npy_system("dubins_car_3d", 7)
    dyn = AbstractedGaussianProcessDynamics(pwa_dyn)

    initial = EmptySet(3)
    sys = System(dyn, initial)

    reach = Hyperrectangle(low=[8.0, 0.0, -0.5], high=[10.0, 1.0, 0.5])
    avoid = Hyperrectangle(low=[4.0, 0.0, -0.5], high=[6.0, 1.0, 0.5])
    prop = FiniteTimeRegionReachAvoid(reach, avoid, time_horizon)
    spec = Specification(prop, Pessimistic, Maximize)

    return sys, spec
end

function dubins_car_decoupled(time_horizon=10; sparse=false)
    sys, spec = dubins_car_sys(time_horizon)

    X = Hyperrectangle(; low=[0.0, 0.0, -0.5], high=[10.0, 2.0, 0.5])
    state_split = (80, 16, 20)
    state_abs = StateUniformGridSplit(X, state_split)

    input_abs = InputDiscrete([1, 2, 3, 4, 5, 6, 7])

    if sparse
        target_model = SparseOrthogonalIMDPTarget()
    else
        target_model = OrthogonalIMDPTarget()
    end

    prob = AbstractionProblem(sys, spec)
    mdp, abstract_spec = abstraction(prob, state_abs, input_abs, target_model)

    upper_bound_spec = Specification(system_property(spec), !satisfaction_mode(spec))
    upper_bound_spec = IntervalSySCoRe.convert_specification(upper_bound_spec, state_abs, target_model)

    return mdp, abstract_spec, upper_bound_spec
end

function dubins_car_direct(time_horizon=10; sparse=false)
    sys, spec = dubins_car_sys(time_horizon)

    X = Hyperrectangle(; low=[0.0, 0.0, -0.5], high=[10.0, 2.0, 0.5])
    state_split = (80, 16, 20)
    state_abs = StateUniformGridSplit(X, state_split)

    input_abs = InputDiscrete([1, 2, 3, 4, 5, 6, 7])

    if sparse
        target_model = SparseIMDPTarget()
    else
        target_model = IMDPTarget()
    end

    prob = AbstractionProblem(sys, spec)
    mdp, abstract_spec = abstraction(prob, state_abs, input_abs, target_model)

    upper_bound_spec = Specification(system_property(spec), !satisfaction_mode(spec))
    upper_bound_spec = IntervalSySCoRe.convert_specification(upper_bound_spec, state_abs, target_model)

    return mdp, abstract_spec, upper_bound_spec
end