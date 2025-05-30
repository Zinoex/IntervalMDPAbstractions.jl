

function odimdp_ic_et_reach_avoid(n; state_split_per_dim = 20, input_split = (10,))

    # Load the problem
    arch_comp_problem = ArchCompStochasticModels.integrator_chain_exact_time_reachavoid(n)
    arch_comp_system = arch_comp_problem.system
    arch_comp_spec = arch_comp_problem.specification

    # Define the abstraction
    Tx = ArchCompStochasticModels.transition_kernel(arch_comp_system)
    stddev = sqrt.(Tx.variance)
    noise = AdditiveDiagonalGaussianNoise(stddev)

    U = ArchCompStochasticModels.control_space(arch_comp_system)

    @assert Tx.mean isa Linear2
    additive_dynamics = AffineAdditiveNoiseDynamics(Tx.mean.A, Tx.mean.B, noise)
    system = System(additive_dynamics)

    # Specification
    @assert arch_comp_spec isa
            ArchCompStochasticModels.ProbabilityGreaterThanInitialConditionSpecification
    inclusion_threshold = arch_comp_spec.threshold
    arch_comp_spec = arch_comp_spec.underlying_spec

    @assert arch_comp_spec isa ArchCompStochasticModels.ControllerSynthesisSpecification
    arch_comp_prop = arch_comp_spec.underlying_prop
    @assert arch_comp_prop isa ArchCompStochasticModels.ExactTimeReachAvoidSpecification

    # Encode avoid set into region of interest
    prop = ExactTimeRegionReachability(arch_comp_prop.target_set, arch_comp_prop.N)
    spec = Specification(
        prop,
        Pessimistic,
        synthesismode2strategymode(arch_comp_spec.synthesis_mode),
    )

    abs_problem = AbstractionProblem(system, spec)

    # Define abstraction parameters
    target_model = SparseOrthogonalIMDPTarget(1e-4)
    state_abs = StateUniformGridSplit(
        Complement(arch_comp_prop.avoid_set),
        ntuple(_ -> state_split_per_dim, n),
    )
    input_abs = InputLinRange(U, input_split)

    # Abstract and compute bounds; warmup then measure time.
    odimdp, lower_bound_spec = abstraction(abs_problem, state_abs, input_abs, target_model)
    abstraction_time = @elapsed abstraction(abs_problem, state_abs, input_abs, target_model)
    lower_bound_problem = Problem(odimdp, lower_bound_spec)

    @info "Abstraction constructed"

    Vlower, _, _ = value_iteration(lower_bound_problem)
    vi_lower_time = @elapsed value_iteration(lower_bound_problem)

    @info "Lower bound computed"

    # Measure memory usage
    mem_bytes = Base.summarysize(lower_bound_problem) + 2 * Base.summarysize(Vlower)
    mem_mb = mem_bytes / 1024^2

    # Remove avoid states - not reach states(!)
    Vlower = Vlower[(1:state_split_per_dim for _ = 1:n)...]

    # Compute necessary statistics
    total_time = abstraction_time + vi_lower_time
    time = (abstraction = abstraction_time, vi_lower = vi_lower_time, total = total_time)

    min_lb, max_lb, mean_lb = minimum(Vlower), maximum(Vlower), mean(Vlower)
    lb = (min = min_lb, max = max_lb, mean = mean_lb)

    volume = 0.0
    for (i, region) in enumerate(regions(state_abs))
        if Vlower[i] ≥ inclusion_threshold
            volume += lebesguemeasure(region)
        end
    end

    return (lb = lb, volume = volume, mem = mem_mb, time = time)
end

function lebesguemeasure(X::Hyperrectangle)
    return prod(w -> 2 * w, radius_hyperrectangle(X))
end
