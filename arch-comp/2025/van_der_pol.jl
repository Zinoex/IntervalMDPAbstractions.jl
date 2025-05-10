

function odimdp_vp_gauss_quantitative(state_split = (50, 50), input_split = (10,))
    arch_comp_problem = ArchCompStochasticModels.controlled_gaussian_van_der_pol_quantitative()
    arch_comp_system = arch_comp_problem.system
    arch_comp_spec = arch_comp_problem.specification

    # Define the abstraction
    Tx = ArchCompStochasticModels.transition_kernel(arch_comp_system)
    stddev = sqrt.(Tx.variance)
    noise = AdditiveDiagonalGaussianNoise(stddev)

    U = ArchCompStochasticModels.control_space(arch_comp_system)
    Usub = Hyperrectangle(; low = [-1.0], high = [1.0])

    @assert Tx.mean isa Smooth2
    additive_dynamics = NonlinearAdditiveNoiseDynamics(Tx.mean.func, IntervalMDPAbstractions.dim(noise), LazySets.dim(U), noise)
    system = System(additive_dynamics)

    # Specification
    @assert arch_comp_spec isa ArchCompStochasticModels.ControllerSynthesisSpecification
    arch_comp_prop = arch_comp_spec.underlying_prop
    @assert arch_comp_prop isa ArchCompStochasticModels.InfiniteTimeReachAvoidSpecification

    # Encode avoid set as through the region of interest (transitioning to outside the region of interest
    # is treated as transitioning to a failure/sink state)
    prop = InfiniteTimeRegionReachability(arch_comp_prop.target_set, arch_comp_prop.convergence_threshold)
    spec = Specification(
        prop,
        Pessimistic,
        synthesismode2strategymode(arch_comp_spec.synthesis_mode),
    )

    abs_problem = AbstractionProblem(system, spec)

    # Define abstraction parameters
    region_of_interest = Complement(arch_comp_prop.avoid_set)
    target_model = OrthogonalIMDPTarget()
    state_abs = StateUniformGridSplit(region_of_interest, state_split)
    input_abs = InputLinRange(Usub, input_split)

    # Abstract and compute lower bound, then warmup and measure time.
    odimdp, lower_bound_spec = abstraction(abs_problem, state_abs, input_abs, target_model)
    abstraction_time = @elapsed abstraction(abs_problem, state_abs, input_abs, target_model)
    lower_bound_problem = Problem(odimdp, lower_bound_spec)

    policy, Vlower, k, res = control_synthesis(lower_bound_problem)
    vi_lower_time = @elapsed value_iteration(lower_bound_problem)

    # Compute upper bound
    upper_bound_spec = Specification(system_property(spec), !satisfaction_mode(spec))
    upper_bound_spec = IntervalMDPAbstractions.convert_specification(
        upper_bound_spec,
        state_abs,
        target_model,
    )
    upper_bound_problem = Problem(odimdp, upper_bound_spec, policy)
    Vupper, k, res, = value_iteration(upper_bound_problem)
    vi_upper_time = @elapsed value_iteration(upper_bound_problem)

    # Measure memory usage
    mem_bytes = Base.summarysize(upper_bound_problem) + 2 * Base.summarysize(Vupper)
    mem_mb = mem_bytes / 1024^2

    # Terminal states
    # The terminal states may not match between the upper and lower bound spec due to the non-alignment
    # with the gridding. The important set of terminal states for the statistics are the lower bound
    # terminal states.
    tstates = terminal_states(system_property(lower_bound_spec))  
    Vlower_nonterm = Vlower[Not(tstates)]
    Vupper_nonterm = Vupper[Not(tstates)]
    error_nonterm = Vupper_nonterm - Vlower_nonterm

    # Compute necessary statistics
    total_time = abstraction_time + vi_lower_time + vi_upper_time

    min_lb, max_lb, mean_lb = minimum(Vlower_nonterm), maximum(Vlower_nonterm), mean(Vlower_nonterm)
    lb = (min=min_lb, max=max_lb, mean=mean_lb)
    min_error, max_error, mean_error = minimum(error_nonterm), maximum(error_nonterm), mean(error_nonterm)
    error = (min=min_error, max=max_error, mean=mean_error)

    return (lb=lb, error=error, mem=mem_mb, time=total_time)
end