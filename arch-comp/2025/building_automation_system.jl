

function odimdp_bs_cs1_safety(state_split = (5, 5, 7, 7), input_split = (4,))
    arch_comp_problem = ArchCompStochasticModels.cs1_bas_finite_time_safety()
    arch_comp_system = arch_comp_problem.system
    arch_comp_spec = arch_comp_problem.specification

    # Define the abstraction
    Tx = ArchCompStochasticModels.transition_kernel(arch_comp_system)
    stddev = sqrt.(Tx.variance)
    noise = AdditiveDiagonalGaussianNoise(stddev)

    U = ArchCompStochasticModels.control_space(arch_comp_system)

    @assert Tx.mean isa Affine2
    additive_dynamics = AffineAdditiveNoiseDynamics(Tx.mean.A, Tx.mean.B, Tx.mean.c, noise)
    system = System(additive_dynamics)

    # Specification
    @assert arch_comp_spec isa ArchCompStochasticModels.ControllerSynthesisSpecification
    arch_comp_prop = arch_comp_spec.underlying_prop
    @assert arch_comp_prop isa ArchCompStochasticModels.FiniteTimeSafetySpecification

    prop = FiniteTimeRegionSafety(Complement(arch_comp_prop.safe_set), arch_comp_prop.N)
    spec = Specification(
        prop,
        Pessimistic,
        synthesismode2strategymode(arch_comp_spec.synthesis_mode),
    )

    abs_problem = AbstractionProblem(system, spec)

    # Define abstraction parameters (box_approximation is going to be exact)
    # safe_set_refinement = box_approximation(Intersection(
    #     Hyperrectangle(;low=[19.5, 19.5, 30.0, 30.0], high=[20.5, 20.5, 36.0, 36.0]),
    #     arch_comp_prop.safe_set,
    # ))
    safe_set_refinement =
        Hyperrectangle(; low = [19.5, 19.5, 30.0, 30.0], high = [20.5, 20.5, 36.0, 36.0])
    @assert safe_set_refinement ⊆ arch_comp_prop.safe_set
    target_model = OrthogonalIMDPTarget()
    state_abs = StateUniformGridSplit(safe_set_refinement, state_split)
    input_abs = InputLinRange(U, input_split)

    # Abstract and compute bounds; warmup then measure time.
    odimdp, lower_bound_spec = abstraction(abs_problem, state_abs, input_abs, target_model)
    abstraction_time = @elapsed abstraction(abs_problem, state_abs, input_abs, target_model)
    lower_bound_problem = Problem(odimdp, lower_bound_spec)

    @info "Abstraction constructed"

    policy, Vlower, k, res = control_synthesis(lower_bound_problem)
    vi_lower_time = @elapsed value_iteration(lower_bound_problem)

    @info "Lower bound computed"

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

    @info "Upper bound computed"

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
    time = (
        abstraction = abstraction_time,
        vi_lower = vi_lower_time,
        vi_upper = vi_upper_time,
        total = total_time,
    )

    min_lb, max_lb, mean_lb =
        minimum(Vlower_nonterm), maximum(Vlower_nonterm), mean(Vlower_nonterm)
    lb = (min = min_lb, max = max_lb, mean = mean_lb)
    min_error, max_error, mean_error =
        minimum(error_nonterm), maximum(error_nonterm), mean(error_nonterm)
    error = (min = min_error, max = max_error, mean = mean_error)

    return (lb = lb, error = error, mem = mem_mb, time = time)
end

function odimdp_bs_cs2_safety(state_split = (5, 6, 6, 6, 6, 6, 6), input_split = (4,))
    arch_comp_problem = ArchCompStochasticModels.cs2_bas_finite_time_safety()
    arch_comp_system = arch_comp_problem.system
    arch_comp_spec = arch_comp_problem.specification

    # Define the abstraction
    Tx = ArchCompStochasticModels.transition_kernel(arch_comp_system)
    noise = AdditiveGaussianNoise(Tx.covariance)

    U = ArchCompStochasticModels.control_space(arch_comp_system)

    @assert Tx.mean isa Affine2
    additive_dynamics = AffineAdditiveNoiseDynamics(Tx.mean.A, Tx.mean.B, Tx.mean.c, noise)
    system = System(additive_dynamics)

    # Specification
    @assert arch_comp_spec isa ArchCompStochasticModels.ControllerSynthesisSpecification
    arch_comp_prop = arch_comp_spec.underlying_prop
    @assert arch_comp_prop isa ArchCompStochasticModels.FiniteTimeSafetySpecification

    avoid_set = Complement(arch_comp_prop.safe_set)
    # Since the safe set is a hyperrectangle, then just encode safety
    # via the region of interest.
    avoid_set = EmptySet(LazySets.dim(avoid_set))
    prop = FiniteTimeRegionSafety(avoid_set, arch_comp_prop.N)
    spec = Specification(
        prop,
        Pessimistic,
        synthesismode2strategymode(arch_comp_spec.synthesis_mode),
    )

    abs_problem = AbstractionProblem(system, spec)


    # Define abstraction parameters (box_approximation is going to be exact)
    # safe_set_refinement = box_approximation(Intersection(
    #     Hyperrectangle(;low=[19.5, 19.0, 18.0, 18.0, 18.0, 18.0, 18.0], high=[20.5, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0]),
    #     arch_comp_prop.safe_set,
    # ))
    safe_set_refinement = Hyperrectangle(;
        low = [19.5, 19.0, 18.0, 18.0, 18.0, 18.0, 18.0],
        high = [20.5, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0],
    )
    @assert safe_set_refinement ⊆ arch_comp_prop.safe_set
    target_model = OrthogonalIMDPTarget()
    state_abs = StateUniformGridSplit(safe_set_refinement, state_split)
    input_abs = InputLinRange(U, input_split)

    # Abstract and compute bounds - warmup is neglible for a problem this big.
    abstraction_time = @elapsed odimdp, lower_bound_spec =
        abstraction(abs_problem, state_abs, input_abs, target_model)
    lower_bound_problem = Problem(odimdp, lower_bound_spec)

    @info "Abstraction constructed"

    vi_lower_time = @elapsed policy, Vlower, k, res = control_synthesis(lower_bound_problem)

    @info "Lower bound computed"

    # Compute upper bound
    upper_bound_spec = Specification(system_property(spec), !satisfaction_mode(spec))
    upper_bound_spec = IntervalMDPAbstractions.convert_specification(
        upper_bound_spec,
        state_abs,
        target_model,
    )
    upper_bound_problem = Problem(odimdp, upper_bound_spec, policy)
    vi_upper_time = @elapsed Vupper, k, res, = value_iteration(upper_bound_problem)

    @info "Upper bound computed"

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
    time = (
        abstraction = abstraction_time,
        vi_lower = vi_lower_time,
        vi_upper = vi_upper_time,
        total = total_time,
    )

    min_lb, max_lb, mean_lb =
        minimum(Vlower_nonterm), maximum(Vlower_nonterm), mean(Vlower_nonterm)
    lb = (min = min_lb, max = max_lb, mean = mean_lb)
    min_error, max_error, mean_error =
        minimum(error_nonterm), maximum(error_nonterm), mean(error_nonterm)
    error = (min = min_error, max = max_error, mean = mean_error)

    return (lb = lb, error = error, mem = mem_mb, time = time)
end
