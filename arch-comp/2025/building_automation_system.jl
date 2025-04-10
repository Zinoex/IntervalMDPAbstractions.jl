

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
    arch_comp_prop = arch_comp_spec.underlying_spec

    prop = FiniteTimeRegionSafety(Complement(arch_comp_prop.safe_set), arch_comp_prop.N)
    spec = Specification(
        prop,
        Pessimistic,
        synthesismode2strategymode(arch_comp_spec.synthesis_mode),
    )

    abs_problem = AbstractionProblem(system, spec)

    # Define abstraction parameters
    safe_set_refinement = intersection(
        arch_comp_prop.safe_set,
        Hyperrectangle([19.0, 19.0, 30.0, 30.0], [21.0, 21.0, 36.0, 36.0]),
    )
    target_model = OrthogonalIMDPTarget()
    state_abs = StateUniformGridSplit(safe_set_refinement, state_split)
    input_abs = InputLinRange(U, input_split)

    # Abstract and compute lower bound - Warmup (and take values) then measure time
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

    # Compute necessary statistics
    total_time = abstraction_time + vi_lower_time + vi_upper_time
    maximum_lower_bound = maximum(Vlower)
    maximum_error = maximum(Vupper - Vlower)

    return maximum_lower_bound, maximum_error, total_time
end

function odimdp_bs_cs2_safety(state_split = (5, 5, 7, 7), input_split = (4,))
    arch_comp_problem = ArchCompStochasticModels.cs1_bas_finite_time_safety()
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
    arch_comp_prop = arch_comp_spec.underlying_spec

    prop = FiniteTimeRegionSafety(Complement(arch_comp_prop.safe_set), arch_comp_prop.N)
    spec = Specification(
        prop,
        Pessimistic,
        synthesismode2strategymode(arch_comp_spec.synthesis_mode),
    )

    abs_problem = AbstractionProblem(system, spec)

    # Define abstraction parameters
    safe_set_refinement = intersection(
        arch_comp_prop.safe_set,
        Hyperrectangle([19.0, 19.0, 30.0, 30.0], [21.0, 21.0, 36.0, 36.0]),
    )
    target_model = OrthogonalIMDPTarget()
    state_abs = StateUniformGridSplit(safe_set_refinement, state_split)
    input_abs = InputLinRange(U, input_split)

    # Abstract and compute lower bound - Warmup (and take values) then measure time
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

    # Compute necessary statistics
    total_time = abstraction_time + vi_lower_time + vi_upper_time
    maximum_lower_bound = maximum(Vlower)
    maximum_error = maximum(Vupper - Vlower)

    return maximum_lower_bound, maximum_error, total_time
end
