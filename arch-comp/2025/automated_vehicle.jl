


function odimdp_av_ft_ra(state_split = (6, 6, 5, 5, 7, 5, 5), input_split = (5, 5))

    # Load the problem
    arch_comp_problem =
        ArchCompStochasticModels.automated_vehicle_finite_time_ra()
    arch_comp_system = arch_comp_problem.system
    arch_comp_spec = arch_comp_problem.specification

    # Define the abstraction
    Tx = ArchCompStochasticModels.transition_kernel(arch_comp_system)
    @assert Tx isa ArchCompStochasticModels.PiecewiseContinuousKernel

    @assert all(map(x -> x[1] isa LazySet, Tx.regions))
    @assert all(map(x -> x[2] isa ArchCompStochasticModels.DiagonalGaussianKernel, Tx.regions))
    @assert all(map(x -> x[2].mean isa Smooth2, Tx.regions))
    @assert all(map(x -> x[2].variance isa Vector, Tx.regions))

    var = first(Tx.regions)[2].variance
    @assert all(map(x -> x[2].variance == var, Tx.regions))
    stddev = sqrt.(var)
    noise = AdditiveDiagonalGaussianNoise(stddev)

    # This is a bit of a hack since LazySets doesn't play nice
    # with intersections between bounded and unbounded sets (regression).
    # We know the specific regions within the region of interest,
    # so we can just use them directly rather than calculate them.
    capped_regions = [
        Hyperrectangle(;
            low=[-12.0, -12.0, -0.5, -2.5, -0.35, -0.5, -0.05],
            high=[12.0, 12.0, 0.5, -0.1, 0.35, 0.5, 0.05],
        ),
        Hyperrectangle(;
            low=[-12.0, -12.0, -0.5, -0.1, -0.35, -0.5, -0.05],
            high=[12.0, 12.0, 0.5, 0.1, 0.35, 0.5, 0.05],
        ),
        Hyperrectangle(;
            low=[-12.0, -12.0, -0.5, 0.1, -0.35, -0.5, -0.05],
            high=[12.0, 12.0, 0.5, 2.5, 0.35, 0.5, 0.05],
        ),
    ]

    regions = map(Tx.regions) do (region, Tx_local)
        Tx_region = nothing
        for capped_region in capped_regions
            if center(capped_region) ∈ region
                Tx_region = capped_region
                break
            end
        end

        @assert Tx_region !== nothing

        f = Tx_local.mean.func
        NonlinearDynamicsRegion(f, Tx_region)
    end

    additive_dynamics = PiecewiseNonlinearAdditiveNoiseDynamics(
        regions,
        LazySets.dim(arch_comp_system.state_space),
        LazySets.dim(arch_comp_system.control_space),
        noise,
    )
    system = System(additive_dynamics)

    # Specification
    @assert arch_comp_spec isa ArchCompStochasticModels.ControllerSynthesisSpecification
    arch_comp_prop = arch_comp_spec.underlying_prop
    @assert arch_comp_prop isa ArchCompStochasticModels.FiniteTimeReachAvoidSpecification

    prop = InfiniteTimeRegionReachAvoid(arch_comp_prop.target_set, arch_comp_prop.avoid_set, arch_comp_prop.N)
    spec = Specification(
        prop,
        Pessimistic,
        synthesismode2strategymode(arch_comp_spec.synthesis_mode),
    )

    abs_problem = AbstractionProblem(system, spec)

    # Define abstraction parameters
    target_model = OrthogonalIMDPTarget()
    X = Hyperrectangle(;  # Add custom region of interest to _actually_ match AMYTISS.
        low=[-12.0, -12.0, -0.5, -2.5, -0.35, -0.5, -0.05],
        high=[12.0, 12.0, 0.5, 2.5, 0.35, 0.5, 0.05],
    )
    state_abs = StateUniformGridSplit(X, state_split)
    U = ArchCompStochasticModels.control_space(arch_comp_system)
    input_abs = InputLinRange(U, input_split)

    # Abstract and compute bounds - warmup is neglible for a problem this big.
    abstraction_time = @elapsed odimdp, lower_bound_spec = abstraction(abs_problem, state_abs, input_abs, target_model)
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
    time = (abstraction=abstraction_time, vi_lower=vi_lower_time, vi_upper=vi_upper_time, total=total_time)

    min_lb, max_lb, mean_lb = minimum(Vlower_nonterm), maximum(Vlower_nonterm), mean(Vlower_nonterm)
    lb = (min=min_lb, max=max_lb, mean=mean_lb)
    min_error, max_error, mean_error = minimum(error_nonterm), maximum(error_nonterm), mean(error_nonterm)
    error = (min=min_error, max=max_error, mean=mean_error)

    return (lb=lb, error=error, mem=mem_mb, time=time)
end