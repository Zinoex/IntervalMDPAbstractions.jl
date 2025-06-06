using JuMP, HiGHS

export theorem1_abstraction, theorem2_abstraction

function theorem1_abstraction(
    prob::AbstractionProblem{S},
    state_abstraction::StateUniformGridSplit,
    input_abstraction::InputAbstraction,
    target_model::AbstractIMDPTarget
) where {S <: System{<:AffineAdditiveNoiseDynamics}}
    sys = system(prob)
    spec = specification(prob)

    if !islinear(dynamics(sys))
        throw(ArgumentError("The system dynamics must be linear for Theorem 1."))
    end

    # State pointer
    stateptr = Int32[[1]
                    (1:numregions(state_abstraction)) .* numinputs(input_abstraction) .+ 1]

    # Transition probabilities
    interval_prob = theorem1_transition_prob(
        dynamics(sys),
        state_abstraction,
        input_abstraction,
        target_model
    )

    # Initial states
    initial_states = Int32[]
    for (i, source_region) in enumerate(regions(state_abstraction))
        if !iszeromeasure(initial(sys), source_region)
            push!(initial_states, i)
        end
    end

    mdp = IntervalMarkovDecisionProcess(interval_prob, stateptr, initial_states)

    # Property
    spec = convert_specification(spec, state_abstraction, target_model)

    return mdp, spec
end

function theorem1_transition_prob(
    dyn::AffineAdditiveNoiseDynamics,
    state_abstraction::StateUniformGridSplit,
    input_abstraction::InputAbstraction,
    target_model::AbstractIMDPTarget
)
    # The first state is absorbing, representing transitioning to outside the partitioned.
    nregions = numregions(state_abstraction)
    ninputs = numinputs(input_abstraction)

    prob_lower, prob_upper = initprob(target_model, nregions, ninputs)

    # Since the partitioning is a uniform grid split, we can compute G for any region,
    # and Xᵢ - Aᵢ is equivalent to Aᵢ - Xᵢ. Furthermore, since we pick the center
    # as the representative point, it is equivalent to preserving the radius,
    # but setting the center to zero.
    first_region = first(regions(state_abstraction))
    G = Hyperrectangle(zero(center(first_region)), radius_hyperrectangle(first_region))
    V̂ = concretize(-dyn.A * G)

    # Print largest L-infinity norm of the disturbance set
    @info "Largest L-infinity norm of the disturbance set: $(maximum(radius_hyperrectangle(box_approximation(V̂))))"

    # Sink state is implicitly encoded

    # Transition probabilities
    prepare_nominal(dyn, input_abstraction)

    Threads.@threads for (i, source_region) in collect(enumerate(regions(state_abstraction)))
        for (j, input) in enumerate(inputs(input_abstraction))
            srcact_idx = (i - 1) * ninputs + j
            AXBU = dyn.A * center(source_region) + dyn.B * input
            Y = concretize(AXBU + V̂)

            theorem1_source_action_transition_prob(
                dyn,
                state_abstraction,
                target_model,
                Y,
                prob_lower,
                prob_upper,
                srcact_idx
            )
        end
    end

    prob_lower, prob_upper = postprocessprob(target_model, prob_lower, prob_upper)

    prob = IntervalProbabilities(; lower=prob_lower, upper=prob_upper)

    return prob
end

function theorem1_source_action_transition_prob(
    dyn::AdditiveNoiseDynamics,
    state_abstraction::StateUniformGridSplit,
    target_model::AbstractIMDPTarget,
    Y::LazySet,
    prob_lower,
    prob_upper,
    srcact_idx
)
    X = statespace(state_abstraction)
    w = noise(dyn)

    # Transition to outside the partitioned region
    pl_outside, pu_outside = transition_prob_bounds(Y, X, w)
    pl_outside, pu_outside = 1.0 - pu_outside, 1.0 - pl_outside

    # Transition to other states
    for (tar_idx, target_region) in enumerate(regions(state_abstraction))
        pl, pu = transition_prob_bounds(Y, target_region, noise(dyn))

        if includetransition(target_model, pu)
            @inbounds prob_lower[tar_idx, srcact_idx] = pl
            @inbounds prob_upper[tar_idx, srcact_idx] = pu
        else  # Allow sparsifying via adding probability to the absorbing avoid state
            pl_outside = pl_outside + pl
            pu_outside = pu_outside + pu
        end
    end

    # Use clamp to ensure that the probabilities are within [0, 1] (due to floating point errors).
    @inbounds prob_lower[end, srcact_idx] = clamp(pl_outside, 0.0, 1.0)
    @inbounds prob_upper[end, srcact_idx] = clamp(pu_outside, 0.0, 1.0)
end

function theorem2_abstraction(
    prob::AbstractionProblem{S},
    state_abstraction::StateUniformGridSplit,
    input_abstraction::InputAbstraction,
    target_model::AbstractIMDPTarget,
    C,
    max_output_error::Float64
) where {S <: System{<:AffineAdditiveNoiseDynamics}}
    sys = system(prob)
    spec = specification(prob)

    if !islinear(dynamics(sys))
        throw(ArgumentError("The system dynamics must be linear for Theorem 1."))
    end

    # State pointer
    stateptr = Int32[[1]
                    (1:numregions(state_abstraction)) .* numinputs(input_abstraction) .+ 1]

    # Transition probabilities
    interval_prob = theorem2_transition_prob(
        dynamics(sys),
        state_abstraction,
        input_abstraction,
        target_model,
        C,
        max_output_error
    )

    # Initial states
    initial_states = Int32[]
    for (i, source_region) in enumerate(regions(state_abstraction))
        if !iszeromeasure(initial(sys), source_region)
            push!(initial_states, i)
        end
    end

    mdp = IntervalMarkovDecisionProcess(interval_prob, stateptr, initial_states)

    # Property
    # TODO: This theorem may require a different specification conversion
    spec = convert_specification(spec, state_abstraction, target_model)

    return mdp, spec
end

function theorem2_transition_prob(
    dyn::AffineAdditiveNoiseDynamics,
    state_abstraction::StateUniformGridSplit,
    input_abstraction::InputAbstraction,
    target_model::AbstractIMDPTarget,
    C,
    max_output_error::Float64
)
    # The first state is absorbing, representing transitioning to outside the partitioned.
    nregions = numregions(state_abstraction)
    ninputs = numinputs(input_abstraction)

    prob_lower, prob_upper = initprob(target_model, nregions, ninputs)

    V̂ = theorem2_disturbance_set(
        dyn,
        state_abstraction,
        C,
        max_output_error
    )
    
    # Print largest L-infinity norm of the disturbance set
    @info "Largest L-infinity norm of the disturbance set: $(maximum(radius_hyperrectangle(V̂)))"

    # Sink state is implicitly encoded

    # Transition probabilities
    prepare_nominal(dyn, input_abstraction)

    Threads.@threads for (i, source_region) in collect(enumerate(regions(state_abstraction)))
        for (j, input) in enumerate(inputs(input_abstraction))
            srcact_idx = (i - 1) * ninputs + j
            AXBU = dyn.A * center(source_region) + dyn.B * input
            Y = concretize(AXBU + V̂)

            theorem2_source_action_transition_prob(
                dyn,
                state_abstraction,
                target_model,
                Y,
                prob_lower,
                prob_upper,
                srcact_idx
            )
        end
    end

    prob_lower, prob_upper = postprocessprob(target_model, prob_lower, prob_upper)

    prob = IntervalProbabilities(; lower=prob_lower, upper=prob_upper)

    return prob
end

function theorem2_disturbance_set(
    dyn::AffineAdditiveNoiseDynamics,
    state_abstraction::StateUniformGridSplit,
    C,
    max_output_error::Float64
)
    # Since the partitioning is a uniform grid split, we can compute G for any region,
    # and Xᵢ - Aᵢ is equivalent to Aᵢ - Xᵢ. Furthermore, since we pick the center
    # as the representative point, it is equivalent to preserving the radius,
    # but setting the center to zero.
    first_region = first(regions(state_abstraction))
    G = Hyperrectangle(zero(center(first_region)), radius_hyperrectangle(first_region))

    # Next solve the linear program to find the disturbance set
    model = Model(HiGHS.Optimizer)

    # delR_radius
    @variable(model, delR_radius[i=1:dimstate(dyn)] >= 0.0)
    delR_center = zero(delR_radius)
    delR = Hyperrectangle(delR_center, delR_radius .+ delR_center; check_bounds=false)

    # G ⊆ delR_radius
    @constraint(model, delR_radius >= radius_hyperrectangle(G))

    # Max output error (L-infinity norm)
    for vertex in vertices_list(delR)
        @constraint(model, C * vertex .<= max_output_error)
    end

    # V_radius
    @variable(model, V_radius[i=1:dimstate(dyn)] >= 0.0)
    V_center = zero(V_radius)
    V = Hyperrectangle(V_center, V_radius .+ V_center; check_bounds=false)

    # Invariance x(t+1) = A * x(t) + v + β with x(t), x(t + 1) ∈ delR, β ∈ G, v ∈ VV
    for delR_vertex in vertices_list(delR)
        # There exists a v ∈ V such that A * delR_vertex + v ∈ delR ⊖ G
        v = @variable(model, [1:dimstate(dyn)])
        @constraint(model, v .<= V_radius)
        @constraint(model, v .>= -V_radius)

        # A * x(t) + v + β ∈ delR
        @constraint(model, dyn.A * delR_vertex + v .<= delR_radius .- radius_hyperrectangle(G))
    end

    # Objective
    @variable(model, t ≥ 0.0)
    @constraint(model, V_radius .<= t)
    @objective(model, Min, t)

    optimize!(model)

    if termination_status(model) != MOI.OPTIMAL
        throw("The linear program did not find a solution.")
    end

    # Extract the disturbance set
    V_radius = value.(V_radius)
    V = Hyperrectangle(zero(V_radius), V_radius)
    
    return V
end

function theorem2_source_action_transition_prob(
    dyn::AdditiveNoiseDynamics,
    state_abstraction::StateUniformGridSplit,
    target_model::AbstractIMDPTarget,
    Y::LazySet,
    prob_lower,
    prob_upper,
    srcact_idx
)
    X = statespace(state_abstraction)
    w = noise(dyn)

    # Transition to outside the partitioned region
    pl_outside, pu_outside = transition_prob_bounds(Y, X, w)
    pl_outside, pu_outside = 1.0 - pu_outside, 1.0 - pl_outside

    # Transition to other states
    for (tar_idx, target_region) in enumerate(regions(state_abstraction))
        pl, pu = transition_prob_bounds(Y, target_region, noise(dyn))

        if includetransition(target_model, pu)
            @inbounds prob_lower[tar_idx, srcact_idx] = pl
            @inbounds prob_upper[tar_idx, srcact_idx] = pu
        else  # Allow sparsifying via adding probability to the absorbing avoid state
            pl_outside = pl_outside + pl
            pu_outside = pu_outside + pu
        end
    end

    # Use clamp to ensure that the probabilities are within [0, 1] (due to floating point errors).
    @inbounds prob_lower[end, srcact_idx] = clamp(pl_outside, 0.0, 1.0)
    @inbounds prob_upper[end, srcact_idx] = clamp(pu_outside, 0.0, 1.0)
end