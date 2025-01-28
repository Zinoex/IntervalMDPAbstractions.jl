
export abstraction


##################
# Model agnostic #
##################
function convert_specification(
    spec::Specification{<:FiniteTimeRegionReachability},
    state_abstraction::StateUniformGridSplit,
    target_model,
)
    reach, avoid = convert_property(spec, state_abstraction, target_model)
    prop = FiniteTimeReachAvoid(reach, avoid, time_horizon(system_property(spec)))

    return Specification(prop, satisfaction_mode(spec), strategy_mode(spec))
end

function convert_specification(
    spec::Specification{<:InfiniteTimeRegionReachability},
    state_abstraction::StateUniformGridSplit,
    target_model,
)
    reach, avoid = convert_property(spec, state_abstraction, target_model)
    prop = InfiniteTimeReachAvoid(reach, avoid, convergence_eps(system_property(spec)))

    return Specification(prop, satisfaction_mode(spec), strategy_mode(spec))
end

function convert_specification(
    spec::Specification{<:FiniteTimeRegionReachAvoid},
    state_abstraction::StateUniformGridSplit,
    target_model,
)
    reach, avoid = convert_property(spec, state_abstraction, target_model)
    prop = FiniteTimeReachAvoid(reach, avoid, time_horizon(system_property(spec)))

    return Specification(prop, satisfaction_mode(spec), strategy_mode(spec))
end

function convert_specification(
    spec::Specification{<:InfiniteTimeRegionReachAvoid},
    state_abstraction::StateUniformGridSplit,
    target_model,
)
    reach, avoid = convert_property(spec, state_abstraction, target_model)
    prop = InfiniteTimeReachAvoid(reach, avoid, convergence_eps(system_property(spec)))

    return Specification(prop, satisfaction_mode(spec), strategy_mode(spec))
end

function convert_specification(
    spec::Specification{<:FiniteTimeRegionSafety},
    state_abstraction::StateUniformGridSplit,
    target_model,
)
    avoid = convert_property(spec, state_abstraction, target_model)
    prop = FiniteTimeSafety(avoid, time_horizon(system_property(spec)))

    return Specification(prop, satisfaction_mode(spec), strategy_mode(spec))
end

function convert_specification(
    spec::Specification{<:InfiniteTimeRegionSafety},
    state_abstraction::StateUniformGridSplit,
    target_model,
)
    avoid = convert_property(spec, state_abstraction, target_model)
    prop = InfiniteTimeSafety(avoid, convergence_eps(system_property(spec)))

    return Specification(prop, satisfaction_mode(spec), strategy_mode(spec))
end


###############
# IMDP target #
###############
"""
    abstraction(prob, state_abstraction::StateUniformGridSplit, input_abstraction, target_model::AbstractIMDPTarget)

Construct an abstraction of a system and a specification under a uniform grid partitioning of the state space with an arbitrary input abstraction
and an IMDP as the target model.

The argument `prob` contains both the system and the specification. The type of the system determines how the transition probability bounds are computed.
The resulting IMDP has `numregions(state_abstraction) + 1` states, where the last state is an absorbing state, representing transitioning to outside
the partitioned region. This absorbing state is implicitly encoded in the ambiguity sets, i.e. not stored and automatically handled by `IntervalMDP.jl`.

The specification is converted based on the `state_abstraction` and `target_model` arguments in addition to whether the specification is pessimistic or optimistic.
To encode the specification, at least one avoid state is required, i.e. the last, absorbing state. As a consequence, (concrete) reachability specifications are
converted to (abstract) reach-avoid specifications with the last state as the avoid state.

Returns `mdp` and `spec` as the abstracted IMDP and the converted specification, respectively.
"""
function abstraction(
    prob::AbstractionProblem,
    state_abstraction::StateUniformGridSplit,
    input_abstraction::InputAbstraction,
    target_model::AbstractIMDPTarget,
)
    sys = system(prob)
    spec = specification(prob)

    # State pointer
    stateptr = Int32[
        [1]
        (1:numregions(state_abstraction)) .* numinputs(input_abstraction) .+ 1
    ]

    # Transition probabilities
    interval_prob = transition_prob(
        dynamics(sys),
        state_abstraction,
        input_abstraction,
        stateptr,
        target_model,
    )

    # Initial states
    initial_states = Int32[]
    for (i, source_region) in enumerate(regions(state_abstraction))
        if !isdisjoint(initial(sys), source_region)
            push!(initial_states, i)
        end
    end

    mdp = IntervalMarkovDecisionProcess(interval_prob, stateptr, initial_states)

    # Property
    spec = convert_specification(spec, state_abstraction, target_model)

    return mdp, spec
end

function initprob(::IMDPTarget, nregions, ninputs)
    nchoices = nregions * ninputs
    prob_lower = zeros(Float64, nregions + 1, nchoices)
    prob_upper = zeros(Float64, nregions + 1, nchoices)

    return prob_lower, prob_upper
end

function postprocessprob(::IMDPTarget, prob_lower, prob_upper)
    return prob_lower, prob_upper
end

function initprob(::SparseIMDPTarget, nregions, ninputs)
    nchoices = nregions * ninputs
    prob_lower = AtomicSparseMatrixCOO{Float64,Int32}(undef, nregions + 1, nchoices)
    prob_upper = AtomicSparseMatrixCOO{Float64,Int32}(undef, nregions + 1, nchoices)

    return prob_lower, prob_upper
end

function postprocessprob(::SparseIMDPTarget, prob_lower, prob_upper)
    prob_lower = sparse(
        prob_lower.rows,
        prob_lower.cols,
        prob_lower.values,
        prob_lower.m,
        prob_lower.n,
    )
    prob_upper = sparse(
        prob_upper.rows,
        prob_upper.cols,
        prob_upper.values,
        prob_upper.m,
        prob_upper.n,
    )

    return prob_lower, prob_upper
end

function convert_property(
    spec::Specification{<:AbstractRegionReachability},
    state_abstraction::StateUniformGridSplit,
    ::AbstractIMDPTarget,
)
    prop = system_property(spec)

    reach_states = Int32[]
    avoid_states = Int32[numregions(state_abstraction)+1]  # Absorbing state

    for (i, source_region) in enumerate(regions(state_abstraction))
        if ispessimistic(spec) && source_region ⊆ reach(prop)
            push!(reach_states, i)
        elseif isoptimistic(spec) && !iszeromeasure(reach(prop), source_region)
            push!(reach_states, i)
        end
    end

    return reach_states, avoid_states
end

function convert_property(
    spec::Specification{<:AbstractRegionReachAvoid},
    state_abstraction::StateUniformGridSplit,
    ::AbstractIMDPTarget,
)
    prop = system_property(spec)

    reach_states = Int32[]
    avoid_states = Int32[numregions(state_abstraction)+1]  # Absorbing state

    for (i, source_region) in enumerate(regions(state_abstraction))
        if ispessimistic(spec) && !iszeromeasure(avoid(prop), source_region)
            push!(avoid_states, i)
        elseif isoptimistic(spec) && source_region ⊆ avoid(prop)
            push!(avoid_states, i)
        elseif ispessimistic(spec) && source_region ⊆ reach(prop)
            push!(reach_states, i)
        elseif isoptimistic(spec) && !iszeromeasure(reach(prop), source_region)
            push!(reach_states, i)
        end
    end

    return reach_states, avoid_states
end

function convert_property(
    spec::Specification{<:AbstractRegionSafety},
    state_abstraction::StateUniformGridSplit,
    ::AbstractIMDPTarget,
)
    prop = system_property(spec)

    avoid_states = Int32[numregions(state_abstraction)+1]  # Absorbing state

    for (i, source_region) in enumerate(regions(state_abstraction))
        if ispessimistic(spec) && !iszeromeasure(avoid(prop), source_region)
            push!(avoid_states, i)
        elseif isoptimistic(spec) && source_region ⊆ avoid(prop)
            push!(avoid_states, i)
        end
    end

    return avoid_states
end


##########################
# Orthogonal IMDP target #
##########################
"""
    abstraction(prob, state_abstraction::StateUniformGridSplit, input_abstraction, target_model::AbstractOrthogonalIMDPTarget)

Construct an abstraction of a system and a specification under a uniform grid partitioning of the state space with an arbitrary input abstraction
and an orthogonal IMDP as the target model.

The argument `prob` contains both the system and the specification. The type of the system determines how the *marginal* transition probability bounds are computed.
The resulting *orthogonal* IMDP has `IntervalMDPAbstractions.splits(state_abstraction) .+ 1` states along each axis, 
where the last state along each axis is an absorbing state, representing transitioning to outside the partitioned region.
This absorbing state for each axis is implicitly encoded in the ambiguity sets, i.e. not stored and automatically handled by `IntervalMDP.jl`.

The specification is converted based on the `state_abstraction` and `target_model` arguments in addition to whether the specification is pessimistic or optimistic.
To encode the specification, at least one avoid state is required, i.e. the last, absorbing state. As a consequence, (concrete) reachability specifications are
converted to (abstract) reach-avoid specifications with the last state as the avoid state.

Returns `mdp` and `spec` as the abstracted IMDP and the converted specification, respectively.
"""
function abstraction(
    prob::AbstractionProblem,
    state_abstraction::StateUniformGridSplit,
    input_abstraction::InputAbstraction,
    target_model::AbstractOrthogonalIMDPTarget,
)
    sys = system(prob)
    spec = specification(prob)

    # State pointer
    stateptr = Int32[1]
    sizehint!(stateptr, prod(splits(state_abstraction)) + 1)

    for I in CartesianIndices(splits(state_abstraction))
        push!(stateptr, stateptr[end] + numinputs(input_abstraction))
    end

    interval_prob = transition_prob(
        dynamics(sys),
        state_abstraction,
        input_abstraction,
        stateptr,
        target_model,
    )

    # Initial states
    initial_states = CartesianIndex{dimstate(sys)}[]
    for (I, source_region) in
        zip(CartesianIndices(splits(state_abstraction)), regions(state_abstraction))
        if !isdisjoint(initial(sys), source_region)
            push!(initial_states, I)
        end
    end

    mdp = OrthogonalIntervalMarkovDecisionProcess(interval_prob, stateptr, initial_states)

    # Property
    spec = convert_specification(spec, state_abstraction, target_model)

    return mdp, spec
end

function initprob(::OrthogonalIMDPTarget, state_abstraction::StateUniformGridSplit, ninputs)
    prob_lower = Matrix{Float64}[]
    prob_upper = Matrix{Float64}[]

    nchoices = numregions(state_abstraction) * ninputs

    for axisregions in splits(state_abstraction)
        local_prob_lower = zeros(Float64, axisregions + 1, nchoices)
        local_prob_upper = zeros(Float64, axisregions + 1, nchoices)

        push!(prob_lower, local_prob_lower)
        push!(prob_upper, local_prob_upper)
    end

    return prob_lower, prob_upper
end

function postprocessprob(::OrthogonalIMDPTarget, prob_lower, prob_upper)
    return prob_lower, prob_upper
end

function initprob(
    ::SparseOrthogonalIMDPTarget,
    state_abstraction::StateUniformGridSplit,
    ninputs,
)
    prob_lower = AtomicSparseMatrixCOO{Float64,Int32}[]
    prob_upper = AtomicSparseMatrixCOO{Float64,Int32}[]

    nchoices = numregions(state_abstraction) * ninputs

    for axisregions in splits(state_abstraction)
        local_prob_lower =
            AtomicSparseMatrixCOO{Float64,Int32}(undef, axisregions + 1, nchoices)
        local_prob_upper =
            AtomicSparseMatrixCOO{Float64,Int32}(undef, axisregions + 1, nchoices)

        push!(prob_lower, local_prob_lower)
        push!(prob_upper, local_prob_upper)
    end

    return prob_lower, prob_upper
end

function postprocessprob(::SparseOrthogonalIMDPTarget, prob_lower, prob_upper)
    prob_lower = [sparse(p.rows, p.cols, p.values, p.m, p.n) for p in prob_lower]
    prob_upper = [sparse(p.rows, p.cols, p.values, p.m, p.n) for p in prob_upper]

    return prob_lower, prob_upper
end

function convert_property(
    spec::Specification{<:AbstractRegionReachability},
    state_abstraction::StateUniformGridSplit,
    ::AbstractOrthogonalIMDPTarget,
)
    prop = system_property(spec)

    reach_states = CartesianIndex{dim(prop)}[]
    avoid_states = CartesianIndex{dim(prop)}[]

    # Absorbing states
    extended_states = splits(state_abstraction) .+ 1
    for I in CartesianIndices(extended_states)
        if any(Tuple(I) .== extended_states)
            push!(avoid_states, I)
        end
    end

    for (I, source_region) in
        zip(CartesianIndices(splits(state_abstraction)), regions(state_abstraction))
        if ispessimistic(spec) && source_region ⊆ reach(prop)
            push!(reach_states, I)
        elseif isoptimistic(spec) && !iszeromeasure(reach(prop), source_region)
            push!(reach_states, I)
        end
    end

    return reach_states, avoid_states
end

function convert_property(
    spec::Specification{<:AbstractRegionReachAvoid},
    state_abstraction::StateUniformGridSplit,
    ::AbstractOrthogonalIMDPTarget,
)
    prop = system_property(spec)

    reach_states = CartesianIndex{dim(prop)}[]
    avoid_states = CartesianIndex{dim(prop)}[]

    # Absorbing states
    extended_states = splits(state_abstraction) .+ 1
    for I in CartesianIndices(extended_states)
        if any(Tuple(I) .== extended_states)
            push!(avoid_states, I)
        end
    end

    for (I, source_region) in
        zip(CartesianIndices(splits(state_abstraction)), regions(state_abstraction))
        if ispessimistic(spec) && !iszeromeasure(avoid(prop), source_region)
            push!(avoid_states, I)
        elseif isoptimistic(spec) && source_region ⊆ avoid(prop)
            push!(avoid_states, I)
        elseif ispessimistic(spec) && source_region ⊆ reach(prop)
            push!(reach_states, I)
        elseif isoptimistic(spec) && !iszeromeasure(reach(prop), source_region)
            push!(reach_states, I)
        end
    end

    return reach_states, avoid_states
end

function convert_property(
    spec::Specification{<:AbstractRegionSafety},
    state_abstraction::StateUniformGridSplit,
    ::AbstractOrthogonalIMDPTarget,
)
    prop = system_property(spec)

    avoid_states = CartesianIndex{dim(prop)}[]

    # Absorbing states
    extended_states = splits(state_abstraction) .+ 1
    for I in CartesianIndices(extended_states)
        if any(Tuple(I) .== extended_states)
            push!(avoid_states, I)
        end
    end

    for (I, source_region) in
        zip(CartesianIndices(splits(state_abstraction)), regions(state_abstraction))
        if ispessimistic(spec) && !iszeromeasure(avoid(prop), source_region)
            push!(avoid_states, I)
        elseif isoptimistic(spec) && source_region ⊆ avoid(prop)
            push!(avoid_states, I)
        end
    end

    return avoid_states
end


#######################
# Mixture IMDP target #
#######################
"""
    abstraction(prob, state_abstraction::StateUniformGridSplit, input_abstraction, target_model::AbstractMixtureIMDPTarget)

Construct an abstraction of a system and a specification under a uniform grid partitioning of the state space with an arbitrary input abstraction
and a mixture of orthogonal IMDPs as the target model.

The argument `prob` contains both the system and the specification. The type of the system determines how the *marginal mixture* transition probability bounds are computed.
The resulting *mixture* IMDP has `IntervalMDPAbstractions.splits(state_abstraction) .+ 1` states along each axis, 
where the last state along each axis is an absorbing state, representing transitioning to outside the partitioned region.
This absorbing state for each axis is implicitly encoded in the ambiguity sets, i.e. not stored and automatically handled by `IntervalMDP.jl`.

The specification is converted based on the `state_abstraction` and `target_model` arguments in addition to whether the specification is pessimistic or optimistic.
To encode the specification, at least one avoid state is required, i.e. the last, absorbing state. As a consequence, (concrete) reachability specifications are
converted to (abstract) reach-avoid specifications with the last state as the avoid state.

Returns `mdp` and `spec` as the abstracted IMDP and the converted specification, respectively.
"""
function abstraction(
    prob::AbstractionProblem,
    state_abstraction::StateUniformGridSplit,
    input_abstraction::InputAbstraction,
    target_model::AbstractMixtureIMDPTarget,
)
    sys = system(prob)
    spec = specification(prob)

    # State pointer
    stateptr = Int32[1]
    sizehint!(stateptr, prod(splits(state_abstraction)) + 1)

    for I in CartesianIndices(splits(state_abstraction))
        push!(stateptr, stateptr[end] + numinputs(input_abstraction))
    end

    # Transition probabilities
    interval_prob = transition_prob(
        dynamics(sys),
        state_abstraction,
        input_abstraction,
        stateptr,
        target_model,
    )

    # Initial states
    initial_states = CartesianIndex{dimstate(sys)}[]
    for (I, source_region) in
        zip(CartesianIndices(splits(state_abstraction)), regions(state_abstraction))
        if !isdisjoint(initial(sys), source_region)
            push!(initial_states, I)
        end
    end

    mdp = MixtureIntervalMarkovDecisionProcess(interval_prob, stateptr, initial_states)

    # Property
    spec = convert_specification(spec, state_abstraction, target_model)

    return mdp, spec
end
