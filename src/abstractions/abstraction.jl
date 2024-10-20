
export abstraction


##################
# Model agnostic #
##################
function convert_specification(spec::Specification{<:FiniteTimeRegionReachability}, state_abstraction::StateUniformGridSplit, target_model)
    reach, avoid = convert_property(spec, state_abstraction, target_model)
    prop = FiniteTimeReachAvoid(reach, avoid, time_horizon(system_property(spec)))

    return Specification(prop, satisfaction_mode(spec), strategy_mode(spec))
end

function convert_specification(spec::Specification{<:InfiniteTimeRegionReachability}, state_abstraction::StateUniformGridSplit, target_model)
    reach, avoid = convert_property(spec, state_abstraction, target_model)
    prop = InfiniteTimeReachAvoid(reach, avoid, convergence_eps(system_property(spec)))

    return Specification(prop, satisfaction_mode(spec), strategy_mode(spec))
end

function convert_specification(spec::Specification{<:FiniteTimeRegionReachAvoid}, state_abstraction::StateUniformGridSplit, target_model)
    reach, avoid = convert_property(spec, state_abstraction, target_model)
    prop = FiniteTimeReachAvoid(reach, avoid, time_horizon(system_property(spec)))

    return Specification(prop, satisfaction_mode(spec), strategy_mode(spec))
end

function convert_specification(spec::Specification{<:InfiniteTimeRegionReachAvoid}, state_abstraction::StateUniformGridSplit, target_model)
    reach, avoid = convert_property(spec, state_abstraction, target_model)
    prop = InfiniteTimeReachAvoid(reach, avoid, convergence_eps(system_property(spec)))

    return Specification(prop, satisfaction_mode(spec), strategy_mode(spec))
end

function convert_specification(spec::Specification{<:FiniteTimeRegionSafety}, state_abstraction::StateUniformGridSplit, target_model)
    avoid = convert_property(spec, state_abstraction, target_model)
    prop = FiniteTimeSafety(avoid, time_horizon(system_property(spec)))

    return Specification(prop, satisfaction_mode(spec), strategy_mode(spec))
end

function convert_specification(spec::Specification{<:InfiniteTimeRegionSafety}, state_abstraction::StateUniformGridSplit, target_model)
    avoid = convert_property(spec, state_abstraction, target_model)
    prop = InfiniteTimeSafety(avoid, convergence_eps(system_property(spec)))

    return Specification(prop, satisfaction_mode(spec), strategy_mode(spec))
end


###############
# IMDP target #
###############
"""
    abstraction(prob::AbstractionProblem, state_abstraction::StateUniformGridSplit, input_abstraction::InputAbstraction, target_model::AbstractIMDPTarget)

Abstract function for creating an abstraction of a system with additive noise with an IMDP as the target model.
"""
function abstraction(prob::AbstractionProblem, state_abstraction::StateUniformGridSplit, input_abstraction::InputAbstraction, target_model::AbstractIMDPTarget)
    sys = system(prob)
    spec = specification(prob)

    # State pointer
    stateptr = Int32[[1, 2]; (1:numregions(state_abstraction)) .* numinputs(input_abstraction) .+ 2]

    # Transition probabilities
    interval_prob = transition_prob(dynamics(sys), state_abstraction, input_abstraction, stateptr, target_model)

    # Initial states
    initial_states = Int32[]
    for (i, source_region) in enumerate(regions(state_abstraction))
        if !isdisjoint(initial(sys), source_region)
            push!(initial_states, i + 1)
        end
    end
    
    mdp = IntervalMarkovDecisionProcess(interval_prob, stateptr, initial_states)

    # Property
    spec = convert_specification(spec, state_abstraction, target_model)

    return mdp, spec
end

function initprob(::IMDPTarget, nregions, ninputs) 
    prob_lower = [zeros(Float64, nregions) for _ in 1:((nregions - 1) * ninputs + 1)]
    prob_upper = [zeros(Float64, nregions) for _ in 1:((nregions - 1) * ninputs + 1)]

    return prob_lower, prob_upper
end

function initprob(::SparseIMDPTarget, nregions, ninputs) 
    prob_lower = [spzeros(Float64, nregions) for _ in 1:((nregions - 1) * ninputs + 1)]
    prob_upper = [spzeros(Float64, nregions) for _ in 1:((nregions - 1) * ninputs + 1)]

    return prob_lower, prob_upper
end

function convert_property(spec::Specification{<:AbstractRegionReachability}, state_abstraction::StateUniformGridSplit, ::AbstractIMDPTarget)
    prop = system_property(spec)
    
    reach_states = Int32[]
    avoid_states = Int32[1]  # Absorbing state

    for (i, source_region) in enumerate(regions(state_abstraction))
        if ispessimistic(spec) && source_region ⊆ reach(prop)
            push!(reach_states, i + 1)
        elseif isoptimistic(spec) && !iszeromeasure(reach(prop), source_region)
            push!(reach_states, i + 1)
        end
    end

    return reach_states, avoid_states
end

function convert_property(spec::Specification{<:AbstractRegionReachAvoid}, state_abstraction::StateUniformGridSplit, ::AbstractIMDPTarget)
    prop = system_property(spec)
    
    reach_states = Int32[]
    avoid_states = Int32[1]  # Absorbing state

    for (i, source_region) in enumerate(regions(state_abstraction))
        if ispessimistic(spec) && !iszeromeasure(avoid(prop), source_region)
            push!(avoid_states, i + 1)
        elseif isoptimistic(spec) && source_region ⊆ avoid(prop)
            push!(avoid_states, i + 1)
        elseif ispessimistic(spec) && source_region ⊆ reach(prop)
            push!(reach_states, i + 1)
        elseif isoptimistic(spec) && !iszeromeasure(reach(prop), source_region)
            push!(reach_states, i + 1)
        end
    end

    return reach_states, avoid_states
end

function convert_property(spec::Specification{<:AbstractRegionSafety}, state_abstraction::StateUniformGridSplit, ::AbstractIMDPTarget)
    prop = system_property(spec)
    
    avoid_states = Int32[1]  # Absorbing state

    for (i, source_region) in enumerate(regions(state_abstraction))
        if ispessimistic(spec) && !iszeromeasure(avoid(prop), source_region)
            push!(avoid_states, i + 1)
        elseif isoptimistic(spec) && source_region ⊆ avoid(prop)
            push!(avoid_states, i + 1)
        end
    end

    return avoid_states
end


##########################
# Orthogonal IMDP target #
##########################
"""
    abstraction(prob::AbstractionProblem, state_abstraction::StateUniformGridSplit, input_abstraction::InputAbstraction, target_model::AbstractOrthogonalIMDPTarget)

Abstract function for creating an abstraction of a system with additive noise with a decoupled IMDP as the target model.
"""
function abstraction(prob::AbstractionProblem, state_abstraction::StateUniformGridSplit, input_abstraction::InputAbstraction, target_model::AbstractOrthogonalIMDPTarget)
    sys = system(prob)
    spec = specification(prob)

    # State pointer
    stateptr = Int32[1]
    sizehint!(stateptr, prod(splits(state_abstraction) .+ 1))

    for I in CartesianIndices(splits(state_abstraction) .+ 1)
        if any(Tuple(I) .== 1)
            push!(stateptr, stateptr[end] + 1)
        else
            push!(stateptr, stateptr[end] + numinputs(input_abstraction))
        end
    end
        
    interval_prob = transition_prob(dynamics(sys), state_abstraction, input_abstraction, stateptr, target_model)

    # Initial states
    initial_states = NTuple{dimstate(sys), Int32}[]
    for (I, source_region) in zip(CartesianIndices(splits(state_abstraction)), regions(state_abstraction))
        if !isdisjoint(initial(sys), source_region)
            push!(initial_states, Tuple(I) .+ 1)
        end
    end

    mdp = OrthogonalIntervalMarkovDecisionProcess(interval_prob, stateptr, initial_states)

    # Property
    spec = convert_specification(spec, state_abstraction, target_model)

    return mdp, spec
end

function initprob(::OrthogonalIMDPTarget, state_abstraction::StateUniformGridSplit, ninputs) 
    prob_lower = Vector{Vector{Float64}}[]
    prob_upper = Vector{Vector{Float64}}[]

    # One action for non-absorbing states is already included in the first term.
    nchoices = prod(splits(state_abstraction) .+ 1) + numregions(state_abstraction) * (ninputs - 1)

    for axisregions in splits(state_abstraction)
        local_prob_lower = [zeros(Float64, axisregions + 1) for _ in 1:nchoices]
        local_prob_upper = [zeros(Float64, axisregions + 1) for _ in 1:nchoices]

        push!(prob_lower, local_prob_lower)
        push!(prob_upper, local_prob_upper)
    end

    return prob_lower, prob_upper
end

function initprob(::SparseOrthogonalIMDPTarget, state_abstraction::StateUniformGridSplit, ninputs) 
    prob_lower = Vector{SparseVector{Float64, Int32}}[]
    prob_upper = Vector{SparseVector{Float64, Int32}}[]

    # One action for non-absorbing states is already included in the first term.
    nchoices = prod(splits(state_abstraction) .+ 1) + numregions(state_abstraction) * (ninputs - 1)

    for axisregions in splits(state_abstraction)
        local_prob_lower = [spzeros(Float64, Int32, axisregions + 1) for _ in 1:nchoices]
        local_prob_upper = [spzeros(Float64, Int32, axisregions + 1) for _ in 1:nchoices]

        push!(prob_lower, local_prob_lower)
        push!(prob_upper, local_prob_upper)
    end

    return prob_lower, prob_upper
end

function convert_property(spec::Specification{<:AbstractRegionReachability}, state_abstraction::StateUniformGridSplit, ::AbstractOrthogonalIMDPTarget)
    prop = system_property(spec)
    
    reach_states = NTuple{dim(prop), Int32}[]
    avoid_states = NTuple{dim(prop), Int32}[]

    # Absorbing states
    for I in CartesianIndices(splits(state_abstraction) .+ 1)
        if any(Tuple(I) .== 1)
            push!(avoid_states, Tuple(I))
        end
    end    

    for (I, source_region) in zip(CartesianIndices(splits(state_abstraction)), regions(state_abstraction))
        if ispessimistic(spec) && source_region ⊆ reach(prop)
            push!(reach_states, Tuple(I) .+ 1)
        elseif isoptimistic(spec) && !iszeromeasure(reach(prop), source_region)
            push!(reach_states, Tuple(I) .+ 1)
        end
    end

    return reach_states, avoid_states
end

function convert_property(spec::Specification{<:AbstractRegionReachAvoid}, state_abstraction::StateUniformGridSplit, ::AbstractOrthogonalIMDPTarget)
    prop = system_property(spec)
    
    reach_states = NTuple{dim(prop), Int32}[]
    avoid_states = NTuple{dim(prop), Int32}[]

    # Absorbing states
    for I in CartesianIndices(splits(state_abstraction) .+ 1)
        if any(Tuple(I) .== 1)
            push!(avoid_states, Tuple(I))
        end
    end

    for (I, source_region) in zip(CartesianIndices(splits(state_abstraction)), regions(state_abstraction))
        if ispessimistic(spec) && !iszeromeasure(avoid(prop), source_region)
            push!(avoid_states, Tuple(I) .+ 1)
        elseif isoptimistic(spec) && source_region ⊆ avoid(prop)
            push!(avoid_states, Tuple(I) .+ 1)
        elseif ispessimistic(spec) && source_region ⊆ reach(prop)
            push!(reach_states, Tuple(I) .+ 1)
        elseif isoptimistic(spec) && !iszeromeasure(reach(prop), source_region)
            push!(reach_states, Tuple(I) .+ 1)
        end
    end

    return reach_states, avoid_states
end

function convert_property(spec::Specification{<:AbstractRegionSafety}, state_abstraction::StateUniformGridSplit, ::AbstractOrthogonalIMDPTarget)
    prop = system_property(spec)
    
    avoid_states = NTuple{dim(prop), Int32}[]

    # Absorbing states
    for I in CartesianIndices(splits(state_abstraction) .+ 1)
        if any(Tuple(I) .== 1)
            push!(avoid_states, Tuple(I))
        end
    end

    for (I, source_region) in zip(CartesianIndices(splits(state_abstraction)), regions(state_abstraction))
        if ispessimistic(spec) && !iszeromeasure(avoid(prop), source_region)
            push!(avoid_states, Tuple(I) .+ 1)
        elseif isoptimistic(spec) && source_region ⊆ avoid(prop)
            push!(avoid_states, Tuple(I) .+ 1)
        end
    end

    return avoid_states
end


#######################
# Mixture IMDP target #
#######################
"""
    abstraction(prob::AbstractionProblem, state_abstraction::StateUniformGridSplit, input_abstraction::InputAbstraction, target_model::AbstractMixtureIMDPTarget)

Abstract function for creating an abstraction of a system with a mixture of orthogonal IMDPs as the target model.
"""
function abstraction(prob::AbstractionProblem, state_abstraction::StateUniformGridSplit, input_abstraction::InputAbstraction, target_model::AbstractMixtureIMDPTarget)
    sys = system(prob)
    spec = specification(prob)

    # State pointer
    stateptr = Int32[1]
    sizehint!(stateptr, prod(splits(state_abstraction) .+ 1))

    for I in CartesianIndices(splits(state_abstraction) .+ 1)
        if any(Tuple(I) .== 1)
            push!(stateptr, stateptr[end] + 1)
        else
            push!(stateptr, stateptr[end] + numinputs(input_abstraction))
        end
    end
    
    # Transition probabilities
    interval_prob = transition_prob(dynamics(sys), state_abstraction, input_abstraction, stateptr, target_model)

    # Initial states
    initial_states = NTuple{dimstate(sys), Int32}[]
    for (I, source_region) in zip(CartesianIndices(splits(state_abstraction)), regions(state_abstraction))
        if !isdisjoint(initial(sys), source_region)
            push!(initial_states, Tuple(I) .+ 1)
        end
    end

    mdp = MixtureIntervalMarkovDecisionProcess(interval_prob, stateptr, initial_states)

    # Property
    spec = convert_specification(spec, state_abstraction, target_model)

    return mdp, spec
end