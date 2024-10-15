using SparseArrays
export abstraction

"""
    abstraction(prob::AbstractionProblem{<:System{<:AdditiveNoiseDynamics}}, state_abstraction::StateUniformGridSplit, input_abstraction::InputAbstraction, target_model::AbstractIMDPTarget)

Abstract function for creating an abstraction of a system with additive noise with an IMDP as the target model.
"""
function abstraction(prob::AbstractionProblem{<:System{<:AdditiveNoiseDynamics}}, state_abstraction::StateUniformGridSplit, input_abstraction::InputAbstraction, target_model::AbstractIMDPTarget)
    sys = system(prob)
    spec = specification(prob)
    
    # The first state is absorbing, representing transitioning to outside the partitioned.
    nregions = numregions(state_abstraction) + 1
    ninputs = numinputs(input_abstraction)

    prob_lower, prob_upper = initprob(target_model, nregions, ninputs)

    # Absorbing
    prob_lower[1][1] = 1.0
    prob_upper[1][1] = 1.0

    # Transition probabilities
    dyn = dynamics(sys)
    prepare_nominal(dyn, input_abstraction)

    Threads.@threads for (i, source_region) in collect(enumerate(regions(state_abstraction)))
        for (j, input) in enumerate(inputs(input_abstraction))
            srcact_idx = (i - 1) * ninputs + j + 1
            Y = nominal(dyn, source_region, input)

            # Transition to outside the partitioned region
            pl, pu = transition_prob_bounds(Y, statespace(state_abstraction), noise(dyn))
            prob_lower[srcact_idx][1] = 1.0 - pu
            prob_upper[srcact_idx][1] = 1.0 - pl

            # Transition to other states
            for (tar_idx, target_region) in enumerate(regions(state_abstraction))
                pl, pu = transition_prob_bounds(Y, target_region, noise(dyn))

                if includetransition(target_model, pu)
                    prob_lower[srcact_idx][tar_idx + 1] = pl
                    prob_upper[srcact_idx][tar_idx + 1] = pu
                else  # Allow sparsifying via adding probability to the absorbing avoid state

                    # Use clamp to ensure that the probabilities are within [0, 1] (due to floating point errors).
                    prob_lower[srcact_idx][1] = clamp(prob_lower[srcact_idx][1] + pl, 0.0, 1.0)
                    prob_upper[srcact_idx][1] = clamp(prob_upper[srcact_idx][1] + pu, 0.0, 1.0)
                end
            end
        end
    end

    prob = IntervalProbabilities(;lower=efficient_hcat(prob_lower), upper=efficient_hcat(prob_upper))

    # State pointer
    stateptr = Int32[[1, 2]; (1:nregions-1) .* ninputs .+ 2]

    # Initial states
    initial_states = Int32[]
    for (i, source_region) in enumerate(regions(state_abstraction))
        if !isdisjoint(initial(sys), source_region)
            push!(initial_states, i + 1)
        end
    end
    
    mdp = IntervalMarkovDecisionProcess(prob, stateptr, initial_states)

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

"""
    abstraction(prob::AbstractionProblem{<:System{<:AdditiveNoiseDynamics}}, state_abstraction::StateUniformGridSplit, input_abstraction::InputAbstraction, target_model::AbstractOrthogonalIMDPTarget)

Abstract function for creating an abstraction of a system with additive noise with a decoupled IMDP as the target model.
"""
function abstraction(prob::AbstractionProblem{<:System{<:AdditiveNoiseDynamics}}, state_abstraction::StateUniformGridSplit, input_abstraction::InputAbstraction, target_model::AbstractOrthogonalIMDPTarget)
    sys = system(prob)
    spec = specification(prob)
    
    dyn = dynamics(sys)
    if !candecouple(noise(dyn))
        throw(ArgumentError("Cannot decouple system with non-diagonal noise covariance matrix"))
    end
    prepare_nominal(dyn, input_abstraction)
    
    # The first state along each axis is absorbing, representing transitioning to outside the partitioned along that axis.
    ninputs = numinputs(input_abstraction)


    # State pointer
    stateptr = Int32[1]
    sizehint!(stateptr, prod(splits(state_abstraction) .+ 1))

    for I in CartesianIndices(splits(state_abstraction) .+ 1)
        if any(Tuple(I) .== 1)
            push!(stateptr, stateptr[end] + 1)
        else
            push!(stateptr, stateptr[end] + ninputs)
        end
    end

    # Transition probabilities
    prob_lower, prob_upper = initprob(target_model, state_abstraction, ninputs)
    region_indices = LinearIndices(splits(state_abstraction))

    linear_indices = LinearIndices(splits(state_abstraction) .+ 1)
    Threads.@threads for Icart in CartesianIndices(splits(state_abstraction) .+ 1)
        Ilinear = linear_indices[Icart]
        srcact_idx = stateptr[Ilinear]

        # Absorbing
        if any(Tuple(Icart) .== 1)
            for axis in eachindex(splits(state_abstraction))
                prob_lower[axis][srcact_idx][1] = 1.0
                prob_upper[axis][srcact_idx][1] = 1.0
            end
            srcact_idx += 1

            continue
        end

        # Other states
        Iregion = CartesianIndex(Tuple(Icart) .- 1)
        source_region = regions(state_abstraction)[region_indices[Iregion]]
        for input in inputs(input_abstraction)
            # To decouple, we need to construct a hyperrectangle around the nominal one-step reachable region
            Y = box_approximation(nominal(dyn, source_region, input))

            # For each axis... 
            for (axis, axisregions) in enumerate(splits(state_abstraction))
                # Transition to outside the partitioned region
                pl, pu = axis_transition_prob_bounds(Y, statespace(state_abstraction), noise(dyn), axis)
                prob_lower[axis][srcact_idx][1] = 1.0 - pu
                prob_upper[axis][srcact_idx][1] = 1.0 - pl

                # Transition to other states
                axis_statespace = Interval(extrema(statespace(state_abstraction), axis)...)
                split_axis = LazySets.split(axis_statespace, [axisregions])
                
                for (tar_idx, target_region) in enumerate(split_axis)
                    pl, pu = axis_transition_prob_bounds(Y, target_region, noise(dyn), axis)

                    if includetransition(target_model, pu)
                        prob_lower[axis][srcact_idx][tar_idx + 1] = pl
                        prob_upper[axis][srcact_idx][tar_idx + 1] = pu
                    else  # Allow sparsifying via adding probability to the absorbing avoid state

                        # Use clamp to ensure that the probabilities are within [0, 1] (due to floating point errors).
                        prob_lower[axis][srcact_idx][1] = clamp(prob_lower[axis][srcact_idx][1] + pl, 0.0, 1.0)
                        prob_upper[axis][srcact_idx][1] = clamp(prob_upper[axis][srcact_idx][1] + pu, 0.0, 1.0)
                    end
                end
            end
            
            srcact_idx += 1
        end
    end

    prob = OrthogonalIntervalProbabilities(
        Tuple(IntervalProbabilities(;lower=efficient_hcat(pl), upper=efficient_hcat(pu)) for (pl, pu) in zip(prob_lower, prob_upper)),
        Int32.(Tuple(splits(state_abstraction) .+ 1))
    )

    # Initial states
    initial_states = NTuple{dimstate(dyn), Int32}[]
    for (I, source_region) in zip(CartesianIndices(splits(state_abstraction)), regions(state_abstraction))
        if !isdisjoint(initial(sys), source_region)
            push!(initial_states, Tuple(I) .+ 1)
        end
    end

    mdp = OrthogonalIntervalMarkovDecisionProcess(prob, stateptr, initial_states)

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