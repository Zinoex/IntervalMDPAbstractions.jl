using SparseArrays

export abstraction

"""
    abstraction(sys::System{<:AdditiveNoiseDynamics}, state_abstraction::StateUniformGridSplit, input_abstraction::InputAbstraction, target_model::AbstractIMDPTarget)

Abstract function for creating an abstraction of a system with additive noise with an IMDP as the target model.
"""
function abstraction(sys::System{<:AdditiveNoiseDynamics}, state_abstraction::StateUniformGridSplit, input_abstraction::InputAbstraction, target_model::AbstractIMDPTarget)
    # The first state is absorbing, representing transitioning to outside the partitioned.
    nregions = numregions(state_abstraction) + 1
    ninputs = numinputs(input_abstraction)

    prob_lower, prob_upper = initprob(target_model, nregions, ninputs)

    # Absorbing
    prob_lower[1, 1] = 1.0
    prob_upper[1, 1] = 1.0

    # Transition probabilities
    dyn = dynamics(sys)

    for (i, source_region) in enumerate(regions(state_abstraction))
        for (j, input) in enumerate(inputs(input_abstraction))
            srcact_idx = (i - 1) * ninputs + j + 1
            Y = nominal(dyn, source_region, input)

            # Transition to outside the partitioned region
            pl, pu = transition_prob_bounds(Y, statespace(state_abstraction), noise(dyn))
            prob_lower[1, srcact_idx] = 1.0 - pu
            prob_upper[1, srcact_idx] = 1.0 - pl

            # Transition to other states
            for (tar_idx, target_region) in enumerate(regions(state_abstraction))
                pl, pu = transition_prob_bounds(Y, target_region, noise(dyn))

                if includetransition(target_model, pu)
                    prob_lower[tar_idx + 1, srcact_idx] = pl
                    prob_upper[tar_idx + 1, srcact_idx] = pu
                else  # Allow sparsifying via adding probability to the absorbing avoid state

                    # Use clamp to ensure that the probabilities are within [0, 1] (due to floating point errors).
                    prob_lower[1, srcact_idx] = clamp(prob_lower[1, srcact_idx] + pl, 0.0, 1.0)
                    prob_upper[1, srcact_idx] = clamp(prob_upper[1, srcact_idx] + pu, 0.0, 1.0)
                end
            end
        end
    end

    prob = IntervalProbabilities(;lower=prob_lower, upper=prob_upper)

    # State pointer
    stateptr = Int32[[1, 2]; (1:nregions-1) .* ninputs .+ 2]

    # Properties
    initial_states = Int32[]
    reach_states = Int32[]
    avoid_states = Int32[1] # Absorbing state

    for (i, source_region) in enumerate(regions(state_abstraction))
        if !isdisjoint(initial(sys), source_region)
            push!(initial_states, i + 1)
        end

        if !isdisjoint(avoid(sys), source_region)
            push!(avoid_states, i + 1)
        elseif source_region ⊆ reach(sys)
            push!(reach_states, i + 1)
        end
    end

    # Final construction
    return IntervalMarkovDecisionProcess(prob, stateptr, initial_states), reach_states, avoid_states
end

function initprob(::IMDPTarget, nregions, ninputs) 
    prob_lower = zeros(Float64, nregions, (nregions - 1) * ninputs + 1)
    prob_upper = copy(prob_lower)

    return prob_lower, prob_upper
end

function initprob(::SparseIMDPTarget, nregions, ninputs) 
    prob_lower = spzeros(Float64, Int32, nregions, (nregions - 1) * ninputs + 1)
    prob_upper = copy(prob_lower)

    return prob_lower, prob_upper
end

"""
    abstraction(sys::System{<:AdditiveNoiseDynamics}, state_abstraction::StateUniformGridSplit, input_abstraction::InputAbstraction, target_model::AbstractOrthogonalIMDPTarget)

Abstract function for creating an abstraction of a system with additive noise with a decoupled IMDP as the target model.
"""
function abstraction(sys::System{<:AdditiveNoiseDynamics}, state_abstraction::StateUniformGridSplit, input_abstraction::InputAbstraction, target_model::AbstractOrthogonalIMDPTarget)
    dyn = dynamics(sys)
    if !candecouple(noise(dyn))
        throw(ArgumentError("Cannot decouple system with non-diagonal noise covariance matrix"))
    end
    
    # The first state along each axis is absorbing, representing transitioning to outside the partitioned along that axis.
    ninputs = numinputs(input_abstraction)

    prob_lower, prob_upper = initprob(target_model, state_abstraction, ninputs)

    # Transition probabilities
    region_indices = LinearIndices(splits(state_abstraction))

    srcact_idx = 1
    for Icart in CartesianIndices(splits(state_abstraction) .+ 1)
        # Absorbing
        if any(Tuple(Icart) .== 1)
            for axis in eachindex(splits(state_abstraction))
                prob_lower[axis][1, srcact_idx] = 1.0
                prob_upper[axis][1, srcact_idx] = 1.0
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
                prob_lower[axis][1, srcact_idx] = 1.0 - pu
                prob_upper[axis][1, srcact_idx] = 1.0 - pl

                # Transition to other states
                axis_statespace = Interval(extrema(statespace(state_abstraction), axis)...)
                split_axis = LazySets.split(axis_statespace, [axisregions])
                
                for (tar_idx, target_region) in enumerate(split_axis)
                    pl, pu = axis_transition_prob_bounds(Y, target_region, noise(dyn), axis)

                    if includetransition(target_model, pu)
                        prob_lower[axis][tar_idx + 1, srcact_idx] = pl
                        prob_upper[axis][tar_idx + 1, srcact_idx] = pu
                    else  # Allow sparsifying via adding probability to the absorbing avoid state

                        # Use clamp to ensure that the probabilities are within [0, 1] (due to floating point errors).
                        prob_lower[axis][1, srcact_idx] = clamp(prob_lower[axis][1, srcact_idx] + pl, 0.0, 1.0)
                        prob_upper[axis][1, srcact_idx] = clamp(prob_upper[axis][1, srcact_idx] + pu, 0.0, 1.0)
                    end
                end
            end

            srcact_idx += 1
        end
    end

    prob = OrthogonalIntervalProbabilities(
        Tuple(IntervalProbabilities(;lower=pl, upper=pu) for (pl, pu) in zip(prob_lower, prob_upper)),
        Int32.(Tuple(splits(state_abstraction) .+ 1))
    )

    # State pointer
    stateptr = Int32[1]

    for I in CartesianIndices(splits(state_abstraction) .+ 1)
        if any(Tuple(I) .== 1)
            push!(stateptr, stateptr[end] + 1)
        else
            push!(stateptr, stateptr[end] + ninputs)
        end
    end

    # Properties
    initial_states = NTuple{dimstate(dyn), Int32}[]
    reach_states = NTuple{dimstate(dyn), Int32}[]
    avoid_states = NTuple{dimstate(dyn), Int32}[] # Absorbing state

    for I in CartesianIndices(splits(state_abstraction) .+ 1)
        if any(Tuple(I) .== 1)
            push!(avoid_states, Tuple(I))
        end
    end

    for (I, source_region) in zip(CartesianIndices(splits(state_abstraction)), regions(state_abstraction))
        if !isdisjoint(initial(sys), source_region)
            push!(initial_states, Tuple(I) .+ 1)
        end

        if !isdisjoint(avoid(sys), source_region)
            push!(avoid_states, Tuple(I) .+ 1)
        elseif source_region ⊆ reach(sys)
            push!(reach_states, Tuple(I) .+ 1)
        end
    end

    # Final construction
    return OrthogonalIntervalMarkovDecisionProcess(prob, stateptr, initial_states), reach_states, avoid_states
end

function initprob(::OrthogonalIMDPTarget, state_abstraction::StateUniformGridSplit, ninputs) 
    prob_lower = Matrix{Float64}[]
    prob_upper = Matrix{Float64}[]

    # One action for non-absorbing states is already included in the first term.
    nchoices = prod(splits(state_abstraction) .+ 1) + numregions(state_abstraction) * (ninputs - 1)

    for axisregions in splits(state_abstraction)
        local_prob_lower = zeros(Float64, axisregions + 1, nchoices)
        local_prob_upper = copy(local_prob_lower)

        push!(prob_lower, local_prob_lower)
        push!(prob_upper, local_prob_upper)
    end

    return prob_lower, prob_upper
end

function initprob(::SparseOrthogonalIMDPTarget, state_abstraction::StateUniformGridSplit, ninputs) 
    prob_lower = SparseMatrixCSC{Float64, Int32}[]
    prob_upper = SparseMatrixCSC{Float64, Int32}[]

    # One action for non-absorbing states is already included in the first term.
    nchoices = prod(splits(state_abstraction) .+ 1) + numregions(state_abstraction) * (ninputs - 1)

    for axisregions in splits(state_abstraction)
        local_prob_lower = spzeros(Float64, Int32, axisregions + 1, nchoices)
        local_prob_upper = copy(local_prob_lower)

        push!(prob_lower, local_prob_lower)
        push!(prob_upper, local_prob_upper)
    end

    return prob_lower, prob_upper
end
