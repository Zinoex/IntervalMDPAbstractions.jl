using SparseArrays

export abstraction

"""
    abstraction(sys::System{<:AffineAdditiveNoiseDynamics}, state_abstraction::StateGridSplit, input_abstraction::InputAbstraction, target_model::Union{DirectIMDP, SparseDirectIMDP})

Abstract function for creating an abstraction of a system with additive noise with an IMDP as the target model.
"""
function abstraction(sys::System{<:AffineAdditiveNoiseDynamics}, state_abstraction::StateGridSplit, input_abstraction::InputAbstraction, target_model::Union{DirectIMDP, SparseDirectIMDP})
    # The first state is absorbing, representing transitioning to outside the partitioned.
    nregions = numregions(state_abstraction) + 1
    ninputs = numinputs(input_abstraction)

    prob_lower, prob_upper = direct_initprob(target_model, nregions, ninputs)

    # Absorbing
    prob_lower[1, 1] = 1.0
    prob_upper[1, 1] = 1.0

    # Transition probabilities
    dyn = dynamics(sys)

    for (i, source_region) in enumerate(regions(state_abstraction))
        for (j, input) in enumerate(inputs(input_abstraction))
            Y = nominal(dyn, source_region, input)
            srcact_idx = (i - 1) * ninputs + j + 1

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
                    prob_lower[1, srcact_idx] += pl
                    prob_upper[1, srcact_idx] += pu
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

        if reach(sys) ⊆ source_region
            push!(reach_states, i + 1)
        end

        if !isdisjoint(avoid(sys), source_region)
            push!(avoid_states, i + 1)
        end
    end

    # Final construction
    return IntervalMarkovDecisionProcess(prob, stateptr, initial_states), reach_states, avoid_states
end

function direct_initprob(target::DirectIMDP, nregions, ninputs) 
    prob_lower = zeros(Float64, nregions, (nregions - 1) * ninputs + 1)
    prob_upper = copy(prob_lower)

    return prob_lower, prob_upper
end

function direct_initprob(target::SparseDirectIMDP, nregions, ninputs) 
    prob_lower = spzeros(Float64, Int32, nregions, (nregions - 1) * ninputs + 1)
    prob_upper = copy(prob_lower)

    return prob_lower, prob_upper
end

"""
    abstraction(sys::System{<:AffineAdditiveNoiseDynamics}, state_abstraction::StateGridSplit, input_abstraction::InputAbstraction, target_model::DecoupledIMDP)

Abstract function for creating an abstraction of a system with additive noise with a decoupled IMDP as the target model.
"""
function abstraction(sys::System{<:AffineAdditiveNoiseDynamics}, state_abstraction::StateGridSplit, input_abstraction::InputAbstraction, target_model::DecoupledIMDP)
    # The first state is absorbing, representing transitioning to outside the partitioned.
    nregions = numregions(state_abstraction) + 1
    ninputs = numinputs(input_abstraction)

    prob_lower, prob_upper = direct_initprob(target_model, nregions, ninputs)

    # Absorbing
    prob_lower[1, 1] = 1.0
    prob_upper[1, 1] = 1.0

    # Transition probabilities
    dyn = dynamics(sys)

    # TODO: Ensure enumeration order is correct!! 
    for (i, source_region) in enumerate(regions(state_abstraction))
        for (j, input) in enumerate(inputs(input_abstraction))
            Y = nominal(dyn, source_region, input)
            srcact_idx = (i - 1) * ninputs + j + 1

            # For each axis... 
            
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
                    prob_lower[1, srcact_idx] += pl
                    prob_upper[1, srcact_idx] += pu
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

        if reach(sys) ⊆ source_region
            push!(reach_states, i + 1)
        end

        if !isdisjoint(avoid(sys), source_region)
            push!(avoid_states, i + 1)
        end
    end

    # Final construction
    return IntervalMarkovDecisionProcess(prob, stateptr, initial_states), reach_states, avoid_states
end