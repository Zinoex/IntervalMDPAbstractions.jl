using SparseArrays

function transition_prob(dyn::AbstractedGaussianProcess, state_abstraction::StateUniformGridSplit, input_abstraction::InputAbstraction, stateptr, target_model::AbstractIMDPTarget)
    # The first state is absorbing, representing transitioning to outside the partitioned.
    nregions = numregions(state_abstraction) + 1
    ninputs = numinputs(input_abstraction)

    prob_lower, prob_upper = initprob(target_model, nregions, ninputs)

    # Absorbing
    prob_lower[1][1] = 1.0
    prob_upper[1][1] = 1.0

    Threads.@threads for (i, source_region) in collect(enumerate(regions(state_abstraction)))
        for (j, input) in enumerate(inputs(input_abstraction))
            srcact_idx = (i - 1) * ninputs + j + 1
            gp_bounds = bounds(dyn, source_region, input)

            source_action_transition_prob(dyn, state_abstraction, target_model, gp_bounds, prob_lower, prob_upper, srcact_idx)
        end
    end

    prob = IntervalProbabilities(;lower=efficient_hcat(prob_lower), upper=efficient_hcat(prob_upper))

    return prob
end

function source_action_transition_prob(dyn::AbstractedGaussianProcess, state_abstraction::StateUniformGridSplit, target_model::AbstractIMDPTarget, gp_bounds::AbstractedGaussianProcessRegion, prob_lower, prob_upper, srcact_idx)
    X = statespace(state_abstraction)
    
    # Transition to outside the partitioned region
    pl, pu = transition_prob_bounds(gp_bounds, X)
    prob_lower[srcact_idx][1] = 1.0 - pu
    prob_upper[srcact_idx][1] = 1.0 - pl

    # Transition to other states
    for (tar_idx, target_region) in enumerate(regions(state_abstraction))
        pl, pu = transition_prob_bounds(gp_bounds, target_region)

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

function transition_prob(dyn::AbstractedGaussianProcess, state_abstraction::StateUniformGridSplit, input_abstraction::InputAbstraction, stateptr, target_model::AbstractOrthogonalIMDPTarget)
    
    # The first state along each axis is absorbing, representing transitioning to outside the partitioned along that axis.
    ninputs = numinputs(input_abstraction)

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
            gp_bounds = bounds(dyn, source_region, input)

            source_action_transition_prob(dyn, state_abstraction, target_model, gp_bounds, prob_lower, prob_upper, srcact_idx)
            
            srcact_idx += 1
        end
    end

    prob = OrthogonalIntervalProbabilities(
        Tuple(IntervalProbabilities(;lower=efficient_hcat(pl), upper=efficient_hcat(pu)) for (pl, pu) in zip(prob_lower, prob_upper)),
        Int32.(Tuple(splits(state_abstraction) .+ 1))
    )

    return prob
end

function source_action_transition_prob(dyn::AbstractedGaussianProcess, state_abstraction::StateUniformGridSplit, target_model::AbstractOrthogonalIMDPTarget, gp_bounds::AbstractedGaussianProcessRegion, prob_lower, prob_upper, srcact_idx)
    X = statespace(state_abstraction)

    # For each axis... 
    for (axis, axisregions) in enumerate(splits(state_abstraction))
        # Transition to outside the partitioned region
        pl, pu = axis_transition_prob_bounds(gp_bounds, X, axis)
        prob_lower[axis][srcact_idx][1] = 1.0 - pu
        prob_upper[axis][srcact_idx][1] = 1.0 - pl

        # Transition to other states
        axis_statespace = Interval(low(X, axis), high(X, axis))
        split_axis = LazySets.split(axis_statespace, axisregions)
        
        for (tar_idx, target_region) in enumerate(split_axis)
            pl, pu = axis_transition_prob_bounds(gp_bounds, target_region, axis)

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
end