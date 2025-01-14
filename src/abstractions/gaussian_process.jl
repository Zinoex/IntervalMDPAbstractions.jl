using SparseArrays

function transition_prob(
    dyn::AbstractedGaussianProcess,
    state_abstraction::StateUniformGridSplit,
    input_abstraction::InputAbstraction,
    stateptr,
    target_model::AbstractIMDPTarget,
)
    # The first state is absorbing, representing transitioning to outside the partitioned.
    nregions = numregions(state_abstraction)
    ninputs = numinputs(input_abstraction)

    prob_lower, prob_upper = initprob(target_model, nregions, ninputs)

    # Sink state is implicitly endcoded

    Threads.@threads for (i, source_region) in
                         collect(enumerate(regions(state_abstraction)))
        for (j, input) in enumerate(inputs(input_abstraction))
            srcact_idx = (i - 1) * ninputs + j
            gp_bounds = bounds(dyn, source_region, input)

            source_action_transition_prob(
                dyn,
                state_abstraction,
                target_model,
                gp_bounds,
                prob_lower,
                prob_upper,
                srcact_idx,
            )
        end
    end

    prob = IntervalProbabilities(;
        lower = prob_lower,
        upper = prob_upper,
    )

    return prob
end

function source_action_transition_prob(
    dyn::AbstractedGaussianProcess,
    state_abstraction::StateUniformGridSplit,
    target_model::AbstractIMDPTarget,
    gp_bounds::AbstractedGaussianProcessRegion,
    prob_lower,
    prob_upper,
    srcact_idx,
)
    X = statespace(state_abstraction)

    # Transition to outside the partitioned region
    pl, pu = transition_prob_bounds(gp_bounds, X)
    prob_lower[end, srcact_idx] = 1.0 - pu
    prob_upper[end, srcact_idx] = 1.0 - pl

    # Transition to other states
    for (tar_idx, target_region) in enumerate(regions(state_abstraction))
        pl, pu = transition_prob_bounds(gp_bounds, target_region)

        if includetransition(target_model, pu)
            prob_lower[tar_idx, srcact_idx] = pl
            prob_upper[tar_idx, srcact_idx] = pu
        else  # Allow sparsifying via adding probability to the absorbing avoid state

            # Use clamp to ensure that the probabilities are within [0, 1] (due to floating point errors).
            prob_lower[end, srcact_idx] = clamp(prob_lower[end, srcact_idx] + pl, 0.0, 1.0)
            prob_upper[end, srcact_idx] = clamp(prob_upper[end, srcact_idx] + pu, 0.0, 1.0)
        end
    end
end

function transition_prob(
    dyn::AbstractedGaussianProcess,
    state_abstraction::StateUniformGridSplit,
    input_abstraction::InputAbstraction,
    stateptr,
    target_model::AbstractOrthogonalIMDPTarget,
)

    # The first state along each axis is absorbing, representing transitioning to outside the partitioned along that axis.
    ninputs = numinputs(input_abstraction)

    # Transition probabilities
    prob_lower, prob_upper = initprob(target_model, state_abstraction, ninputs)

    linear_indices = LinearIndices(splits(state_abstraction))
    Threads.@threads for Icart in CartesianIndices(splits(state_abstraction))
        Ilinear = linear_indices[Icart]
        srcact_idx = stateptr[Ilinear]

        # Sink states are implicitly endcoded

        # Other states
        source_region = regions(state_abstraction)[Ilinear]
        for input in inputs(input_abstraction)
            gp_bounds = bounds(dyn, source_region, input)

            source_action_transition_prob(
                dyn,
                state_abstraction,
                target_model,
                gp_bounds,
                prob_lower,
                prob_upper,
                srcact_idx,
            )

            srcact_idx += 1
        end
    end

    prob = OrthogonalIntervalProbabilities(
        Tuple(
            IntervalProbabilities(; lower = pl, upper = pu) for (pl, pu) in zip(prob_lower, prob_upper)
        ),
        Int32.(Tuple(splits(state_abstraction))),
    )

    return prob
end

function source_action_transition_prob(
    dyn::AbstractedGaussianProcess,
    state_abstraction::StateUniformGridSplit,
    target_model::AbstractOrthogonalIMDPTarget,
    gp_bounds::AbstractedGaussianProcessRegion,
    prob_lower,
    prob_upper,
    srcact_idx,
)
    X = statespace(state_abstraction)

    # For each axis... 
    for (axis, axisregions) in enumerate(splits(state_abstraction))
        # Transition to outside the partitioned region
        pl, pu = axis_transition_prob_bounds(gp_bounds, X, axis)
        prob_lower[axis][end, srcact_idx] = 1.0 - pu
        prob_upper[axis][end, srcact_idx] = 1.0 - pl

        # Transition to other states
        axis_statespace = Interval(low(X, axis), high(X, axis))
        split_axis = LazySets.split(axis_statespace, axisregions)

        for (tar_idx, target_region) in enumerate(split_axis)
            pl, pu = axis_transition_prob_bounds(gp_bounds, target_region, axis)

            if includetransition(target_model, pu)
                prob_lower[axis][tar_idx, srcact_idx] = pl
                prob_upper[axis][tar_idx, srcact_idx] = pu
            else  # Allow sparsifying via adding probability to the absorbing avoid state

                # Use clamp to ensure that the probabilities are within [0, 1] (due to floating point errors).
                prob_lower[axis][end, srcact_idx] =
                    clamp(prob_lower[axis][end, srcact_idx] + pl, 0.0, 1.0)
                prob_upper[axis][end, srcact_idx] =
                    clamp(prob_upper[axis][end, srcact_idx] + pu, 0.0, 1.0)
            end
        end
    end
end
