using SparseArrays

function transition_prob(
    dyn::AdditiveNoiseDynamics,
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

    # Transition probabilities
    prepare_nominal(dyn, input_abstraction)

    Threads.@threads for (i, source_region) in
                         collect(enumerate(regions(state_abstraction)))
        for (j, input) in enumerate(inputs(input_abstraction))
            srcact_idx = (i - 1) * ninputs + j
            Y = nominal(dyn, source_region, input)

            source_action_transition_prob(
                dyn,
                state_abstraction,
                target_model,
                Y,
                prob_lower,
                prob_upper,
                srcact_idx,
            )
        end
    end

    prob_lower, prob_upper = postprocessprob(target_model, prob_lower, prob_upper)

    prob = IntervalProbabilities(;
        lower = prob_lower,
        upper = prob_upper,
    )

    return prob
end

function source_action_transition_prob(
    dyn::AdditiveNoiseDynamics,
    state_abstraction::StateUniformGridSplit,
    target_model::AbstractIMDPTarget,
    Y::LazySet,
    prob_lower,
    prob_upper,
    srcact_idx,
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
            prob_lower[tar_idx, srcact_idx] = pl
            prob_upper[tar_idx, srcact_idx] = pu
        else  # Allow sparsifying via adding probability to the absorbing avoid state
            pl_outside = pl_outside + pl
            pu_outside = pu_outside + pu
        end
    end

    # Use clamp to ensure that the probabilities are within [0, 1] (due to floating point errors).
    prob_lower[end, srcact_idx] = clamp(pl_outside, 0.0, 1.0)
    prob_upper[end, srcact_idx] = clamp(pu_outside, 0.0, 1.0)
end

function transition_prob(
    dyn::AdditiveNoiseDynamics,
    state_abstraction::StateUniformGridSplit,
    input_abstraction::InputAbstraction,
    stateptr,
    target_model::AbstractOrthogonalIMDPTarget,
)
    if !candecouple(noise(dyn))
        throw(
            ArgumentError(
                "Cannot decouple dynamics with non-diagonal noise covariance matrix",
            ),
        )
    end
    prepare_nominal(dyn, input_abstraction)

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
            # To decouple, we need to construct a hyperrectangle around the nominal one-step reachable region
            Yhat = nominal(dyn, source_region, input)
            Y = box_approximation(Yhat)

            source_action_transition_prob(
                dyn,
                state_abstraction,
                target_model,
                Y,
                prob_lower,
                prob_upper,
                srcact_idx,
            )

            srcact_idx += 1
        end
    end

    prob_lower, prob_upper = postprocessprob(target_model, prob_lower, prob_upper)

    prob = OrthogonalIntervalProbabilities(
        Tuple(
            IntervalProbabilities(; lower = pl, upper = pu) for (pl, pu) in zip(prob_lower, prob_upper)
        ),
        Int32.(Tuple(splits(state_abstraction))),
    )

    return prob
end

function source_action_transition_prob(
    dyn::AdditiveNoiseDynamics,
    state_abstraction::StateUniformGridSplit,
    target_model::AbstractOrthogonalIMDPTarget,
    Y::Hyperrectangle,
    prob_lower,
    prob_upper,
    srcact_idx,
)
    w = noise(dyn)
    X = statespace(state_abstraction)

    # For each axis... 
    for (axis, axisregions) in enumerate(splits(state_abstraction))
        # Transition to outside the partitioned region
        pl_outside, pu_outside = axis_transition_prob_bounds(Y, X, w, axis)
        pl_outside, pu_outside = 1.0 - pu_outside, 1.0 - pl_outside

        # Transition to other states
        axis_statespace = Interval(low(X, axis), high(X, axis))
        split_axis = LazySets.split(axis_statespace, axisregions)

        for (tar_idx, target_region) in enumerate(split_axis)
            pl, pu = axis_transition_prob_bounds(Y, target_region, w, axis)

            if includetransition(target_model, pu)
                prob_lower[axis][tar_idx, srcact_idx] = pl
                prob_upper[axis][tar_idx, srcact_idx] = pu
            else  # Allow sparsifying via adding probability to the absorbing avoid state
                pl_outside = pl_outside + pl
                pu_outside = pu_outside + pu
            end
        end

        # Use clamp to ensure that the probabilities are within [0, 1] (due to floating point errors).
        prob_lower[axis][end, srcact_idx] = clamp(pl_outside, 0.0, 1.0)
        prob_upper[axis][end, srcact_idx] = clamp(pu_outside, 0.0, 1.0)
    end
end
