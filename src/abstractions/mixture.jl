function transition_prob(dyn::StochasticSwitchedDynamics, state_abstraction::StateUniformGridSplit, input_abstraction::InputAbstraction, stateptr, target_model::AbstractIMDPTarget)
    transition_probs = map(dyn.dynamics) do mixture_dyn
        return transition_prob(mixture_dyn, state_abstraction, input_abstraction, stateptr, target_model)
    end

    l = sum(zip(transition_probs, dyn.weights)) do (prob, w)
        lower(prob) .* w
    end

    u = sum(zip(transition_probs, dyn.weights)) do (prob, w)
        upper(prob) .* w
    end

    prob = IntervalProbabilities(; lower=l, upper=u)

    return prob
end

function transition_prob(dyn::StochasticSwitchedDynamics, state_abstraction::StateUniformGridSplit, input_abstraction::InputAbstraction, stateptr, target_model::AbstractMixtureIMDPTarget)
    transition_probs = map(dyn.dynamics) do mixture_dyn
        return transition_prob(mixture_dyn, state_abstraction, input_abstraction, stateptr, mixture_target(target_model))
    end

    weight_matrix = repeat(dyn.weights, 1, last(stateptr) - 1)
    weighting_probs = IntervalProbabilities(; lower=weight_matrix, upper=weight_matrix)

    prob = MixtureIntervalProbabilities(ntuple(i -> transition_probs[i], length(transition_probs)), weighting_probs)

    return prob
end