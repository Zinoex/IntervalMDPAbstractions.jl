using SparseArrays

export abstraction

"""
    abstraction(sys::AffineAdditiveNoise, state_abstraction::StateGridSplit, input_abstraction::InputAbstraction, target_model::Union{DirectIMDP, SparseDirectIMDP})

Abstract function for creating an abstraction of a system with additive noise.
"""
function abstraction(sys::AffineAdditiveNoise, state_abstraction::StateGridSplit, input_abstraction::InputAbstraction, target_model::Union{DirectIMDP, SparseDirectIMDP})
    # The first state is absorbing, representing transitioning to outside the partitioned.
    nregions = numregions(state_abstraction) + 1
    nactions = numactions(input_abstraction)

    prob_lower, prob_upper = direct_initprob(target_model, nregions, nactions)

    # Absorbing
    prob_lower[1, 1] = 1.0
    prob_upper[1, 1] = 1.0

    # Transition probabilities
    for (i, source_region) in enumerate(regions(state_abstraction))
        for (j, input) in enumerate(inputs(input_abstraction))
            Y = nominaldynamics(sys, source_region, input)

            # Transition to outside the partitioned region
            pl, pu = transition_prob_bounds(Y, statespace(state_abstraction), noise(sys))
            prob_lower[1, (i - 1) * nactions + j + 1] = 1.0 - pu
            prob_upper[1, (i - 1) * nactions + j + 1] = 1.0 - pl

            # Transition to other states
            for (k, target_region) in enumerate(regions(state_abstraction))
                pl, pu = transition_prob_bounds(Y, target_region, noise(sys))

                if includetransition(target_model, pu)
                    prob_lower[k + 1, (i - 1) * nactions + j + 1] = pl
                    prob_upper[k + 1, (i - 1) * nactions + j + 1] = pu
                else
                    prob_lower[1, (i - 1) * nactions + j + 1] += pl
                    prob_upper[1, (i - 1) * nactions + j + 1] += pu
                end
            end
        end
    end

    prob = IntervalProbabilities(prob_lower, prob_upper)

    # State pointer
    stateptr = [[1, 2]; (1:nregions) .* nactions .+ 2]

    return IntervalMarkovDecisionProcess(prob, stateptr)
end

function direct_initprob(target::DirectIMDP, nregions, nactions) 
    prob_lower = zeros(nregions, (nregions - 1) * nactions + 1)
    prob_upper = copy(prob_lower)

    return prob_lower, prob_upper
end

function direct_initprob(target::SparseDirectIMDP, nregions, nactions) 
    prob_lower = spzeros(nregions, (nregions - 1) * nactions + 1)
    prob_upper = copy(prob_lower)

    return prob_lower, prob_upper
end