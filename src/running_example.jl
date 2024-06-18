
function test()
    A = 0.9I(2)
    B = 0.7I(2)
    # w_mean = I(2), 0.0I(2)

    X = Hyperrectangle(; low=[-10.0, -10.0], high=[10.0, 10.0])
    X1 = Interval(-10.0, 10.0)
    X2 = Interval(-10.0, 10.0)
    U = Hyperrectangle(; low=[-1.0, -1.0], high=[1.0, 1.0])

    reach_region = Hyperrectangle(; low=[4.0, -4.0], high=[10.0, 0.0])
    avoid_region = Hyperrectangle(; low=[4.0, 0.0], high=[10.0, 4.0])

    l = [20, 20]
    X1_split = split(X1, l[1])
    X2_split = split(X2, l[2])

    X_split = Matrix{Hyperrectangle}(undef, l[1], l[2])
    for (j, x2) in enumerate(X2_split)
        for (i, x1) in enumerate(X1_split)
            X_split[i, j] = Hyperrectangle([center(x1)[1], center(x2)[1]], [radius_hyperrectangle(x1)[1], radius_hyperrectangle(x2)[1]])
        end
    end

    U_split = split(U, [3, 3])

    transition_prob(x, v_lower, v_upper) = 0.5 * erf((x - v_upper) * invsqrt2, (x - v_lower) * invsqrt2)

    probs1_lower = Matrix{Float64}(undef, l[1], l[1])
    probs1_upper = Matrix{Float64}(undef, l[1], l[1])
    for source in 1:l[1]
        for target in 1:l[1]
            if source == target
                probs1_upper[target, source] = transition_prob(center(X1_split[source])[1], low(X1_split[target])[1], high(X1_split[target])[1])
                probs1_lower[target, source] = transition_prob(low(X1_split[source])[1], low(X1_split[target])[1], high(X1_split[target])[1])
            elseif source < target
                probs1_upper[target, source] = transition_prob(high(X1_split[source])[1], low(X1_split[target])[1], high(X1_split[target])[1])
                probs1_lower[target, source] = transition_prob(low(X1_split[source])[1], low(X1_split[target])[1], high(X1_split[target])[1])
            else
                probs1_upper[target, source] = transition_prob(low(X1_split[source])[1], low(X1_split[target])[1], high(X1_split[target])[1])
                probs1_lower[target, source] = transition_prob(high(X1_split[source])[1], low(X1_split[target])[1], high(X1_split[target])[1])
            end
        end
    end
    # TODO: Add a sink state that is added to the avoid set

    probs1 = IntervalProbabilities(; lower=probs1_lower, upper=probs1_upper)
    mc1 = IntervalMarkovChain(probs1)

    probs2_lower = Matrix{Float64}(undef, l[1], l[1])
    probs2_upper = Matrix{Float64}(undef, l[1], l[1])
    for source in 1:l[1]
        for target in 1:l[1]
            if source == target
                probs2_upper[target, source] = transition_prob(center(X2_split[source])[1], low(X2_split[target])[1], high(X2_split[target])[1])
                probs2_lower[target, source] = transition_prob(low(X2_split[source])[1], low(X2_split[target])[1], high(X2_split[target])[1])
            elseif source < target
                probs2_upper[target, source] = transition_prob(high(X2_split[source])[1], low(X2_split[target])[1], high(X2_split[target])[1])
                probs2_lower[target, source] = transition_prob(low(X2_split[source])[1], low(X2_split[target])[1], high(X2_split[target])[1])
            else
                probs2_upper[target, source] = transition_prob(low(X2_split[source])[1], low(X2_split[target])[1], high(X2_split[target])[1])
                probs2_lower[target, source] = transition_prob(high(X2_split[source])[1], low(X2_split[target])[1], high(X2_split[target])[1])
            end
        end
    end
    probs2 = IntervalProbabilities(; lower=probs2_lower, upper=probs2_upper)
    mc2 = IntervalMarkovChain(probs2)

    noise_mc = ParallelProduct([mc1, mc2])

    stateptr = Int32[0; length(U_split) * UnitRange{Int32}(1, l[1] * l[2])] .+ one(Int32)

    colptr = Int32[1]
    rowval = Int32[]

    reach = Tuple{Int32, Int32}[]
    avoid = Tuple{Int32, Int32}[]
    
    for j in 1:l[2]
        for i in 1:l[1]
            Xij = X_split[i, j]
            
            if !isdisjoint(Xij, avoid_region)
                push!(avoid, (i, j))
            elseif !isdisjoint(Xij, reach_region)
                push!(reach, (i, j))
            end

            for u in U_split
                Xij_u = A * Xij + B * u
                
                for j2 in 1:l[2]
                    for i2 in 1:l[1]
                        if !isdisjoint(Xij_u, X_split[i2, j2])
                            push!(rowval, (j2 - 1) * l[1] + i2 + 1)
                        end
                    end
                end

                push!(colptr, length(rowval) + 1)
            end
        end
    end

    transitions = Transitions(colptr, rowval, (prod(l), length(colptr) - 1))
    deterministic_mdp = DeterministicMarkovDecisionProcess(transitions, stateptr)

    final_mdp = Sequential([MultiDim(deterministic_mdp, Tuple(l)), noise_mc])

    return final_mdp, reach, avoid
end
