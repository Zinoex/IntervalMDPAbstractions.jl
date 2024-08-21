using LazySets, IntervalMDP, IntervalSySCoRe


function running_example_sys()
    A = 0.9I(2)
    B = 0.7I(2)
    w_stddev = I(2)

    dyn = AffineAdditiveNoiseDynamics(A, B, AdditiveDiagonalGaussianNoise(w_stddev))

    initial_region = EmptySet(2)
    reach_region = Hyperrectangle(; low=[4.0, -4.0], high=[10.0, 0.0])
    avoid_region = Hyperrectangle(; low=[4.0, 0.0], high=[10.0, 4.0])

    sys = System(dyn, initial_region, reach_region, avoid_region)

    return sys
end


function running_example()
    sys = running_example_sys()

    X = Hyperrectangle(; low=[-10.0, -10.0], high=[10.0, 10.0])
    l = [20, 20]
    state_abs = StateGridSplit(X, l)

    U = Hyperrectangle(; low=[-1.0, -1.0], high=[1.0, 1.0])
    input_abs = InputGridSplit(U, [3, 3])

    target_model = DecoupledIMDP()

    X1_split = split(X1, l[1] + 1)
    X2_split = split(X2, l[2] + 1)

    X_split = Matrix{LazySet}(undef, l[1] + 1, l[2] + 1)
    for (j, x2) in enumerate(X2_split)
        for (i, x1) in enumerate(X1_split)
            if i == 1 && j == 1
                X_split[i, j] = CartesianProduct(
                    Complement(Interval(low(x1)[1], high(x1)[1])),
                    Complement(Interval(low(x2)[1], high(x2)[1]))
                )
            elseif i == 1
                X_split[i, j] = CartesianProduct(
                    Complement(Interval(low(x1)[1], high(x1)[1])),
                    Interval(low(x2)[1], high(x2)[1])
                )
            elseif j == 1
                X_split[i, j] = CartesianProduct(
                    Interval(low(x1)[1], high(x1)[1]),
                    Complement(Interval(low(x2)[1], high(x2)[1]))
                )
            else
                X_split[i, j] = Hyperrectangle([center(x1)[1], center(x2)[1]], [radius_hyperrectangle(x1)[1], radius_hyperrectangle(x2)[1]])
            end

            X_split[i, j] = Hyperrectangle([center(x1)[1], center(x2)[1]], [radius_hyperrectangle(x1)[1], radius_hyperrectangle(x2)[1]])
        end
    end

    U_split = split(U, [3, 3])

    transition_prob(x, v_lower, v_upper) = 0.5 * erf((x - v_upper) * invsqrt2, (x - v_lower) * invsqrt2)

    probs1_lower = zeros(l[1] + 1, l[1] + 1)
    probs1_upper = zeros(l[1] + 1, l[1] + 1)
    for source in 1:l[1] + 1
        if source == 1
            probs1_upper[source, source] = 1
            probs1_lower[source, source] = 1
        else
            for target in 1:l[1] + 1
                if target == 1
                    probs1_upper[target, source] = max(
                        1 - transition_prob(low(X1_split[source])[1], low(X)[1], high(X)[1]),
                        1 - transition_prob(high(X1_split[source])[1], low(X)[1], high(X)[1])
                    )
                    probs1_lower[target, source] = min(
                        1 - transition_prob(center(X1_split[source])[1], low(X)[1], high(X)[1]),
                        1 - transition_prob(low(X1_split[source])[1], low(X)[1], high(X)[1]),
                        1 - transition_prob(high(X1_split[source])[1], low(X)[1], high(X)[1])
                    )
                elseif source == target
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
    end
    probs1 = IntervalProbabilities(; lower=probs1_lower, upper=probs1_upper)
    mc1 = IntervalMarkovChain(probs1)

    probs2_lower = zeros(l[2] + 1, l[2] + 1)
    probs2_upper = zeros(l[2] + 1, l[2] + 1)
    for source in 1:l[2] + 1
        if source == 1
            probs2_upper[source, source] = 1
            probs2_lower[source, source] = 1
        else
            for target in 1:l[2] + 1
                if target == 1
                    probs2_upper[target, source] = max(
                        1 - transition_prob(low(X2_split[source])[1], low(X)[2], high(X)[2]),
                        1 - transition_prob(high(X2_split[source])[1], low(X)[2], high(X)[2])
                    )
                    probs2_lower[target, source] = min(
                        1 - transition_prob(center(X2_split[source])[1], low(X)[2], high(X)[2]),
                        1 - transition_prob(low(X2_split[source])[1], low(X)[2], high(X)[2]),
                        1 - transition_prob(high(X2_split[source])[1], low(X)[2], high(X)[2])
                    )
                elseif source == target
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
    end
    probs2 = IntervalProbabilities(; lower=probs2_lower, upper=probs2_upper)
    mc2 = IntervalMarkovChain(probs2)

    noise_mc = ParallelProduct([mc1, mc2])

    stateptr = Int32[1]

    colptr = Int32[1]
    rowval = Int32[]

    reach = Tuple{Int32, Int32}[]
    avoid = Tuple{Int32, Int32}[]
    
    for j in 1:l[2] + 1
        for i in 1:l[1] + 1
            Xij = X_split[i, j]
            
            if j == 1 || i == 1 || !isdisjoint(Xij, avoid_region)
                push!(avoid, (i, j))
            elseif Xij ⊆ reach_region
                push!(reach, (i, j))
            end

            if i == 1 || j == 1
                push!(rowval, (j - 1) * (l[1] + 1) + i)
                push!(colptr, length(rowval) + 1)
            else
                for u in U_split
                    Xij_u = A * Xij + B * u
                    
                    for j2 in 1:l[2] + 1
                        for i2 in 1:l[1] + 1
                            if !isdisjoint(Xij_u, X_split[i2, j2])
                                push!(rowval, (j2 - 1) * (l[1] + 1) + i2)
                            end
                        end
                    end

                    push!(colptr, length(rowval) + 1)
                end
            end

            push!(stateptr, length(colptr))
        end
    end

    transitions = Transitions(colptr, rowval, (prod(l .+ 1), length(colptr) - 1))
    deterministic_mdp = DeterministicMarkovDecisionProcess(transitions, stateptr)

    final_mdp = Sequential([MultiDim(deterministic_mdp, Tuple(l .+ 1)), noise_mc])

    return final_mdp, reach, avoid
end

function running_example_direct(; sparse=false, range_vs_grid=:grid, state_split=[20, 20], input_split=[3, 3])
    sys = running_example_sys()

    X = Hyperrectangle(; low=[-10.0, -10.0], high=[10.0, 10.0])
    state_abs = StateGridSplit(X, state_split)

    U = Hyperrectangle(; low=[-1.0, -1.0], high=[1.0, 1.0])
    if range_vs_grid == :range
        input_abs = InputLinRange(U, input_split)
    elseif range_vs_grid == :grid
        input_abs = InputGridSplit(U, input_split)
    else
        throw(ArgumentError("Invalid range_vs_grid argument"))
    end

    if sparse
        target_model = SparseDirectIMDP()
    else
        target_model = DirectIMDP()
    end

    mdp, reach, avoid = abstraction(sys, state_abs, input_abs, target_model)

    return mdp, reach, avoid
end

function main()
    mdp_direct, reach_direct, avoid_direct = running_example_direct()
    prop_direct = FiniteTimeReachAvoid(reach_direct, avoid_direct, 10)
    spec_direct = Specification(prop_direct, Pessimistic, Maximize)
    prob_direct = Problem(mdp_direct, spec_direct)

    V, k, res = value_iteration(prob_direct)

    println(V, k, res)

end