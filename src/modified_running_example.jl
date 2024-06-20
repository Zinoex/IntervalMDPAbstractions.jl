
function modified_running_example()
    A = 0.9I(2)
    B = 0.7I(2)
    sigma = 2.0

    X = Hyperrectangle(; low=[-10.0, -10.0], high=[10.0, 10.0])
    X1 = Interval(-10.0, 10.0)
    X2 = Interval(-10.0, 10.0)
    U = Hyperrectangle(; low=[-1.0, -1.0], high=[1.0, 1.0])

    reach_region = Hyperrectangle(; low=[4.0, -6.0], high=[10.0, -2.0])

    l = [5, 5]
    X1_split = split(X1, l[1])
    X2_split = split(X2, l[2])

    X_split = Matrix{LazySet}(undef, l[1] + 1, l[2] + 1)
    for j in 1:l[2] + 1
        for i in 1:l[1] + 1
            if i == 1 || j == 1
                X_split[i, j] = Complement(X)  # We can only do this because we know that the regions is going to the avoid set.
            else
                x1 = X1_split[i - 1]
                x2 = X2_split[j - 1]
                X_split[i, j] = Hyperrectangle([center(x1)[1], center(x2)[1]], [radius_hyperrectangle(x1)[1], radius_hyperrectangle(x2)[1]])
            end
        end
    end

    U_split = split(U, [3, 3])

    transition_prob(x, v_lower, v_upper) = 0.5 * erf((x - v_upper) * invsqrt2 / sigma, (x - v_lower) * invsqrt2 / sigma)

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
                        1 - transition_prob(low(X1_split[source - 1])[1], low(X)[1], high(X)[1]),
                        1 - transition_prob(high(X1_split[source - 1])[1], low(X)[1], high(X)[1])
                    )
                    probs1_lower[target, source] = min(
                        1 - transition_prob(center(X1_split[source - 1])[1], low(X)[1], high(X)[1]),
                        1 - transition_prob(low(X1_split[source - 1])[1], low(X)[1], high(X)[1]),
                        1 - transition_prob(high(X1_split[source - 1])[1], low(X)[1], high(X)[1])
                    )
                else
                    probs1_upper[target, source] = max(
                        transition_prob(center(X1_split[source - 1])[1], low(X1_split[target - 1])[1], high(X1_split[target - 1])[1]),
                        transition_prob(low(X1_split[source - 1])[1], low(X1_split[target - 1])[1], high(X1_split[target - 1])[1]),
                        transition_prob(high(X1_split[source - 1])[1], low(X1_split[target - 1])[1], high(X1_split[target - 1])[1])
                    )
                    probs1_lower[target, source] = min(
                        transition_prob(low(X1_split[source - 1])[1], low(X1_split[target - 1])[1], high(X1_split[target - 1])[1]),
                        transition_prob(high(X1_split[source - 1])[1], low(X1_split[target - 1])[1], high(X1_split[target - 1])[1])
                    )
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
                    probs2_upper[target, source] = 1 - min(
                        transition_prob(low(X2_split[source - 1])[1], low(X)[2], high(X)[2]),
                        transition_prob(high(X2_split[source - 1])[1], low(X)[2], high(X)[2])
                    )
                    probs2_lower[target, source] = 1 - max(
                        transition_prob(center(X2_split[source - 1])[1], low(X)[2], high(X)[2]),
                        transition_prob(low(X2_split[source - 1])[1], low(X)[2], high(X)[2]),
                        transition_prob(high(X2_split[source - 1])[1], low(X)[2], high(X)[2])
                    )
                else
                    probs2_upper[target, source] = max(
                        transition_prob(center(X2_split[source - 1])[1], low(X2_split[target - 1])[1], high(X2_split[target - 1])[1]),
                        transition_prob(low(X2_split[source - 1])[1], low(X2_split[target - 1])[1], high(X2_split[target - 1])[1]),
                        transition_prob(high(X2_split[source - 1])[1], low(X2_split[target - 1])[1], high(X2_split[target - 1])[1])
                    )
                    probs2_lower[target, source] = min(
                        transition_prob(low(X2_split[source - 1])[1], low(X2_split[target - 1])[1], high(X2_split[target - 1])[1]),
                        transition_prob(high(X2_split[source - 1])[1], low(X2_split[target - 1])[1], high(X2_split[target - 1])[1])
                    )
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

            if j == 1 || i == 1
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

function modified_running_example_direct()
    A = 0.9I(2)
    B = 0.7I(2)
    sigma = 2.0

    X = Hyperrectangle(; low=[-10.0, -10.0], high=[10.0, 10.0])
    X1 = Interval(-10.0, 10.0)
    X2 = Interval(-10.0, 10.0)
    U = Hyperrectangle(; low=[-1.0, -1.0], high=[1.0, 1.0])

    reach_region = Hyperrectangle(; low=[4.0, -6.0], high=[10.0, -2.0])

    l = [5, 5]
    X1_split = split(X1, l[1])
    X2_split = split(X2, l[2])

    X_split = Matrix{LazySet}(undef, l[1] + 1, l[2] + 1)
    for j in 1:l[2] + 1
        for i in 1:l[1] + 1
            if i == 1 && j == 1
                X_split[i, j] = CartesianProduct(
                    Complement(Interval(low(X, 1), high(X, 1))),
                    Complement(Interval(low(X, 2), high(X, 2)))
                )
            elseif i == 1
                X_split[i, j] = CartesianProduct(
                    Complement(Interval(low(X, 1), high(X, 1))),
                    Interval(low(X, 2), high(X, 2))
                )
            elseif j == 1
                X_split[i, j] = CartesianProduct(
                    Interval(low(X, 1), high(X, 1)),
                    Complement(Interval(low(X, 2), high(X, 2)))
                )
            else
                x1 = X1_split[i - 1]
                x2 = X2_split[j - 1]
                X_split[i, j] = Hyperrectangle([center(x1)[1], center(x2)[1]], [radius_hyperrectangle(x1)[1], radius_hyperrectangle(x2)[1]])
            end
        end
    end

    U_split = split(U, [3, 3])

    transition_prob(x, v_lower, v_upper) = 0.5 * erf((x - v_upper) * invsqrt2 / sigma, (x - v_lower) * invsqrt2 / sigma)

    probs = IntervalProbabilities{Float64, Vector{Float64}, Matrix{Float64}}[]
    for source2 in 1:l[2] + 1
        for source1 in 1:l[1] + 1
            source = (source2 - 1) * (l[1] + 1) + source1

            probs_lower = Vector{Float64}[]
            probs_upper = Vector{Float64}[]

            if source1 == 1 || source2 == 1
                prob_upper = zeros(prod(l .+ 1))
                prob_lower = zeros(prod(l .+ 1))

                prob_upper[source] = 1
                prob_lower[source] = 1

                push!(probs_lower, prob_lower)
                push!(probs_upper, prob_upper)
            else
                Xij = X_split[source1, source2]

                for u in U_split
                    Xij_u = A * Xij + B * u
                    box_Xij_u = box_approximation(Xij_u)

                    prob_upper = zeros(prod(l .+ 1))
                    prob_lower = zeros(prod(l .+ 1))

                    for target2 in 1:l[2] + 1
                        for target1 in 1:l[1] + 1
                            Xij_target = X_split[target1, target2]
                            target = (target2 - 1) * (l[1] + 1) + target1

                            if target1 == 1 && target2 == 1
                                prob_upper[target] = max(
                                    1 - transition_prob(low(box_Xij_u)[1], low(X)[1], high(X)[1]),
                                    1 - transition_prob(high(box_Xij_u)[1], low(X)[1], high(X)[1])
                                ) * max(
                                    1 - transition_prob(low(box_Xij_u)[2], low(X)[2], high(X)[2]),
                                    1 - transition_prob(high(box_Xij_u)[2], low(X)[2], high(X)[2])
                                )
                                prob_lower[target] = minimum(vertices_list(box_Xij_u)) do v
                                    (1 - transition_prob(v[1], low(X)[1], high(X)[1])) * (1 - transition_prob(v[2], low(X)[2], high(X)[2]))
                                end
                            elseif target1 == 1
                                prob_upper[target] = max(
                                    1 - transition_prob(low(box_Xij_u)[1], low(X)[1], high(X)[1]),
                                    1 - transition_prob(high(box_Xij_u)[1], low(X)[1], high(X)[1])
                                ) * max(
                                    transition_prob(center(box_Xij_u)[2], low(Xij_target.Y)[1], high(Xij_target.Y)[1]),
                                    transition_prob(low(box_Xij_u)[2], low(Xij_target.Y)[1], high(Xij_target.Y)[1]),
                                    transition_prob(high(box_Xij_u)[2], low(Xij_target.Y)[1], high(Xij_target.Y)[1])
                                )
                                prob_lower[target] = minimum(vertices_list(box_Xij_u)) do v
                                    (1 - transition_prob(v[1], low(X)[1], high(X)[1])) * transition_prob(v[2], low(Xij_target.Y)[1], high(Xij_target.Y)[1])
                                end
                            elseif target2 == 1
                                prob_upper[target] = max(
                                    transition_prob(center(box_Xij_u)[1], low(Xij_target.X)[1], high(Xij_target.X)[1]),
                                    transition_prob(low(box_Xij_u)[1], low(Xij_target.X)[1], high(Xij_target.X)[1]),
                                    transition_prob(high(box_Xij_u)[1], low(Xij_target.X)[1], high(Xij_target.X)[1])
                                ) * max(
                                    1 - transition_prob(low(box_Xij_u)[2], low(X)[2], high(X)[2]),
                                    1 - transition_prob(high(box_Xij_u)[2], low(X)[2], high(X)[2])
                                )
                                prob_lower[target] = minimum(vertices_list(box_Xij_u)) do v
                                    transition_prob(v[1], low(Xij_target.X)[1], high(Xij_target.X)[1]) * (1 - transition_prob(v[2], low(X)[2], high(X)[2]))
                                end
                            else
                                prob_upper[target] = max(
                                    transition_prob(center(box_Xij_u)[1], low(Xij_target)[1], high(Xij_target)[1]),
                                    transition_prob(low(box_Xij_u)[1], low(Xij_target)[1], high(Xij_target)[1]),
                                    transition_prob(high(box_Xij_u)[1], low(Xij_target)[1], high(Xij_target)[1])
                                ) * max(
                                    transition_prob(center(box_Xij_u)[2], low(Xij_target)[2], high(Xij_target)[2]),
                                    transition_prob(low(box_Xij_u)[2], low(Xij_target)[2], high(Xij_target)[2]),
                                    transition_prob(high(box_Xij_u)[2], low(Xij_target)[2], high(Xij_target)[2])
                                )
                                prob_lower[target] = minimum(vertices_list(box_Xij_u)) do v
                                    transition_prob(v[1], low(Xij_target)[1], high(Xij_target)[1]) * transition_prob(v[2], low(Xij_target)[2], high(Xij_target)[2])
                                end
                            end
                        end
                    end

                    push!(probs_lower, prob_lower)
                    push!(probs_upper, prob_upper)
                    
                end
            end

            prob = IntervalProbabilities(; lower=reduce(hcat, probs_lower), upper=reduce(hcat, probs_upper))
            push!(probs, prob)
        end
    end
    mdp = IntervalMarkovDecisionProcess(probs)

    reach = Int32[]
    avoid = Int32[]
    
    for source2 in 1:l[2] + 1
        for source1 in 1:l[1] + 1
            Xij = X_split[source1, source2]
            source = (source2 - 1) * (l[1] + 1) + source1
            
            if source1 == 1 || source2 == 1
                push!(avoid, source)
            elseif Xij ⊆ reach_region
                push!(reach, source)
            end
        end
    end

    return mdp, reach, avoid
end