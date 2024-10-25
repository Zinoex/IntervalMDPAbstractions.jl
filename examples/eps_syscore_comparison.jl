module EpsSySCoReComparison

using BenchmarkTools
using IntervalMDP, IntervalSySCoRe
using Statistics

using MAT, MAT.MAT_v4, MAT.MAT_v5, MAT.MAT_HDF5

include("systems/systems.jl")

struct EpsSySCoReComparisonProblem
    name::String

    constructor::Function
    retime_prop::Function

    state_splits
    input_split
    time_horizons
end

car_parking = EpsSySCoReComparisonProblem(
    "car_parking",
    (state_split, input_split, time_horizon) -> car_parking_decoupled(time_horizon; range_vs_grid=:range, state_split=state_split, input_split=input_split),
    (prop, time_horizon) -> FiniteTimeReachAvoid(IntervalMDP.reach(prop), IntervalMDP.avoid(prop), time_horizon),
    [(50, 50), (100, 100), (150, 150), (300, 300), (600, 600)],
    (3, 3),
    [[1, 2]; collect(5:5:100)]
)


bas_4d = EpsSySCoReComparisonProblem(
    "4d_bas",
    (state_split, input_split, time_horizon) -> building_automation_system_4d_decoupled(time_horizon; state_split=state_split, input_split=input_split),
    (prop, time_horizon) -> FiniteTimeSafety(IntervalMDP.avoid(prop), time_horizon),
    [(4, 4, 4, 4), (8, 8, 8, 8), (12, 12, 12, 12), (16, 16, 16, 16)],
    4,
    [[1, 2]; collect(5:5:100)]
)

function retime_spec(prob, spec, time_horizon) 
    prop = prob.retime_prop(system_property(spec), time_horizon)
    return Specification(prop, satisfaction_mode(spec), strategy_mode(spec))
end

function benchmark_intervalsyscore(mdp, lower_spec, upper_spec, time_horizons)
    BenchmarkTools.gcscrub()

    @assert time_horizon(system_property(lower_spec)) == time_horizon(system_property(upper_spec))
    @assert time_horizon(system_property(lower_spec)) <= maximum(time_horizons)

    V_lowers = Vector{AbstractArray{Float64}}(undef, length(time_horizons))
    V_uppers = Vector{AbstractArray{Float64}}(undef, length(time_horizons))

    function lower_callback(V, k)
        for (i, time_horizon) in enumerate(time_horizons)
            if k == time_horizon
                V_lowers[i] = copy(V)
            end
        end
    end

    lower_bound_prob = Problem(mdp, lower_spec)
    strategy, _, k, res = control_synthesis(lower_bound_prob; callback=lower_callback)

    function upper_callback(V, k)
        for (i, time_horizon) in enumerate(time_horizons)
            if k == time_horizon
                V_uppers[i] = copy(V)
            end
        end
    end

    upper_bound_prob = Problem(mdp, upper_spec, strategy)
    _, k, res = value_iteration(upper_bound_prob; callback=upper_callback)

    tstates = terminal_states(system_property(lower_spec))

    V_epses = Vector{Vector{Float64}}(undef, length(time_horizons))

    for i in eachindex(time_horizons)
        V_lower = V_lowers[i]
        V_upper = V_uppers[i]

        V_eps = V_upper - V_lower
        # This is a bit of a hack, but it is a way to prune the terminal states,
        # aka. keep non-terminal states only (which is the hard part).
        # We only keep non-terminal states since only they are relevant for the
        # benchmark. 
        V_eps[tstates] .= -1.0
        V_eps = V_eps[V_eps .>= 0.0]

        V_epses[i] = V_eps
    end

    means = mean.(V_epses)
    maxs = maximum.(V_epses)

    return means, maxs
end

function construct_model(problem, state_split, time_horizon=1)
    BenchmarkTools.gcscrub()

    mdp, lower_spec, upper_spec = problem.constructor(state_split, problem.input_split, time_horizon)
    return mdp, lower_spec, upper_spec
end

function benchmark_system(problem, state_split, time_horizon)
    println(("num_regions_per_axis", state_split, "time_horizon", time_horizon))
    mdp, lower_spec, upper_spec = construct_model(problem, state_split, time_horizon)

    mean_V_eps, max_V_eps = benchmark_intervalsyscore(mdp, lower_spec, upper_spec, [time_horizon])

    println(("mean_V_eps", mean_V_eps[1], "max_V_eps", max_V_eps[1]))
end

function benchmark_system(problem::EpsSySCoReComparisonProblem, state_split)
    mdp, lower_spec, upper_spec = construct_model(problem, state_split)
    lower_spec = retime_spec(problem, lower_spec, maximum(problem.time_horizons))
    upper_spec = retime_spec(problem, upper_spec, maximum(problem.time_horizons))

    println(("num_regions_per_axis", state_split))

    mean_V_eps, max_V_eps = benchmark_intervalsyscore(mdp, lower_spec, upper_spec, problem.time_horizons)
    
    for (time_horizon, mean_V_ep, max_V_ep) in zip(problem.time_horizons, mean_V_eps, max_V_eps)
        println(("time_horizon", time_horizon, "mean_V_eps", mean_V_ep, "max_V_eps", max_V_ep))
    end
end

function benchmark_system(problem::EpsSySCoReComparisonProblem)
    for state_split in problem.state_splits
        benchmark_system(problem, state_split)
    end
end

end
