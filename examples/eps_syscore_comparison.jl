module EpsSySCoReComparison

using BenchmarkTools
using IntervalMDP
using IntervalSySCoRe

using MAT, MAT.MAT_v4, MAT.MAT_v5, MAT.MAT_HDF5

include("systems/systems.jl")

struct EpsSySCoReComparisonProblem
    name::String

    constructor::Function

    state_splits
    input_split
    time_horizons
end


syscore_running_example = EpsSySCoReComparisonProblem(
    "running_example",
    (state_split, input_split, time_horizon) -> running_example_decoupled(time_horizon; range_vs_grid=:range, state_split=state_split, input_split=input_split),
    [(50, 50), (100, 100), (150, 150)],
    (3, 3),
    [1, 2, 5, 10, 20, 50, 100, 200]
)


bas_4d = EpsSySCoReComparisonProblem(
    "4d_bas",
    (state_split, input_split, time_horizon) -> building_automation_system_4d_decoupled(time_horizon; state_split=state_split, input_split=input_split),
    [(4, 4, 4, 4), (8, 8, 8, 8), (12, 12, 12, 12), (16, 16, 16, 16)],
    4,
    [1, 2, 5, 10, 20, 50, 100, 200]
)

function benchmark_intervalsyscore(problem::EpsSySCoReComparisonProblem, state_split, time_horizon)
    BenchmarkTools.gcscrub()

    mdp, lower_spec = problem.constructor(state_split, problem.input_split, time_horizon)
    lower_bound_prob = Problem(mdp, lower_spec)

    strategy, V_lower, k, res = control_synthesis(lower_bound_prob)

    upper_spec = Specification(system_property(lower_spec), !satisfaction_mode(lower_spec))
    upper_bound_prob = Problem(mdp, upper_spec, strategy)
    V_upper, k, res = value_iteration(upper_bound_prob)

    V_eps = V_upper - V_lower
    V_eps[terminal_states(system_property(lower_spec))] .= 0.0

    max_V_eps = maximum(V_eps)

    return max_V_eps
end

function benchmark_system(problem, state_split, time_horizon)
    println(("num_regions_per_axis", state_split, "time_horizon", time_horizon))
    max_V_eps = benchmark_intervalsyscore(problem, state_split, time_horizon)

    println(("max_V_eps", max_V_eps))
end

function benchmark_system(problem::EpsSySCoReComparisonProblem, time_horizon)
    for state_split in problem.state_splits
        benchmark_system(problem, state_split, time_horizon)
    end
end

function benchmark_system(problem::EpsSySCoReComparisonProblem)
    for time_horizon in problem.time_horizons
        benchmark_system(problem, time_horizon)
    end
end

end
