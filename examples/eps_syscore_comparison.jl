module EpsSySCoReComparison

using BenchmarkTools
using IntervalMDP
using IntervalSySCoRe

using MAT, MAT.MAT_v4, MAT.MAT_v5, MAT.MAT_HDF5

include("systems/systems.jl")

struct EpsSySCoReComparisonProblem
    name::String

    constructor::Function
    problem_constructor::Function
    other_bound_problem_constructor::Function
    post_process_value_function::Function

    state_splits
    input_split
    time_horizon
end


syscore_running_example = EpsSySCoReComparisonProblem(
    "running_example",
    (state_split, input_split) -> running_example_decoupled(;range_vs_grid=:range, state_split=state_split, input_split=input_split),
    (mdp, reach, avoid, time_horizon) -> Problem(mdp, Specification(FiniteTimeReachAvoid(reach, avoid, time_horizon), Pessimistic, Maximize)),
    (mdp, reach, avoid, time_horizon, strategy) -> Problem(mdp, Specification(FiniteTimeReachAvoid(reach, avoid, time_horizon), Optimistic), strategy),
    identity,
    [(50, 50), (100, 100), (150, 150)],
    (3, 3),
    10
)


bas_4d = EpsSySCoReComparisonProblem(
    "4d_bas",
    (state_split, input_split) -> building_automation_system_4d_decoupled(;state_split=state_split, input_split=input_split),
    (mdp, reach, avoid, time_horizon) -> Problem(mdp, Specification(FiniteTimeReachability(avoid, time_horizon), Optimistic, Minimize)),
    (mdp, reach, avoid, time_horizon, strategy) -> Problem(mdp, Specification(FiniteTimeReachability(avoid, time_horizon), Pessimistic), strategy),
    (V) -> 1.0 .- V,
    [(4, 4, 4, 4), (8, 8, 8, 8), (12, 12, 12, 12), (16, 16, 16, 16)],
    4,
    10
)

function benchmark_intervalsyscore(problem::EpsSySCoReComparisonProblem, state_split, time_horizon)
    BenchmarkTools.gcscrub()

    mdp, reach, avoid = problem.constructor(state_split, problem.input_split)
    prob = problem.problem_constructor(mdp, reach, avoid, time_horizon)

    strategy, V_lower, k, res = control_synthesis(prob)
    V_lower = problem.post_process_value_function(V_lower)
    V_lower = V_lower[(2:size(V_lower, i) for i in 1:ndims(V_lower))...]

    upper_bound_prob = problem.other_bound_problem_constructor(mdp, reach, avoid, time_horizon, strategy)
    V_upper, k, res = value_iteration(upper_bound_prob)
    V_upper = problem.post_process_value_function(V_upper)
    V_upper = V_upper[(2:size(V_upper, i) for i in 1:ndims(V_upper))...]

    V_eps = V_upper - V_lower

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
    for time_horizon in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        benchmark_system(problem, time_horizon)
    end
end

end
