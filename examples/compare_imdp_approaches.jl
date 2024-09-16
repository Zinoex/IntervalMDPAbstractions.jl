module CompareIMDPApproaches

using BenchmarkTools, ProgressMeter
using DataFrames, CSV, JSON, Statistics
using IntervalMDP, IntervalSySCoRe

include("systems/systems.jl")

struct ComparisonProblem{N, M}
    name::String

    direct_constructor::Function
    decoupled_constructor::Function
    problem_constructor::Function
    post_process_value_function::Function
    impact_evaluator::Function

    include_impact::Bool

    state_split::NTuple{N, Int}
    input_split::Tuple{M, Int}
    time_horizon::Int
end

robot_2d_reachability = ComparisonProblem(
    "robot_2d_reachability",
    (state_split, input_split) -> robot_2d_direct(;spec=:reachability, sparse=true, state_split=state_split, input_split=input_split),
    (state_split, input_split) -> robot_2d_decoupled(;spec=:reachability, state_split=state_split, input_split=input_split),
    (mdp, reach, avoid, time_horizon) -> Problem(mdp, Specification(FiniteTimeReachAvoid(reach, avoid, time_horizon), Pessimistic, Maximize)),
    identity,
    () -> run_impact("ex_2Drobot-R-U"),
    true,
    (20, 20),
    (11, 11),
    10
)

robot_2d_reachavoid = ComparisonProblem(
    "robot_2d_reachavoid",
    (state_split, input_split) -> robot_2d_direct(;spec=:reachavoid, sparse=true, state_split=state_split, input_split=input_split),
    (state_split, input_split) -> robot_2d_decoupled(;spec=:reachavoid, state_split=state_split, input_split=input_split),
    (mdp, reach, avoid, time_horizon) -> Problem(mdp, Specification(FiniteTimeReachAvoid(reach, avoid, time_horizon), Pessimistic, Maximize)),
    identity,
    () -> run_impact("ex_2Drobot-RA-U"),
    true,
    (40, 40),
    (21, 21),
    10
)

syscore_running_example = ComparisonProblem(
    "syscore_running_example",
    (state_split, input_split) -> running_example_direct(;range_vs_grid=:range, sparse=true, state_split=state_split, input_split=input_split),
    (state_split, input_split) -> running_example_decoupled(;range_vs_grid=:range, state_split=state_split, input_split=input_split),
    (mdp, reach, avoid, time_horizon) -> Problem(mdp, Specification(FiniteTimeReachAvoid(reach, avoid, time_horizon), Pessimistic, Maximize)),
    identity,
    () -> run_impact("ex_syscore_running_example-RA"),
    true,
    (40, 40),
    (3, 3),
    10
)

van_der_pol = ComparisonProblem(
    "van_der_pol",
    (state_split, input_split) -> van_der_pol_direct(;state_split=state_split, input_split=input_split),
    (state_split, input_split) -> van_der_pol_decoupled(;state_split=state_split, input_split=input_split),
    (mdp, reach, avoid, time_horizon) -> Problem(mdp, Specification(FiniteTimeReachAvoid(reach, avoid, time_horizon), Pessimistic, Maximize)),
    identity,
    () -> run_impact("ex_van_der_pol-R"),
    true,
    (40, 40),
    (3, 3),
    10
)

bas4d = ComparisonProblem(
    "bas4d",
    (state_split, input_split) -> building_automation_system_4d_direct(;sparse=true, state_split=state_split, input_split=input_split),
    (state_split, input_split) -> building_automation_system_4d_decoupled(;sparse=true, state_split=state_split, input_split=input_split),
    (mdp, reach, avoid, time_horizon) -> Problem(mdp, Specification(FiniteTimeReachability(avoid, time_horizon), Optimistic, Minimize)),
    (V) -> 1.0 .- V,
    () -> run_impact("ex_4DBAS-S"),
    true,
    (4, 4, 6, 6),
    (4,),
    10
)

bas7d = ComparisonProblem(
    "bas7d",
    (state_split, input_split) -> building_automation_system_7d_direct(;sparse=true, state_split=state_split),
    (state_split, input_split) -> building_automation_system_7d_decoupled(;sparse=true, state_split=state_split),
    (mdp, reach, avoid, time_horizon) -> Problem(mdp, Specification(FiniteTimeReachability(avoid, time_horizon), Optimistic, Minimize)),
    (V) -> 1.0 .- V,
    () -> run_impact("ex_4DBAS-S"),
    true,
    (20, 20, 2, 2, 2, 2, 2),
    (1,),
    10
)

problems = [
    robot_2d_reachability,
    robot_2d_reachavoid,
    syscore_running_example,
    van_der_pol,
    bas4d,
    bas7d
]

function benchmark_impact(problem::ComparisonProblem)
    @info "Benchmarking IMPaCT"
    res = problem.impact_evaluator()

    return res
end

function benchmark_intervalsyscore(problem::ComparisonProblem, constructor::Function)
    @info "Measuring abstraction time"
    abstraction_time = @benchmark constructor($problem.state_split, $problem.input_split)
    abstraction_median_seconds = time(median(abstraction_time)) / 1e9
    @info "Abstraction time" abstraction_median_seconds

    mdp, reach, avoid = constructor(problem.state_split, problem.input_split)
    prob_mem = Base.summarysize(mdp)

    prob = problem.problem_constructor(mdp, reach, avoid, time_horizon)

    @info "Measuring certification time"
    certification_time = @benchmark value_iteration($prob)
    certification_median_seconds = time(median(certification_time)) / 1e9
    @info "Certification time" certification_median_seconds

    @info "Measuring peak mem"
    BenchmarkTools.gcscrub()
    GC.enable(false)
    V, k, res = value_iteration(prob)
    peak_mem = Base.gc_live_bytes() / 1000^2
    GC.enable(true)
    @info "Peak mem" peak_mem

    V = problem.post_process_value_function(V)

    return abstraction_median_seconds, certification_median_seconds, peak_mem, prob_mem, V
end

function to_impact_format(V)
    # Transpose to match row-major order of IMPaCT
    V = permutedims(V, reverse(1:ndims(V)))
    V = vec(V)

    # Remove reach regions (i.e with probability 1.0) to match IMPaCT
    V = V[V .< 1.0]

    return V
end

function benchmark_direct(problem::ComparisonProblem)
    @info "Benchmarking direct"
    try
        abstraction_time, certification_time, peak_mem, prob_mem, V = benchmark_intervalsyscore(problem, problem.direct_constructor)

        # Remove the first element of the value function, which is the absorbing avoid state
        V = reshape(V[2:end], state_split...)
        V = to_impact_format(V)
    
        return Dict(
            "oom" => false,
            "abstraction_time" => abstraction_time,
            "certification_time" => certification_time,
            "peak_mem" => peak_mem,
            "prob_mem" => prob_mem,
            "value_function" => V
        )
    catch e
        if isa(e, OutOfMemoryError)
            @warn "Direct failed due to OOM"
    
            return Dict(
                "oom" => true,
                "abstraction_time" => NaN,
                "certification_time" => NaN,
                "peak_mem" => NaN,
                "prob_mem" => NaN,
                "value_function" => NaN
            )
        else
            rethrow(e)
        end
    end
end

function benchmark_decoupled(problem::ComparisonProblem)
    @info "Benchmarking decoupled"
    try
        abstraction_time, certification_time, peak_mem, prob_mem, V = benchmark_intervalsyscore(problem, problem.direct_constructor)

        # Remove the first element of the value function, which is the absorbing avoid state
        V = V[(2:size(V, i) for i in ndims(V))...]
        V = to_impact_format(V)

        return Dict(
            "oom" => false,
            "abstraction_time" => abstraction_time,
            "certification_time" => certification_time,
            "peak_mem" => peak_mem,
            "prob_mem" => prob_mem,
            "value_function" => V
        )
    catch e
        if isa(e, OutOfMemoryError)
            @warn "Direct failed due to OOM"
    
            return Dict(
                "oom" => true,
                "abstraction_time" => NaN,
                "certification_time" => NaN,
                "peak_mem" => NaN,
                "prob_mem" => NaN,
                "value_function" => NaN
            )
        else
            rethrow(e)
        end
    end
end

function benchmark()
    @showprogress dt=1 desc="Benchmarking..." for problem in problems
        @info "Benchmarking problem: $(problem.name)"

        direct = benchmark_direct(state_split)
        decoupled = benchmark_decoupled(state_split)

        impact = if problem.include_impact
            benchmark_impact(problem)
        else
            Dict{String, Any}()
        end

        res = Dict(
            "name" => problem.name,
            "state_split" => problem.state_split,
            "input_split" => problem.input_split,
            "direct" => direct,
            "decoupled" => decoupled,
            "impact" => impact
        )

        save_results(problem.name, res)
    end
end


function save_results(name, res)
    mkpath("results/compare_imdp_approaches", mode=0o754)
    open("results/compare_imdp_approaches/$name.json", "w") do io
        JSON.print(io, res, 4)
    end
end

# function read_results()
#     df = CSV.read("results/direct_vs_decoupled_imdp.csv", DataFrame)
#     return df
# end

# TODO: plot results

# function compare()
#     res = benchmark()
#     df = to_dataframe(res)
#     save_results(df)
# end

end


CompareIMDPApproaches.benchmark()
