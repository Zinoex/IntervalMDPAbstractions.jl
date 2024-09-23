module IsolatedCompareIMDPApproaches

using BenchmarkTools
using IntervalMDP
using IntervalSySCoRe

include("systems/systems.jl")

struct IntervalSySCoReComparisonProblem{N, M}
    name::String

    direct_constructor::Function
    decoupled_constructor::Function
    problem_constructor::Function
    post_process_value_function::Function

    state_split::NTuple{N, Int}
    input_split::NTuple{M, Int}
    time_horizon::Int
end

robot_2d_reachability = IntervalSySCoReComparisonProblem(
    "robot_2d_reachability",
    (state_split, input_split) -> robot_2d_direct(;spec=:reachability, sparse=true, state_split=state_split, input_split=input_split),
    (state_split, input_split) -> robot_2d_decoupled(;spec=:reachability, state_split=state_split, input_split=input_split),
    (mdp, reach, avoid, time_horizon) -> Problem(mdp, Specification(FiniteTimeReachAvoid(reach, avoid, time_horizon), Pessimistic, Maximize)),
    identity,
    (20, 20),
    (11, 11),
    10
)

robot_2d_reachavoid = IntervalSySCoReComparisonProblem(
    "robot_2d_reachavoid",
    (state_split, input_split) -> robot_2d_direct(;spec=:reachavoid, sparse=true, state_split=state_split, input_split=input_split),
    (state_split, input_split) -> robot_2d_decoupled(;spec=:reachavoid, sparse=true, state_split=state_split, input_split=input_split),
    (mdp, reach, avoid, time_horizon) -> Problem(mdp, Specification(FiniteTimeReachAvoid(reach, avoid, time_horizon), Pessimistic, Maximize)),
    identity,
    (40, 40),
    (21, 21),
    10
)

syscore_running_example = IntervalSySCoReComparisonProblem(
    "syscore_running_example",
    (state_split, input_split) -> running_example_direct(;range_vs_grid=:range, sparse=true, state_split=state_split, input_split=input_split),
    (state_split, input_split) -> running_example_decoupled(;range_vs_grid=:range, state_split=state_split, input_split=input_split),
    (mdp, reach, avoid, time_horizon) -> Problem(mdp, Specification(FiniteTimeReachAvoid(reach, avoid, time_horizon), Pessimistic, Maximize)),
    identity,
    (40, 40),
    (3, 3),
    10
)

van_der_pol = IntervalSySCoReComparisonProblem(
    "van_der_pol",
    (state_split, input_split) -> van_der_pol_direct(;state_split=state_split, input_split=input_split),
    (state_split, input_split) -> van_der_pol_decoupled(;state_split=state_split, input_split=input_split),
    (mdp, reach, avoid, time_horizon) -> Problem(mdp, Specification(FiniteTimeReachAvoid(reach, avoid, time_horizon), Pessimistic, Maximize)),
    identity,
    (50, 50),
    (11,),
    10
)

bas4d = IntervalSySCoReComparisonProblem(
    "bas4d",
    (state_split, input_split) -> building_automation_system_4d_direct(;sparse=true, state_split=state_split, input_split=input_split),
    (state_split, input_split) -> building_automation_system_4d_decoupled(;state_split=state_split, input_split=input_split),
    (mdp, reach, avoid, time_horizon) -> Problem(mdp, Specification(FiniteTimeReachability(avoid, time_horizon), Optimistic, Minimize)),
    (V) -> 1.0 .- V,
    (5, 5, 7, 7),
    (4,),
    10
)

bas7d = IntervalSySCoReComparisonProblem(
    "bas7d",
    (state_split, input_split) -> building_automation_system_7d_direct(;sparse=true, state_split=state_split),
    (state_split, input_split) -> building_automation_system_7d_decoupled(;state_split=state_split),
    (mdp, reach, avoid, time_horizon) -> Problem(mdp, Specification(FiniteTimeReachability(avoid, time_horizon), Optimistic, Minimize)),
    (V) -> 1.0 .- V,
    (21, 21, 3, 3, 3, 3, 3),
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

function warmup_abstraction(problem::IntervalSySCoReComparisonProblem, constructor)
    constructor(problem.state_split, problem.input_split)

    return nothing
end

function measure_abstraction_time(problem::IntervalSySCoReComparisonProblem, constructor)
    BenchmarkTools.gcscrub()
    start_time = time_ns()
    mdp, reach, avoid = constructor(problem.state_split, problem.input_split)
    end_time = time_ns()
    abstraction_time = (end_time - start_time) / 1e9

    return abstraction_time, mdp, reach, avoid
end

function warmup_certification(prob)
    value_iteration(prob)

    return nothing
end

function measure_certification_time(prob)
    BenchmarkTools.gcscrub()
    start_time = time_ns()
    V, k, res = value_iteration(prob)
    end_time = time_ns()
    certification_time = (end_time - start_time) / 1e9

    return certification_time, V
end

function benchmark_intervalsyscore(name::String, direct=true)
    problem = problems[findfirst(problem -> problem.name == name, problems)]

    constructor = if direct
        problem.direct_constructor
    else
        problem.decoupled_constructor
    end
  
    # Warmup
    warmup_abstraction(problem, constructor)

    abstraction_time, mdp, reach, avoid = measure_abstraction_time(problem, constructor)
    println(("Abstraction time", abstraction_time))

    prob_mem = Base.summarysize(mdp) / 1000^2
    println(("Transition probability memory", prob_mem))

    prob = problem.problem_constructor(mdp, reach, avoid, problem.time_horizon)

    # Warmup
    warmup_certification(prob)

    certification_time, V = measure_certification_time(prob)
    println(("Certification time", certification_time))

    V = problem.post_process_value_function(V)

    for v in V
        println(v)
    end
end

end

IsolatedCompareIMDPApproaches.benchmark_intervalsyscore(ARGS[1], parse(Bool, ARGS[2]))