using BenchmarkTools
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
    input_split::NTuple{M, Int}
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
    (state_split, input_split) -> robot_2d_decoupled(;spec=:reachavoid, sparse=true, state_split=state_split, input_split=input_split),
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