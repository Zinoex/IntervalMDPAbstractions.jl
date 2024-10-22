module IsolatedCompareIMDPApproaches

using BenchmarkTools
using IntervalMDP
using IntervalSySCoRe

include("systems/systems.jl")

struct IntervalSySCoReComparisonProblem{N, M}
    name::String

    direct_constructor::Function
    decoupled_constructor::Function

    state_split::NTuple{N, Int}
    input_split::NTuple{M, Int}
    time_horizon::Int
end

car_parking = IntervalSySCoReComparisonProblem(
    "car_parking",
    (state_split, input_split, time_horizon) -> car_parking_direct(time_horizon; range_vs_grid=:range, sparse=true, state_split=state_split, input_split=input_split),
    (state_split, input_split, time_horizon) -> car_parking_decoupled(time_horizon; range_vs_grid=:range, state_split=state_split, input_split=input_split),
    (40, 40),
    (3, 3),
    10
)

robot_2d_reachability = IntervalSySCoReComparisonProblem(
    "robot_2d_reachability",
    (state_split, input_split, time_horizon) -> robot_2d_direct(time_horizon; spec=:reachability, sparse=true, state_split=state_split, input_split=input_split),
    (state_split, input_split, time_horizon) -> robot_2d_decoupled(time_horizon; spec=:reachability, state_split=state_split, input_split=input_split),
    (20, 20),
    (11, 11),
    10
)

robot_2d_reachavoid = IntervalSySCoReComparisonProblem(
    "robot_2d_reachavoid",
    (state_split, input_split, time_horizon) -> robot_2d_direct(time_horizon; spec=:reachavoid, sparse=true, state_split=state_split, input_split=input_split),
    (state_split, input_split, time_horizon) -> robot_2d_decoupled(time_horizon; spec=:reachavoid, sparse=true, state_split=state_split, input_split=input_split),
    (40, 40),
    (21, 21),
    10
)

bas4d = IntervalSySCoReComparisonProblem(
    "bas4d",
    (state_split, input_split, time_horizon) -> building_automation_system_4d_direct(time_horizon; sparse=true, state_split=state_split, input_split=input_split),
    (state_split, input_split, time_horizon) -> building_automation_system_4d_decoupled(time_horizon; state_split=state_split, input_split=input_split),
    (5, 5, 7, 7),
    (4,),
    10
)

van_der_pol = IntervalSySCoReComparisonProblem(
    "van_der_pol",
    (state_split, input_split, time_horizon) -> van_der_pol_direct(time_horizon; state_split=state_split, input_split=input_split),
    (state_split, input_split, time_horizon) -> van_der_pol_decoupled(time_horizon; state_split=state_split, input_split=input_split),
    (50, 50),
    (11,),
    10
)

nndm_cartpole = IntervalSySCoReComparisonProblem(
    "nndm_cartpole",
    (state_split, input_split, time_horizon) -> action_cartpole_direct(time_horizon; sparse=true),
    (state_split, input_split, time_horizon) -> action_cartpole_decoupled(time_horizon; sparse=false),
    (20, 20, 24, 20),
    (1,),
    10
)

linear6d = IntervalSySCoReComparisonProblem(
    "linear6d",
    (state_split, input_split, time_horizon) -> almost_identity_direct(6, time_horizon; sparse=true, state_split_per_dim=8),
    (state_split, input_split, time_horizon) -> almost_identity_decoupled(6, time_horizon; sparse=true, state_split_per_dim=8),
    (8, 8, 8, 8, 8, 8),
    (1,),
    10
)

linear7d = IntervalSySCoReComparisonProblem(
    "linear7d",
    (state_split, input_split, time_horizon) -> almost_identity_direct(7, time_horizon; sparse=true, state_split_per_dim=8),
    (state_split, input_split, time_horizon) -> almost_identity_decoupled(7, time_horizon; sparse=true, state_split_per_dim=8),
    (8, 8, 8, 8, 8, 8, 8),
    (1,),
    10
)

linear_stochastically_switched = IntervalSySCoReComparisonProblem(
    "linear_stochastically_switched",
    (state_split, input_split, time_horizon) -> linear_stochastically_switched_direct(time_horizon; state_split=state_split),
    (state_split, input_split, time_horizon) -> linear_stochastically_switched_mixture(time_horizon; state_split=state_split),
    (40, 40),
    (1,),
    10
)

problems = [
    robot_2d_reachability,
    robot_2d_reachavoid,
    van_der_pol,
    bas4d,
    nndm_cartpole,
    linear6d,
    linear7d,
    linear_stochastically_switched
]

function warmup_abstraction(problem::IntervalSySCoReComparisonProblem, constructor)
    constructor(problem.state_split, problem.input_split, problem.time_horizon)

    return nothing
end

function measure_abstraction_time(problem::IntervalSySCoReComparisonProblem, constructor)
    BenchmarkTools.gcscrub()
    start_time = time_ns()
    mdp, spec, upper_bound_spec = constructor(problem.state_split, problem.input_split, problem.time_horizon)
    end_time = time_ns()
    abstraction_time = (end_time - start_time) / 1e9

    return abstraction_time, mdp, spec, upper_bound_spec
end

function warmup_certification(prob, upper_bound_spec)
    strategy, _ = control_synthesis(prob)

    upper_bound_prob = Problem(IntervalMDP.system(prob), upper_bound_spec, strategy)
    value_iteration(upper_bound_prob)

    return upper_bound_prob
end

function measure_certification_time(prob, upper_bound_prob)
    BenchmarkTools.gcscrub()
    start_time = time_ns()
    
    _, V_lower, _ = control_synthesis(prob)
    V_upper, _ = value_iteration(upper_bound_prob)

    end_time = time_ns()
    certification_time = (end_time - start_time) / 1e9

    return certification_time, V_lower, V_upper
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

    # Measure abstraction time
    abstraction_time, mdp, spec, upper_bound_spec = measure_abstraction_time(problem, constructor)
    println(("Abstraction time", abstraction_time))

    # Measure memory usage
    prob_mem = Base.summarysize(mdp) / 1000^2
    println(("Transition probability memory", prob_mem))

    prob = Problem(mdp, spec)

    # Warmup
    upper_bound_prob = warmup_certification(prob, upper_bound_spec)

    # Measure certification time
    certification_time, V_lower, V_upper = measure_certification_time(prob, upper_bound_prob)
    println(("Certification time", certification_time))

    println("reach")
    reach_set = if system_property(spec) isa AbstractReachability
        IntervalMDP.reach(system_property(spec))
    else
        []
    end
    for r in reach_set
        println(r)
    end

    println("avoid")
    for a in IntervalMDP.avoid(system_property(spec))
        println(a)
    end

    println("V_lower")
    for v in V_lower
        println(v)
    end

    println("V_upper")
    for v in V_upper
        println(v)
    end
end

end

IsolatedCompareIMDPApproaches.benchmark_intervalsyscore(ARGS[1], parse(Bool, ARGS[2]))