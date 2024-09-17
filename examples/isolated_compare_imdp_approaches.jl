module IsolatedCompareIMDPApproaches

include("comparison_problems.jl")

function benchmark_intervalsyscore(name::String, direct=true)
    problem = problems[findfirst(problem -> problem.name == name, problems)]

    constructor = if direct
        problem.direct_constructor
    else
        problem.decoupled_constructor
    end
  
    # Warmup
    BenchmarkTools.gcscrub()
    constructor(problem.state_split, problem.input_split)

    start_time = time_ns()
    mdp, reach, avoid = constructor(problem.state_split, problem.input_split)
    end_time = time_ns()
    abstraction_time = (end_time - start_time) / 1e9
    println(("Abstraction time", abstraction_time))

    prob_mem = Base.summarysize(mdp) / 1000^2
    println(("Transition probability memory", prob_mem))

    prob = problem.problem_constructor(mdp, reach, avoid, problem.time_horizon)

    # Warmup
    BenchmarkTools.gcscrub()
    value_iteration(prob)

    start_time = time_ns()
    V, k, res = value_iteration(prob)
    end_time = time_ns()
    certification_time = (end_time - start_time) / 1e9
    println(("Certification time", certification_time))

    V = problem.post_process_value_function(V)

    for v in V
        println(v)
    end
end

end

benchmark_intervalsyscore(ARGS[1], parse(Bool, ARGS[2]))