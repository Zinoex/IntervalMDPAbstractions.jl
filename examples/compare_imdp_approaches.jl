module CompareIMDPApproaches

using BenchmarkTools, ProgressMeter
using DataFrames, CSV, JSON, Statistics

include("systems/IMPaCT.jl")

struct ComparisonProblem
    name::String
    impact_evaluator::Function

    include_impact::Bool

    state_split
    input_split
end

robot_2d_reachability = ComparisonProblem(
    "robot_2d_reachability",
    () -> run_impact("ex_2Drobot-R-U"),
    true,
    (20, 20),
    (11, 11),
)

robot_2d_reachavoid = ComparisonProblem(
    "robot_2d_reachavoid",
    () -> run_impact("ex_2Drobot-RA-U"),
    true,
    (40, 40),
    (21, 21),
)

syscore_running_example = ComparisonProblem(
    "syscore_running_example",
    () -> run_impact("ex_syscore_running_example-RA"),
    true,
    (40, 40),
    (3, 3),
)

van_der_pol = ComparisonProblem(
    "van_der_pol",
    () -> run_impact("ex_van_der_pol-R"),
    true,
    (50, 50),
    (11,),
)

bas4d = ComparisonProblem(
    "bas4d",
    () -> run_impact("ex_4DBAS-S"),
    true,
    (5, 5, 7, 7),
    (4,),
)

bas7d = ComparisonProblem(
    "bas7d",
    () -> run_impact("ex_7DBAS-S"),
    true,
    (21, 21, 3, 3, 3, 3, 3),
    (1,),
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
        BenchmarkTools.gcscrub()
        output = read(`julia -tauto --project=$(@__DIR__) isolated_compare_imdp_approaches.jl $(problem.name) true`, String)

        m1 = match(r"\(\"Abstraction time\",\s(\d+\.\d+)\)", output)
        m2 = match(r"\(\"Certification time\",\s(\d+\.\d+)\)", output)
        m3 = match(r"\(\"Transition probability memory\",\s(\d+\.\d+)\)", output)
        if m1 === nothing || m2 === nothing || m3 === nothing
            @warn "Direct failed due to OOM"
    
            return Dict(
                "oom" => true,
                "abstraction_time" => NaN,
                "certification_time" => NaN,
                "prob_mem" => NaN,
                "value_function" => NaN
            )
        end
        abstraction_time = parse(Float64, m1.captures[1])
        certification_time = parse(Float64, m2.captures[1])
        prob_mem = parse(Float64, m3.captures[1])

        lines = split(chomp(output), '\n')
        lines = lines[4:end]

        V = map(line -> parse(Float64, line), lines)

        # Remove the first element of the value function, which is the absorbing avoid state
        V = reshape(V[2:end], problem.state_split...)
        V = to_impact_format(V)
    
        return Dict(
            "oom" => false,
            "abstraction_time" => abstraction_time,
            "certification_time" => certification_time,
            "prob_mem" => prob_mem,
            "value_function" => V
        )
    catch e
        if isa(e, ProcessFailedException)
            @warn "Direct failed due to OOM"
    
            return Dict(
                "oom" => true,
                "abstraction_time" => NaN,
                "certification_time" => NaN,
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
        BenchmarkTools.gcscrub()
        output = read(`julia -tauto --project=$(@__DIR__) isolated_compare_imdp_approaches.jl $(problem.name) false`, String)

        m1 = match(r"\(\"Abstraction time\",\s(\d+\.\d+)\)", output)
        m2 = match(r"\(\"Certification time\",\s(\d+\.\d+)\)", output)
        m3 = match(r"\(\"Transition probability memory\",\s(\d+\.\d+)\)", output)
        if m1 === nothing || m2 === nothing || m3 === nothing
            @warn "Direct failed due to OOM"
    
            return Dict(
                "oom" => true,
                "abstraction_time" => NaN,
                "certification_time" => NaN,
                "prob_mem" => NaN,
                "value_function" => NaN
            )
        end
        abstraction_time = parse(Float64, m1.captures[1])
        certification_time = parse(Float64, m2.captures[1])
        prob_mem = parse(Float64, m3.captures[1])

        lines = split(chomp(output), '\n')
        lines = lines[4:end]

        V = map(line -> parse(Float64, line), lines)

        # Remove the first element of the value function, which is the absorbing avoid state
        V = reshape(V, (problem.state_split .+ 1)...)
        V = V[(2:size(V, i) for i in 1:ndims(V))...]
        V = to_impact_format(V)

        return Dict(
            "oom" => false,
            "abstraction_time" => abstraction_time,
            "certification_time" => certification_time,
            "prob_mem" => prob_mem,
            "value_function" => V
        )
    catch e
        if isa(e, ProcessFailedException)
            @warn "Decoupled failed due to OOM"
    
            return Dict(
                "oom" => true,
                "abstraction_time" => NaN,
                "certification_time" => NaN,
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

        direct = benchmark_direct(problem)
        decoupled = benchmark_decoupled(problem)

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

function read_results(name)
    res = JSON.parsefile("results/compare_imdp_approaches/$name.json")
    return res
end

function read_results()
    res = Dict()
    for problem in problems
        res[problem.name] = read_results(problem.name)
    end

    return res
end

function to_dataframe(res)
    rows = []
    for (name, data) in res
        n_decoupled = length(data["decoupled"]["value_function"])
        n_direct = length(data["direct"]["value_function"])
        n_impact = length(data["impact"]["value_function"])

        if n_decoupled != n_direct
            @warn "Skipping $name due to different value function sizes (decoupled vs direct)"
        elseif n_decoupled != n_impact
            @warn "Skipping $name due to different value function sizes (decoupled vs impact)"
            continue
        end

        row = Dict(
            "name" => name,
            "state_split" => data["state_split"],
            "input_split" => data["input_split"],
            "direct_abstraction_time" => data["direct"]["abstraction_time"],
            "direct_certification_time" => data["direct"]["certification_time"],
            "direct_prob_mem" => data["direct"]["prob_mem"],
            "direct_min_prob" => minimum(data["direct"]["value_function"]),
            "direct_max_prob" => maximum(data["direct"]["value_function"]),
            "direct_min_prob_diff" => minimum(data["decoupled"]["value_function"] - data["direct"]["value_function"]),
            "direct_max_prob_diff" => maximum(data["decoupled"]["value_function"] - data["direct"]["value_function"]),
            "impact_abstraction_time" => data["impact"]["abstraction_time"],
            "impact_certification_time" => data["impact"]["certification_time"],
            "impact_prob_mem" => data["impact"]["prob_mem"],
            "impact_min_prob" => minimum(data["impact"]["value_function"]),
            "impact_max_prob" => maximum(data["impact"]["value_function"]),
            "impact_min_prob_diff" => minimum(data["decoupled"]["value_function"] - data["impact"]["value_function"]),
            "impact_max_prob_diff" => maximum(data["decoupled"]["value_function"] - data["impact"]["value_function"]),
            "decoupled_abstraction_time" => data["decoupled"]["abstraction_time"],
            "decoupled_certification_time" => data["decoupled"]["certification_time"],
            "decoupled_prob_mem" => data["decoupled"]["prob_mem"],
            "decoupled_min_prob" => minimum(data["decoupled"]["value_function"]),
            "decoupled_max_prob" => maximum(data["decoupled"]["value_function"]),
        )
        push!(rows, row)
    end

    df = DataFrame(rows)
    return df
end

# TODO: plot results

# function compare()
#     res = benchmark()
#     df = to_dataframe(res)
#     save_results(df)
# end

end

CompareIMDPApproaches.benchmark()
