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

car_parking = ComparisonProblem(
    "car_parking",
    () -> run_impact("ex_car_parking-RA"),
    true,
    (40, 40),
    (3, 3),
)

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

bas4d = ComparisonProblem(
    "bas4d",
    () -> run_impact("ex_4DBAS-S"),
    true,
    (5, 5, 7, 7),
    (4,),
)

van_der_pol = ComparisonProblem(
    "van_der_pol",
    () -> run_impact("ex_van_der_pol-R"),
    true,
    (50, 50),
    (11,),
)

nndm_cartpole = ComparisonProblem(
    "nndm_cartpole",
    () -> nothing,
    false,
    (20, 20, 24, 20),
    (1,),
)

linear6d = ComparisonProblem(
    "linear6d",
    () -> nothing,
    false,
    (8, 8, 8, 8, 8, 8),
    (1,),
)

linear7d = ComparisonProblem(
    "linear7d",
    () -> nothing,
    false,
    (8, 8, 8, 8, 8, 8, 8),
    (1,),
)

linear_stochastically_switched = ComparisonProblem(
    "linear_stochastically_switched",
    () -> run_impact("ex_linear_stochastically_switched-R"),
    false, # Running IMPaCT on this problem is will result in nlopt failure for every transition.
    (40, 40),
    (1,),
)

problems = [
    car_parking,
    robot_2d_reachability,
    robot_2d_reachavoid,
    bas4d,
    van_der_pol,
    nndm_cartpole,
    linear6d,
    linear7d,
    linear_stochastically_switched
]

function benchmark_impact(problem::ComparisonProblem)
    @info "Benchmarking IMPaCT"
    res = problem.impact_evaluator()

    return res
end

function to_impact_format(V, reach, avoid)
    V[reach] .= -1.0
    V[avoid] .= -1.0

    # Transpose to match row-major order of IMPaCT
    V = permutedims(V, reverse(1:ndims(V)))
    V = vec(V)

    # Remove reach and avoid regions to match IMPaCT
    V = V[V .!= -1.0]

    return V
end

# Has to be double, since, for a realiable time measurement, julia has to run twice (it may compile the first time).
TIMEOUT_DURATION = "48h"
TIMEOUT_CMD = `timeout --signal SIGKILL --verbose $TIMEOUT_DURATION`
JULIA_CMD = `julia -tauto --project=$(@__DIR__) executor.jl`

function benchmark_direct(problem::ComparisonProblem)
    @info "Benchmarking direct"
    try
        BenchmarkTools.gcscrub()
        output = read(`$TIMEOUT_CMD $JULIA_CMD $(problem.name) true`, String)

        if occursin("timeout", output)
            @warn "Decoupled timeout"
    
            return Dict(
                "oom" => false,
                "timeout" => true,
                "abstraction_time" => NaN,
                "certification_time" => NaN,
                "prob_mem" => NaN,
                "value_function_lower" => NaN,
                "value_function_upper" => NaN
            )
        end

        m1 = match(r"\(\"Abstraction time\",\s(\d+\.\d+)\)", output)
        m2 = match(r"\(\"Certification time\",\s(\d+\.\d+)\)", output)
        m3 = match(r"\(\"Transition probability memory\",\s(\d+\.\d+)\)", output)
        if m1 === nothing || m2 === nothing || m3 === nothing
            @warn "Direct failed due to OOM"
    
            return Dict(
                "oom" => true,
                "timeout" => false,
                "abstraction_time" => NaN,
                "certification_time" => NaN,
                "prob_mem" => NaN,
                "value_function_lower" => NaN,
                "value_function_upper" => NaN
            )
        end
        abstraction_time = parse(Float64, m1.captures[1])
        certification_time = parse(Float64, m2.captures[1])
        prob_mem = parse(Float64, m3.captures[1])

        # Split output
        prefix, rest = split(output, "reach\n")
        reach, rest = split(rest, "avoid\n")
        avoid, rest = split(rest, "V_lower\n")
        V_lower, rest = split(rest, "V_upper\n")

        cartesian_indices = CartesianIndices(problem.state_split)

        # Read reach states
        if reach == ""
            reach = []
        else
            reach_lines = split(chomp(reach), '\n')
            reach = map(reach_lines) do line
                line_match = match(r"CartesianIndex\(([0-9]+),\)", line)
                return parse(Int32, line_match.captures[1])
            end
            reach = reach .- 1 # Subtract 1 to match 1-based indexing without the avoid state
            reach = map(x -> cartesian_indices[x], reach) # Convert from a linear index to a CartesianIndex
        end

        # Read avoid states
        avoid_lines = split(chomp(avoid), '\n') 
        avoid = map(avoid_lines) do line
            line_match = match(r"CartesianIndex\(([0-9]+),\)", line)
            return parse(Int32, line_match.captures[1])
        end
        avoid = filter(x -> x != 1, avoid) # Remove the absorbing avoid state (we remove this manually later)
        avoid = avoid .- 1 # Subtract 1 to match 1-based indexing without the avoid state
        avoid = map(x -> cartesian_indices[x], avoid) # Convert from a linear index to a CartesianIndex

        ## V_lower
        V_lower_lines = split(chomp(V_lower), '\n')
        V_lower = map(line -> parse(Float64, line), V_lower_lines)

        # Remove the first element of the value function, which is the absorbing avoid state
        V_lower = reshape(V_lower[2:end], problem.state_split...)
        V_lower = to_impact_format(V_lower, reach, avoid)

        ## V_upper
        V_upper_lines = split(chomp(rest), '\n')
        V_upper = map(line -> parse(Float64, line), V_upper_lines)

        # Remove the first element of the value function, which is the absorbing avoid state
        V_upper = reshape(V_upper[2:end], problem.state_split...)
        V_upper = to_impact_format(V_upper, reach, avoid)

        return Dict(
            "oom" => false,
            "timeout" => false,
            "abstraction_time" => abstraction_time,
            "certification_time" => certification_time,
            "prob_mem" => prob_mem,
            "value_function_lower" => V_lower,
            "value_function_upper" => V_upper
        )
    catch e
        if isa(e, ProcessFailedException)
            @warn "Direct failed due to OOM"
    
            return Dict(
                "oom" => true,
                "timeout" => false,
                "abstraction_time" => NaN,
                "certification_time" => NaN,
                "prob_mem" => NaN,
                "value_function_lower" => NaN,
                "value_function_upper" => NaN
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
        output = read(`$TIMEOUT_CMD $JULIA_CMD $(problem.name) false`, String)

        if occursin("timeout", output)
            @warn "Decoupled timeout"
    
            return Dict(
                "oom" => false,
                "timeout" => true,
                "abstraction_time" => NaN,
                "certification_time" => NaN,
                "prob_mem" => NaN,
                "value_function_lower" => NaN,
                "value_function_upper" => NaN
            )
        end

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
                "value_function_lower" => NaN,
                "value_function_upper" => NaN
            )
        end
        abstraction_time = parse(Float64, m1.captures[1])
        certification_time = parse(Float64, m2.captures[1])
        prob_mem = parse(Float64, m3.captures[1])

        # Split output
        prefix, rest = split(output, "reach\n")
        reach, rest = split(rest, "avoid\n")
        avoid, rest = split(rest, "V_lower\n")
        V_lower, rest = split(rest, "V_upper\n")
        
        # Read reach states
        if reach == ""
            reach = []
        else
            reach_lines = split(chomp(reach), '\n')
            reach = map(reach_lines) do line # Parse each line as a tuple of indices
                line = replace(line, "CartesianIndex" => "")
                indices = split(line[2:end - 1], ",")
                indices = map(index -> parse(Int32, index), indices)
                return Tuple(indices)
            end
            reach = map(x -> CartesianIndex(x .- 1), reach) # Subtract 1 to match 1-based indexing without the avoid states
        end

        # Read avoid states
        avoid_lines = split(chomp(avoid), '\n')
        avoid = map(avoid_lines) do line # Parse each line as a tuple of indices
            line = replace(line, "CartesianIndex" => "")
            indices = split(line[2:end - 1], ",")
            indices = map(index -> parse(Int32, index), indices)
            return Tuple(indices)
        end
        avoid = filter(x -> !any(isone, x), avoid) # Remove the absorbing avoid states (we remove these manually later)
        avoid = map(x -> CartesianIndex(x .- 1), avoid) # Subtract 1 to match 1-based indexing without the avoid states

        ## V_lower
        V_lower_lines = split(chomp(V_lower), '\n')
        V_lower = map(line -> parse(Float64, line), V_lower_lines)

        # Remove the first element of the value function, which is the absorbing avoid state
        V_lower = reshape(V_lower, (problem.state_split .+ 1)...)
        V_lower = V_lower[(2:size(V_lower, i) for i in 1:ndims(V_lower))...]
        V_lower = to_impact_format(V_lower, reach, avoid)

        ## V_upper
        V_upper_lines = split(chomp(rest), '\n')
        V_upper = map(line -> parse(Float64, line), V_upper_lines)

        # Remove the first element of the value function, which is the absorbing avoid state
        V_upper = reshape(V_upper, (problem.state_split .+ 1)...)
        V_upper = V_upper[(2:size(V_upper, i) for i in 1:ndims(V_upper))...]
        V_upper = to_impact_format(V_upper, reach, avoid)

        return Dict(
            "oom" => false,
            "timeout" => false,
            "abstraction_time" => abstraction_time,
            "certification_time" => certification_time,
            "prob_mem" => prob_mem,
            "value_function_lower" => V_lower,
            "value_function_upper" => V_upper
        )
    catch e
        if isa(e, ProcessFailedException)
            @warn "Decoupled failed due to OOM"
    
            return Dict(
                "oom" => true,
                "timeout" => false,
                "abstraction_time" => NaN,
                "certification_time" => NaN,
                "prob_mem" => NaN,
                "value_function_lower" => NaN,
                "value_function_upper" => NaN
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
            "impact" => impact,
            "include_impact" => problem.include_impact
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
        n_decoupled = length(data["decoupled"]["value_function_lower"])

        row = Dict(
            "name" => name,
            "state_split" => data["state_split"],
            "input_split" => data["input_split"],
            "decoupled_abstraction_time" => data["decoupled"]["abstraction_time"],
            "decoupled_certification_time" => data["decoupled"]["certification_time"],
            "decoupled_prob_mem" => data["decoupled"]["prob_mem"],
            "decoupled_min_prob" => minimum(data["decoupled"]["value_function_lower"]),
            "decoupled_max_prob" => maximum(data["decoupled"]["value_function_lower"]),
            "decoupled_mean_error" => mean(data["decoupled"]["value_function_upper"] - data["decoupled"]["value_function_lower"]),
        )

        if !data["direct"]["oom"] && !data["direct"]["timeout"]
            if n_decoupled != length(data["direct"]["value_function_lower"])
                @warn "Direct value function length mismatch" name n_decoupled n_direct=length(data["direct"]["value_function_lower"])

                row["direct_abstraction_time"] = NaN
                row["direct_certification_time"] = NaN
                row["direct_prob_mem"] = NaN
                row["direct_min_prob"] = NaN
                row["direct_max_prob"] = NaN
                row["direct_mean_error"] = NaN
                row["direct_min_prob_diff"] = NaN
                row["direct_max_prob_diff"] = NaN
                row["direct_avg_prob_diff"] = NaN
            else
                row["direct_abstraction_time"] = data["direct"]["abstraction_time"]
                row["direct_certification_time"] = data["direct"]["certification_time"]
                row["direct_prob_mem"] = data["direct"]["prob_mem"]
                row["direct_min_prob"] = minimum(data["direct"]["value_function_lower"])
                row["direct_max_prob"] = maximum(data["direct"]["value_function_lower"])
                row["direct_mean_error"] = mean(data["direct"]["value_function_upper"] - data["direct"]["value_function_lower"])
                row["direct_min_prob_diff"] = minimum(data["decoupled"]["value_function_lower"] - data["direct"]["value_function_lower"])
                row["direct_max_prob_diff"] = maximum(data["decoupled"]["value_function_lower"] - data["direct"]["value_function_lower"])
                row["direct_avg_prob_diff"] = mean(data["decoupled"]["value_function_lower"] - data["direct"]["value_function_lower"])
            end
        else
            row["direct_abstraction_time"] = NaN
            row["direct_certification_time"] = NaN
            row["direct_prob_mem"] = NaN
            row["direct_min_prob"] = NaN
            row["direct_max_prob"] = NaN
            row["direct_mean_error"] = NaN
            row["direct_min_prob_diff"] = NaN
            row["direct_max_prob_diff"] = NaN
            row["direct_avg_prob_diff"] = NaN
        end

        if data["include_impact"] && !data["impact"]["oom"] && !data["impact"]["timeout"]
            if n_decoupled != length(data["impact"]["value_function_lower"]) 
                @warn "Impact value function length mismatch" name n_decoupled n_impact=length(data["impact"]["value_function_lower"])

                row["impact_abstraction_time"] = NaN
                row["impact_certification_time"] = NaN
                row["impact_prob_mem"] = NaN
                row["impact_min_prob"] = NaN
                row["impact_max_prob"] = NaN
                row["impact_mean_error"] = NaN
                row["impact_min_prob_diff"] = NaN
                row["impact_max_prob_diff"] = NaN
                row["impact_avg_prob_diff"] = NaN
            else
                row["impact_abstraction_time"] = data["impact"]["abstraction_time"]
                row["impact_certification_time"] = data["impact"]["certification_time"]
                row["impact_prob_mem"] = data["impact"]["prob_mem"]
                if any(isnothing, data["impact"]["value_function_lower"])
                    @warn "Impact value function contains NaN (null in JSON)" name

                    row["impact_min_prob"] = NaN
                    row["impact_max_prob"] = NaN
                    row["impact_mean_error"] = NaN
                    row["impact_min_prob_diff"] = NaN
                    row["impact_max_prob_diff"] = NaN
                    row["impact_avg_prob_diff"] = NaN
                else
                    row["impact_min_prob"] = minimum(data["impact"]["value_function_lower"])
                    row["impact_max_prob"] = maximum(data["impact"]["value_function_lower"])
                    row["impact_mean_error"] = mean(data["impact"]["value_function_upper"] - data["impact"]["value_function_lower"])
                    row["impact_min_prob_diff"] = minimum(data["decoupled"]["value_function_lower"] - data["impact"]["value_function_lower"])
                    row["impact_max_prob_diff"] = maximum(data["decoupled"]["value_function_lower"] - data["impact"]["value_function_lower"])
                    row["impact_avg_prob_diff"] = mean(data["decoupled"]["value_function_lower"] - data["impact"]["value_function_lower"])
                end
            end
        else
            row["impact_abstraction_time"] = NaN
            row["impact_certification_time"] = NaN
            row["impact_prob_mem"] = NaN
            row["impact_min_prob"] = NaN
            row["impact_max_prob"] = NaN
            row["impact_mean_error"] = NaN
            row["impact_min_prob_diff"] = NaN
            row["impact_max_prob_diff"] = NaN
            row["impact_avg_prob_diff"] = NaN
        end

        push!(rows, row)
    end

    df = DataFrame(rows)
    select!(df, 
        :name,
        :state_split,
        :input_split,
        :decoupled_abstraction_time,
        :decoupled_certification_time,
        :decoupled_prob_mem,
        :decoupled_min_prob,
        :decoupled_max_prob,
        :decoupled_mean_error,
        :direct_abstraction_time,
        :direct_certification_time,
        :direct_prob_mem,
        :direct_min_prob,
        :direct_max_prob,
        :direct_mean_error,
        :direct_min_prob_diff,
        :direct_max_prob_diff,
        :direct_avg_prob_diff,
        :impact_abstraction_time,
        :impact_certification_time,
        :impact_prob_mem,
        :impact_min_prob,
        :impact_max_prob,
        :impact_mean_error,
        :impact_min_prob_diff,
        :impact_max_prob_diff,
        :impact_avg_prob_diff
    )
    return df
end

function save_dataframe(df)
    mkpath("results/compare_imdp_approaches", mode=0o754)
    CSV.write("results/compare_imdp_approaches/summarized_results.csv", df)
end

# TODO: plot results

function compare()
    res = read_results()
    df = to_dataframe(res)
    save_dataframe(df)
end

end

if ARGS[1] == "run_benchmark"
    CompareIMDPApproaches.benchmark()
elseif ARGS[1] == "compute_results"
    CompareIMDPApproaches.compare()
else
    @error "Invalid argument $(ARGS[1])"
end