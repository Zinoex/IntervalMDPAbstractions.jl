module SySCoReComparison

using BenchmarkTools
using IntervalMDP
using IntervalSySCoRe

using MAT, MAT.MAT_v4, MAT.MAT_v5, MAT.MAT_HDF5

include("systems/systems.jl")

struct SySCoReComparisonProblem
    name::String

    constructor::Function

    state_split
    input_split
    time_horizon
end


syscore_running_example = SySCoReComparisonProblem(
    "running_example",
    (state_split, input_split, time_horizon) -> running_example_decoupled(time_horizon; range_vs_grid=:range, state_split=state_split, input_split=input_split),
    n -> (n, n),
    (3, 3),
    10
)


bas_4d = SySCoReComparisonProblem(
    "4d_bas",
    (state_split, input_split, time_horizon) -> building_automation_system_4d_decoupled(time_horizon; state_split=state_split, input_split=input_split),
    n -> (n, n, n, n),
    4,
    10
)

function load_syscore_mat(system, num_regions_per_axis)
    # Load the data
    mat = matread("results/syscore/$system/satprob_$num_regions_per_axis.mat")
    return mat["satProb_initial"]
end

function warmup_abstraction(problem::SySCoReComparisonProblem, num_regions_per_axis)
    problem.constructor(problem.state_split(num_regions_per_axis), problem.input_split)

    return nothing
end

function measure_abstraction_time(problem::SySCoReComparisonProblem, num_regions_per_axis)
    BenchmarkTools.gcscrub()
    start_time = time_ns()
    mdp, spec = problem.constructor(problem.state_split(num_regions_per_axis), problem.input_split, problem.time_horizon)
    end_time = time_ns()
    abstraction_time = (end_time - start_time) / 1e9

    return abstraction_time, mdp, spec
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

function benchmark_intervalsyscore(problem::SySCoReComparisonProblem, num_regions_per_axis)    
    # Warmup
    warmup_abstraction(problem, num_regions_per_axis)

    abstraction_time, mdp, spec = measure_abstraction_time(problem, num_regions_per_axis)
    println(("Abstraction time", abstraction_time))

    prob_mem = Base.summarysize(mdp) / 1000^2
    println(("Transition probability memory", prob_mem))

    prob = Problem(mdp, spec)

    # Warmup
    warmup_certification(prob)

    certification_time, V = measure_certification_time(prob)
    println(("Certification time", certification_time))

    println(("Total time", abstraction_time + certification_time))

    # println("reach")
    # for r in reach
    #     println(r)
    # end

    # println("avoid")
    # for a in avoid
    #     println(a)
    # end

    V = problem.post_process_value_function(V)

    if spec isa AbstractReachability
        reach = map(x -> CartesianIndex(x .- 1), reach) # Subtract 1 to match 1-based indexing without the avoid states
    else
        reach = []
    end

    avoid = filter(x -> !any(isone, x), avoid) # Remove the absorbing avoid states (we remove these manually later)
    avoid = map(x -> CartesianIndex(x .- 1), avoid) # Subtract 1 to match 1-based indexing without the avoid states

    return V, reach, avoid
end

function benchmark_system(problem, num_regions_per_axis)
    println(("num_regions_per_axis", num_regions_per_axis))
    V, reach, avoid = benchmark_intervalsyscore(problem, num_regions_per_axis)

    Vtofilter = copy(V)
    Vtofilter[CartesianIndex.(reach)] .= -1.0
    Vtofilter[CartesianIndex.(avoid)] .= -1.0

    Vfiltered = Vtofilter[Vtofilter .>= 0.0]

    println(("Minimum V", minimum(Vfiltered)))
    println(("Maximum V", maximum(Vfiltered)))


    # Properties
    reach = NTuple{4, Int32}[]
    avoid = NTuple{4, Int32}[] # Absorbing state

    Vsyscore = load_syscore_mat(problem.name, num_regions_per_axis)

    Vsyscoretofilter = copy(Vsyscore)
    Vsyscoretofilter[CartesianIndex.(reach)] .= -1.0
    Vsyscoretofilter[CartesianIndex.(avoid)] .= -1.0

    Vsyscorefiltered = Vsyscoretofilter[Vsyscoretofilter .>= 0.0]
    println(("Minimum Vsyscore", minimum(Vsyscorefiltered)))
    println(("Maximum Vsyscore", maximum(Vsyscorefiltered)))

    Vdiff = V .- Vsyscore
    Vdifftofilter = copy(Vdiff)
    Vdifftofilter[CartesianIndex.(reach)] .= -10.0
    Vdifftofilter[CartesianIndex.(avoid)] .= -10.0

    Vdiff = Vdifftofilter[Vdifftofilter .>= -5.0]

    println(("Minimum Vdiff", minimum(Vdiff)))
    println(("Maximum Vdiff", maximum(Vdiff)))
    println(("Mean Vdiff", mean(Vdiff)))
end

function benchmark_system(problem::SySCoReComparisonProblem)
    for num_regions_per_axis in 2:2:30
        benchmark_system(problem, num_regions_per_axis)
    end
end

end
