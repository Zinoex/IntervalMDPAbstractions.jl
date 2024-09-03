module DirectVsDecoupledIMDP

using BenchmarkTools, ProgressMeter
using DataFrames, CSV, Statistics
using IntervalMDP, IntervalSySCoRe

include("systems/robot_2d.jl")

input_split = (20, 20)
time_horizon = 10

function benchmark_direct(state_split)
    @info "Benchmarking direct"

    @info "Measuring abstraction time"
    abstraction_time = @benchmark robot_2d_direct(;state_split=$state_split, input_split=$input_split)
    abstraction_median_seconds = time(median(abstraction_time)) / 1e9
    @info "Abstraction time" abstraction_median_seconds

    mdp, reach, avoid = robot_2d_direct(;state_split=state_split, input_split=input_split)

    prop = FiniteTimeReachAvoid(reach, avoid, time_horizon)
    spec = Specification(prop, Pessimistic, Maximize)
    prob = Problem(mdp, spec)

    @info "Measuring certification time"
    certification_time = @benchmark value_iteration($prob)
    certification_median_seconds = time(median(certification_time)) / 1e9
    @info "Certification time" certification_median_seconds

    @info "Measuring peak mem"
    BenchmarkTools.gcscrub()
    GC.enable(false)
    V, k, res = value_iteration(prob)
    peak_mem = Base.gc_live_bytes() / 1024^2
    GC.enable(true)
    @info "Peak mem" peak_mem

    return (abstraction_time=abstraction_median_seconds, certification_time=certification_median_seconds, peak_mem=peak_mem, V=V)
end

function benchmark_decoupled(state_split)
    @info "Benchmarking decoupled"

    @info "Measuring abstraction time"
    abstraction_time = @benchmark robot_2d_decoupled(;state_split=$state_split, input_split=$input_split)
    abstraction_median_seconds = time(median(abstraction_time)) / 1e9
    @info "Abstraction time" abstraction_median_seconds

    mdp, reach, avoid = robot_2d_decoupled(;state_split=state_split, input_split=input_split)

    prop = FiniteTimeReachAvoid(reach, avoid, time_horizon)
    spec = Specification(prop, Pessimistic, Maximize)
    prob = Problem(mdp, spec)

    @info "Measuring certification time"
    certification_time = @benchmark value_iteration($prob)
    certification_median_seconds = time(median(certification_time)) / 1e9
    @info "Certification time" certification_median_seconds

    @info "Measuring peak mem"
    BenchmarkTools.gcscrub()
    GC.enable(false)
    V, k, res = value_iteration(prob)
    peak_mem = Base.gc_live_bytes() / 1024^2
    GC.enable(true)
    @info "Peak mem" peak_mem

    return (abstraction_time=abstraction_median_seconds, certification_time=certification_median_seconds, peak_mem=peak_mem, V=V)
end

function benchmark()
    state_splits = [(s, s) for s in 5:5:50]

    res = []

    @showprogress dt=1 desc="Benchmarking..." for state_split in state_splits
        @info "Benchmarking state_split" state_split

        direct = benchmark_direct(state_split)
        decoupled = benchmark_decoupled(state_split)

        push!(res, (state_split=state_split, direct=direct, decoupled=decoupled))
    end

    return res
end

function to_dataframe(res)
    res = map(res) do row
        # Remove the first element of the value function, which is the absorbing avoid state
        V_diff = vec(row.decoupled.V[2:end, 2:end]) - row.direct.V[2:end] 

        (
            state_split=row.state_split[1],
            direct_abstraction_time=row.direct.abstraction_time,
            direct_certification_time=row.direct.certification_time,
            direct_peak_mem=row.direct.peak_mem,
            direct_Psat_min=minimum(row.direct.V[2:end]),
            decoupled_abstraction_time=row.decoupled.abstraction_time,
            decoupled_certification_time=row.decoupled.certification_time,
            decoupled_peak_mem=row.decoupled.peak_mem,
            decoupled_Psat_min=minimum(row.decoupled.V[2:end, 2:end]),
            Psat_diff_max=maximum(V_diff),
            Psat_diff_median=median(V_diff),
            Psat_diff_mean=mean(V_diff),
        )
    end

    df = DataFrame(res)
    return df
end

function save_results(df)
    # Read-write permissions for user and group, read-only for others
    # No execute permissions for anyone (since it is only data files anyways)
    mkpath("results", mode=0o664)
    CSV.write("results/direct_vs_decoupled_imdp.csv", df)
end

function read_results()
    df = CSV.read("results/direct_vs_decoupled_imdp.csv", DataFrame)
    return df
end

# TODO: plot results

function compare()
    res = benchmark()
    df = to_dataframe(res)
    save_results(df)
end

end


DirectVsDecoupledIMDP.compare()
