module DirectVsDecoupledIMDP

using BenchmarkTools, ProgressMeter
using IntervalMDP, IntervalSySCoRe

include("systems/robot_2d.jl")

input_split = (20, 20)
time_horizon = 10

function benchmark_direct(state_split)
    abstraction_time = @benchmark robot_2d_direct(;state_split=$state_split, input_split=$input_split)

    GC.gc()
    GC.enable(false)
    mdp, reach, avoid = robot_2d_direct(;state_split=state_split, input_split=input_split)
    abstraction_mem = Base.gc_live_bytes()
    GC.enable(true)

    prop = FiniteTimeReachAvoid(reach, avoid, time_horizon)
    spec = Specification(prop, Pessimistic, Maximize)
    prob = Problem(mdp, spec)

    certification_time = @benchmark value_iteration($prob)

    abstraction_median_seconds = time(median(abstraction_time)) / 1e9
    certification_median_seconds = time(median(certification_time)) / 1e9

    GC.gc()
    GC.enable(false)
    V, k, res = value_iteration(prob)
    certification_mem = Base.gc_live_bytes()
    GC.enable(true)

    return (abstraction_time=abstraction_median_seconds, abstraction_mem=abstraction_mem,
            certification_time=certification_median_seconds, certification_mem=certification_mem, V=V)
end

function benchmark_decoupled(state_split)
    abstraction_time = @benchmark robot_2d_decoupled(;state_split=$state_split, input_split=$input_split)

    GC.gc()
    GC.enable(false)
    mdp, reach, avoid = robot_2d_decoupled(;state_split=state_split, input_split=input_split)
    abstraction_mem = Base.gc_live_bytes()
    GC.enable(true)

    prop = FiniteTimeReachAvoid(reach, avoid, time_horizon)
    spec = Specification(prop, Pessimistic, Maximize)
    prob = Problem(mdp, spec)

    certification_time = @benchmark value_iteration($prob)

    abstraction_median_seconds = time(median(abstraction_time)) / 1e9
    certification_median_seconds = time(median(certification_time)) / 1e9

    GC.gc()
    GC.enable(false)
    V, k, res = value_iteration(prob)
    certification_mem = Base.gc_live_bytes()
    GC.enable(true)

    return (abstraction_time=abstraction_median_seconds, abstraction_mem=abstraction_mem,
            certification_time=certification_median_seconds, certification_mem=certification_mem, V=V)
end

function compare()
    state_splits = [(s, s) for s in 10:10:200]

    res = []

    @showprogress for state_split in state_splits
        direct = benchmark_direct(state_split)
        decoupled = benchmark_decoupled(state_split)

        push!(res, (state_split=state_split, direct=direct, decoupled=decoupled))
    end

    # TODO: save results to CSV
    # TODO: plot results
end

end


DirectVsDecoupledIMDP.compare()
