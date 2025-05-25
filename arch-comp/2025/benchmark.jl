using Revise, BenchmarkTools, LazySets
using IntervalMDP, IntervalMDPAbstractions, ArchCompStochasticModels
using InvertedIndices

function synthesismode2strategymode(synthesismode)
    if synthesismode == ArchCompStochasticModels.maximize
        return Maximize
    else
        return Minimize
    end
end

include("automated_anaesthesia.jl")
include("building_automation_system.jl")
include("van_der_pol.jl")
include("patrol_robot.jl")
include("automated_vehicle.jl")
include("integration_chain.jl")

function run_all_benchmarks()
    @info "Running Automated Anaesthesia Benchmark..."
    res = odimdp_as_faa_safety()
    @info "Completed Automated Anaesthesia Benchmark" res

    @info "Running Building Automation System CS1 Benchmark..."
    res = odimdp_bs_cs1_safety()
    @info "Completed Building Automation System CS1 Benchmark" res

    @info "Running Building Automation System CS2 Benchmark..."
    res = odimdp_bs_cs2_safety()
    @info "Completed Building Automation System CS2 Benchmark" res

    @info "Running controlled Gaussian Van der Pol Benchmark..."
    res = odimdp_vp_gauss_quantitative()
    @info "Completed controlled Gaussian Van der Pol Benchmark" res

    @info "Running Reduced Patrol Robot Reachability Benchmark..."
    res = odimdp_rpr_it_r()
    @info "Completed Reduced Patrol Robot Reachability Benchmark" res

    @info "Running Reduced Patrol Robot Reach-Avoid Benchmark..."
    res = odimdp_rpr_it_ra()
    @info "Completed Reduced Patrol Robot Reach-Avoid Benchmark" res

    @info "Running Automated Vehicle Benchmark..."
    res = odimdp_av_ft_ra()
    @info "Completed Automated Vehicle Benchmark" res

    @info "Running Integration Chain Benchmark..."
    for n in 1:5
        res = odimdp_ic_it_ra(i)
        @info "Integration Chain Benchmark" n res
    end
end
