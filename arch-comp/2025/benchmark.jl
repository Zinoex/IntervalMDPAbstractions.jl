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
    print("Running Automated Anaesthesia Benchmark...")
    res = odimdp_as_faa_safety()
    display(res)

    print("Running Building Automation System CS1 Benchmark...")
    res = odimdp_bs_cs1_safety()
    display(res)

    print("Running Building Automation System CS2 Benchmark...")
    res = odimdp_bs_cs2_safety()
    display(res)

    print("Running Van der Pol Benchmark...")
    res = odimdp_vp_gauss_quantitative()
    display(res)

    print("Running Patrol Robot Benchmark...")
    res = odimdp_rpr_it_ra()
    display(res)

    print("Running Automated Vehicle Benchmark...")
    res = odimdp_av_it_ra()
    display(res)

    print("Running Integration Chain Benchmark...")
    for n in 1:5
        res = odimdp_ic_it_ra(i)
        println("Integration Chain Benchmark $n")
        display(res)
    end
end
