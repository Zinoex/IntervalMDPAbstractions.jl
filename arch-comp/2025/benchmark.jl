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
include("integration_chain.jl")

function run_all_benchmarks()

end
