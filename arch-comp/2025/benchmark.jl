using Revise, LazySets
using IntervalMDP, IntervalMDPAbstractions, ArchCompStochasticModels

function synthesismode2strategymode(synthesismode)
    if synthesismode == ArchCompStochasticModels.maximize
        return Maximize
    else
        return Minimize
    end
end

include("automated_anaesthesia.jl")

function run_all_benchmarks()

end
