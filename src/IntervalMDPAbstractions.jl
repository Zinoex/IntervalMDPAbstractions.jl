module IntervalMDPAbstractions

using LinearAlgebra, SparseArrays
using SpecialFunctions: erf
using IrrationalConstants: invsqrt2
using TaylorModels: TaylorModelN, set_variables, constant_term, linear_polynomial, remainder
using IntervalArithmetic: IntervalBox, inf, sup
using LazySets, IntervalMDP
using HiGHS, JuMP

function __init__()
    # WARN: This is generally not recommended, but it is necessary to multi-thread the abstraction,
    # since LazySets defaults to GLPK for the LP solver, and GLPK is not thread-safe (non-reentrant).
    @eval @inline function LazySets.default_lp_solver_factory(::Type{<:AbstractFloat})
        return JuMP.optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true)
    end
end

include("utils.jl")
include("matrix.jl")

# Dynamics
include("dynamics/base.jl")
include("dynamics/additive_noise.jl")
include("dynamics/AffineAdditiveNoiseDynamics.jl")
include("dynamics/NonlinearAdditiveNoiseDynamics.jl")
include("dynamics/UncertainPWAAdditiveNoiseDynamics.jl")
include("dynamics/StochasticSwitchedDynamics.jl")
include("dynamics/AbstractedGaussianProcess.jl")

include("systems.jl")
include("specification.jl")

# Abstractions
include("abstractions/input_abstraction.jl")
include("abstractions/state_abstraction.jl")
include("abstractions/target.jl")

include("abstractions/abstraction.jl")
include("abstractions/additive_noise.jl")
include("abstractions/mixture.jl")
include("abstractions/gaussian_process.jl")

end
