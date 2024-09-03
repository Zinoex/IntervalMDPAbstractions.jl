module IntervalSySCoRe

using LinearAlgebra
using SpecialFunctions: erf
using IrrationalConstants: invsqrt2
using LazySets, IntervalMDP

# Dynamics
include("dynamics/base.jl")
include("dynamics/additive_noise.jl")
include("dynamics/LinearAdditiveNoiseDynamics.jl")

include("systems.jl")

# Abstractions
include("abstractions/input_abstraction.jl")
include("abstractions/state_abstraction.jl")
include("abstractions/target.jl")

include("abstractions/additive_noise.jl")

end
