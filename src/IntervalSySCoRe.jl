module IntervalSySCoRe

using LinearAlgebra
using SpecialFunctions: erf
using IrrationalConstants: invsqrt2
using LazySets, IntervalMDP

# Systems
include("systems/base.jl")
include("systems/additive_noise.jl")
include("systems/AffineAdditiveNoise.jl")

# Abstractions
include("abstractions/input_abstraction.jl")
include("abstractions/state_abstraction.jl")
include("abstractions/target.jl")

include("abstractions/additive_noise.jl")

end
