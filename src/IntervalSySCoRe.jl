module IntervalSySCoRe

using LinearAlgebra, SpecialFunctions
using IrrationalConstants: invsqrt2
using LazySets, IntervalMDP


# Systems
include("systems/base.jl")
include("systems/AffineAdditiveGaussian.jl")

end
