module IntervalSySCoRe

using LinearAlgebra, SpecialFunctions
using IrrationalConstants: invsqrt2
using LazySets, IntervalMDP

include("running_example.jl")
export running_example, running_example_direct

include("modified_running_example.jl")
export modified_running_example, modified_running_example_direct

end
