export InputAbstraction, InputGridSplit, InputLinRange, InputRandom, InputDiscrete
export inputs, numinputs

abstract type InputAbstraction end

"""
    InputGridSplit

Input abstraction for splitting the input space into a grid.
"""
struct InputGridSplit <: InputAbstraction
    input_space::Hyperrectangle
    splits::Any
end
numinputs(input::InputGridSplit) = prod(input.splits)
issetbased(input::InputGridSplit) = true
function inputs(input::InputGridSplit)
    regions = LazySets.split(input.input_space, [input.splits...])

    return regions
end

"""
    InputLinRange

Input abstraction for _points_ on a grid of the input space.
"""
struct InputLinRange <: InputAbstraction
    input_space::Hyperrectangle
    ranges::Any
end
numinputs(input::InputLinRange) = prod(input.ranges)
issetbased(input::InputLinRange) = false
function inputs(input::InputLinRange)
    l = low(input.input_space)
    h = high(input.input_space)
    ranges = [LinRange(l, h, num_steps) for (l, h, num_steps) in zip(l, h, input.ranges)]

    regions = [Singleton([xᵢ for xᵢ in x]) for x in Iterators.product(ranges...)]

    return regions
end

"""
    InputDiscrete

Input abstraction for a set of discrete points in the input space.
"""
struct InputDiscrete{S} <: InputAbstraction
    inputs::Vector{S}
end
numinputs(input::InputDiscrete) = length(input.inputs)
issetbased(input::InputDiscrete{<:Singleton}) = false
issetbased(input::InputDiscrete{<:LazySet}) = true
issetbased(input::InputDiscrete) = false
function inputs(input::InputDiscrete)
    return input.inputs
end
