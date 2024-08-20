
abstract type InputAbstraction end

"""
    InputGridSplit

Input abstraction for splitting the input space into a grid.
"""
struct InputGridSplit <: InputAbstraction
    input_space::Hyperrectangle
    splits
end

function inputs(input::InputGridSplit)
    regions = LazySets.split(input.input_space, input.splits)
    
    return regions
end

"""
    InputGridSplit

Input abstraction for _points_ on a grid of the input space.
"""
struct InputLinRange <: InputAbstraction
    input_space::Hyperrectangle
    ranges
end

function inputs(input::InputLinRange)
    l = low(input.input_space)
    h = high(input.input_space)
    ranges = [LinRange(l, h, num_steps) for (l, h, num_steps) in zip(l, h, input.ranges)]

    regions = [Singleton([x]) for x in Iterators.product(ranges...)]
    
    return regions
end