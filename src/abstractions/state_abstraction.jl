export StateUniformGridSplit, numregions, statespace

abstract type StateAbstraction end

"""
    StateUniformGridSplit

State abstraction for splitting the state space into a grid.
"""
struct StateUniformGridSplit{N, H<:AbstractHyperrectangle} <: StateAbstraction
    state_space::Hyperrectangle
    splits::NTuple{N, Int}
    regions::Vector{H}
end

function StateUniformGridSplit(state_space::Hyperrectangle, splits::NTuple{N, Int}) where {N}
    regions = LazySets.split(state_space, [axisregions for axisregions in splits])
    return StateUniformGridSplit(state_space, splits, regions)
end

splits(state::StateUniformGridSplit) = state.splits
regions(state::StateUniformGridSplit) = state.regions
numregions(state::StateUniformGridSplit) = prod(state.splits)
statespace(state::StateUniformGridSplit) = state.state_space
