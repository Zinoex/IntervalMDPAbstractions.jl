export StateGridSplit, numregions, statespace

abstract type StateAbstraction end

"""
    StateGridSplit

State abstraction for splitting the state space into a grid.
"""
struct StateGridSplit{N, H<:AbstractHyperrectangle} <: StateAbstraction
    state_space::Hyperrectangle
    splits::NTuple{N, Int}
    regions::Vector{H}
end

function StateGridSplit(state_space::Hyperrectangle, splits::NTuple{N, Int}) where {N}
    regions = LazySets.split(state_space, [axisregions for axisregions in splits])
    return StateGridSplit(state_space, splits, regions)
end

splits(state::StateGridSplit) = state.splits
regions(state::StateGridSplit) = state.regions
numregions(state::StateGridSplit) = prod(state.splits)
statespace(state::StateGridSplit) = state.state_space
