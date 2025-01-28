export StateAbstraction, StateUniformGridSplit, regions, numregions, statespace

"""
    StateAbstraction

Abstract type for state abstractions.
"""
abstract type StateAbstraction end

"""
    StateUniformGridSplit

State abstraction for splitting the state space into a uniform grid.
"""
struct StateUniformGridSplit{N,I<:Int,H<:AbstractHyperrectangle} <: StateAbstraction
    state_space::H
    splits::NTuple{N,I}
    regions::Vector{H}
end

function StateUniformGridSplit(state_space::Hyperrectangle, splits::NTuple{N,Int}) where {N}
    regions = LazySets.split(state_space, [axisregions for axisregions in splits])
    return StateUniformGridSplit(state_space, splits, regions)
end

splits(state::StateUniformGridSplit) = state.splits

"""
    regions(state::StateUniformGridSplit)

Return the regions of the state abstraction.
"""
regions(state::StateUniformGridSplit) = state.regions

"""
    numregions(state::StateUniformGridSplit)

Return the number of regions of the state abstraction.
"""
numregions(state::StateUniformGridSplit) = prod(state.splits)

"""
    statespace(state::StateUniformGridSplit)

Return the state space of the state abstraction. This should must be a hyperrectangle
and should be equal to the union of the regions.
"""
statespace(state::StateUniformGridSplit) = state.state_space
