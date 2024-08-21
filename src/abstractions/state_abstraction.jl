export StateGridSplit, numregions, statespace

abstract type StateAbstraction end

"""
    StateGridSplit

State abstraction for splitting the state space into a grid.
"""
struct StateGridSplit <: StateAbstraction
    state_space::Hyperrectangle
    splits
end

splits(state::StateGridSplit) = state.splits
regions(state::StateGridSplit) = LazySets.split(state.state_space, state.splits)
numregions(state::StateGridSplit) = prod(state.splits)
statespace(state::StateGridSplit) = state.state_space
