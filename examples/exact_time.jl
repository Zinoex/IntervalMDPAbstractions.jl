using IntervalMDP

struct ExactTimeReachAvoid{VT <: AbstractVector{<:CartesianIndex}} <: IntervalMDP.AbstractReachAvoid
    reach::VT
    avoid::VT
    time_horizon::Any
end

function ExactTimeReachAvoid(
    reach::Vector{<:IntervalMDP.UnionIndex},
    avoid::Vector{<:IntervalMDP.UnionIndex},
    time_horizon,
)
    reach = CartesianIndex.(reach)
    avoid = CartesianIndex.(avoid)
    return ExactTimeReachAvoid(reach, avoid, time_horizon)
end

function IntervalMDP.checkproperty!(prop::ExactTimeReachAvoid, system)
    IntervalMDP.checktimehorizon!(prop, system)
    IntervalMDP.checkterminal!(IntervalMDP.terminal_states(prop), system)
    IntervalMDP.checkdisjoint!(IntervalMDP.reach(prop), IntervalMDP.avoid(prop))
end

"""
    isfinitetime(prop::ExactTimeReachAvoid)

Return `true` for ExactTimeReachAvoid.
"""
IntervalMDP.isfinitetime(prop::ExactTimeReachAvoid) = true

"""
    time_horizon(prop::ExactTimeReachAvoid)

Return the time horizon of a finite time reach-avoid property.
"""
IntervalMDP.time_horizon(prop::ExactTimeReachAvoid) = prop.time_horizon

"""
    terminal_states(prop::ExactTimeReachAvoid)

Return the set of terminal states of a finite time reach-avoid property.
That is, the union of the reach and avoid sets.
"""
IntervalMDP.terminal_states(prop::ExactTimeReachAvoid) = [prop.reach; prop.avoid]

"""
    reach(prop::ExactTimeReachAvoid)

Return the set of target states.
"""
IntervalMDP.reach(prop::ExactTimeReachAvoid) = prop.reach

"""
    avoid(prop::ExactTimeReachAvoid)

Return the set of states to avoid.
"""
IntervalMDP.avoid(prop::ExactTimeReachAvoid) = prop.avoid

function IntervalMDP.postprocess_value_function!(value_function, prop::ExactTimeReachAvoid)
    @inbounds value_function.current[IntervalMDP.avoid(prop)] .= 0.0
end
