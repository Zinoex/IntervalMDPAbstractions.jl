export System, dynamics, initial

"""
    System{D<:DiscreteTimeStochasticDynamics, I<:LazySet}

A struct representing a system with dynamics and an initial set.
"""
struct System{D<:DiscreteTimeStochasticDynamics,I<:LazySet}
    dynamics::D
    initial::I
end

"""
    dynamics(sys::System)

Return the dynamics of the system.
"""
dynamics(sys) = sys.dynamics

"""
    initial(sys::System)

Return the initial set of the system.
"""
initial(sys) = sys.initial

dimstate(sys::System) = dimstate(dynamics(sys))
diminput(sys::System) = diminput(dynamics(sys))
