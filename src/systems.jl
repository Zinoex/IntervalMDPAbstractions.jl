export System, dynamics, initial

"""
    System{D<:DiscreteTimeStochasticDynamics, I<:LazySet}

A struct representing a system with dynamics of type `D`, initial set of type `I`, reach set of type `R`, and avoid set of type `A`.
"""
struct System{D<:DiscreteTimeStochasticDynamics,I<:LazySet}
    dynamics::D
    initial::I
end
dynamics(sys) = sys.dynamics
initial(sys) = sys.initial
dimstate(sys::System) = dimstate(dynamics(sys))
diminput(sys::System) = diminput(dynamics(sys))
