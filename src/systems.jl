export System, dynamics, initial, reach, avoid

"""
    System{D<:DiscreteTimeStochasticDynamics, I<:LazySet, R<:LazySet, A<:LazySet}

A struct representing a system with dynamics of type `D`, initial set of type `I`, reach set of type `R`, and avoid set of type `A`.
"""
struct System{D <: DiscreteTimeStochasticDynamics, I<:LazySet, R<:LazySet, A<:LazySet}
    dynamics::D
    initial::I
    reach::R
    avoid::A
end
dynamics(sys) = sys.dynamics
initial(sys) = sys.initial
reach(sys) = sys.reach
avoid(sys) = sys.avoid