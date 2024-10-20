export StochasticSwitchedDynamics

"""
    StochasticSwitchedDynamics

A type that represents dynamics with a stochastic transition between the modes.
"""
struct StochasticSwitchedDynamics <: DiscreteTimeStochasticDynamics
    dynamics::Vector{<:DiscreteTimeStochasticDynamics}
    weights::Vector{Float64}

    function StochasticSwitchedDynamics(dynamics::Vector{<:DiscreteTimeStochasticDynamics}, weights::Vector{Float64})
        dstate = dimstate(first(dynamics))
        dinput = diminput(first(dynamics))

        for dyn in dynamics
            if dimstate(dyn) != dstate
                throw(DimensionMismatch("The dimension of the state space must be the same for all dynamics"))
            end

            if diminput(dyn) != dinput
                throw(DimensionMismatch("The dimension of the input space must be the same for all dynamics"))
            end
        end
        
        if length(dynamics) != length(weights)
            throw(DimensionMismatch("The number of dynamics and weights must be the same"))
        end

        if sum(weights) != 1.0
            throw(ArgumentError("The sum of the weights must be equal to 1"))
        end

        new(dynamics, weights)
    end
end

dimstate(dyn::StochasticSwitchedDynamics) = dimstate(first(dyn.dynamics))
diminput(dyn::StochasticSwitchedDynamics) = diminput(first(dyn.dynamics))