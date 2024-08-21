
export DiscreteTimeStochasticDynamics, dimstate, diminput, dimnoise

"""
    DiscreteTimeStochasticDynamics

Abstract type for discrete-time stochastic dynamicss, i.e. `x_{k+1} = f(x_k, u_k, w_k)`.
"""
abstract type DiscreteTimeStochasticDynamics end

"""
    dimstate(dyn::DiscreteTimeStochasticDynamics)

Return the dimension of the state space of the dynamics `dyn`.
"""
function dimstate end

"""
    diminput(dyn::DiscreteTimeStochasticDynamics)

Return the dimension of the input space of the dynamics `dyn`.
"""
function diminput end

"""
    dimnoise(dyn::DiscreteTimeStochasticDynamics)

Return the dimension of the noise space of the dynamics `dyn`.
"""
function dimnoise end
