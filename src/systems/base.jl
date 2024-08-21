
export DiscreteTimeStochasticSystem, dimstate, diminput, dimnoise

"""
    DiscreteTimeStochasticSystem

Abstract type for discrete-time stochastic systems, i.e. `x_{k+1} = f(x_k, u_k, w_k)`.
"""
abstract type DiscreteTimeStochasticSystem end

"""
    dimstate(sys::DiscreteTimeStochasticSystem)

Return the dimension of the state space of the system `sys`.
"""
function dimstate end

"""
    diminput(sys::DiscreteTimeStochasticSystem)

Return the dimension of the input space of the system `sys`.
"""
function diminput end

"""
    dimnoise(sys::DiscreteTimeStochasticSystem)

Return the dimension of the noise space of the system `sys`.
"""
function dimnoise end
