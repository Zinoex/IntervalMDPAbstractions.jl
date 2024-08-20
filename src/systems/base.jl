
export DiscreteTimeStochasticSystem

"""
    DiscreteTimeStochasticSystem

Abstract type for discrete-time stochastic systems, i.e. `x_{k+1} = f(x_k, u_k, w_k)`.
"""
abstract type DiscreteTimeStochasticSystem end

#### Additive noise systems
export AdditiveNoiseSystem, nominal_dynamics

"""
    AdditiveNoiseSystem

System with additive noise, i.e. `x_{k+1} = f(x_k, u_k) + w_k`.
"""
abstract type AdditiveNoiseSystem <: DiscreteTimeStochasticSystem end

"""
    nominal_dynamics(sys::AdditiveNoiseSystem, X::LazySet, U::LazySet)

Compute the reachable set under the nominal dynamics of the system `sys` over the state region `X` and control region `U`.
The nominal dynamics are given by `x_{k+1} = f(x_k, u_k)`, i.e. only well-defined for systems with additive noise.
"""
function nominal_dynamics end

## Additive Gaussian noise systems
export AdditiveGaussianSystem, stddev

"""
    AdditiveGaussianSystem

System with additive Gaussian noise, i.e. `x_{k+1} = f(x_k, u_k) + w_k`, where `w_k ~ N(0, diag(w_stddev))`.
The benefit is that bounds on the transition probability can be computed efficiently with analytical formulas.
"""
abstract type AdditiveGaussianSystem <: AdditiveNoiseSystem end

"""
    stddev(sys::AdditiveGaussianSystem)

Return the standard deviation of the Gaussian noise of the system `sys`.
"""
function stddev end