
export DiscreteTimeStochasticDynamics, dimstate, diminput

"""
    DiscreteTimeStochasticDynamics

Abstract type for discrete-time stochastic dynamicss, i.e. ``x_{k+1} = f(x_k, u_k, w_k)``.
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

abstract type CanDecouple end
abstract type TransformationRequired <: CanDecouple end
struct DirectDecoupling <: CanDecouple end
struct LinearTransformationRequired <: TransformationRequired end
struct CannotDecouple <: CanDecouple end

abstract type Transformation end
struct LinearTransformation{R, MR <: AbstractMatrix{R}} <: Transformation
    T::MR
    Tinv::MR
end