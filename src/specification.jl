export AbstractRegionReachability,
    FiniteTimeRegionReachability,
    InfiniteTimeRegionReachability,
    ExactTimeRegionReachability,
    reach,
    dim
export AbstractRegionReachAvoid,
    FiniteTimeRegionReachAvoid,
    InfiniteTimeRegionReachAvoid,
    ExactTimeRegionReachAvoid,
    avoid
export AbstractRegionSafety, FiniteTimeRegionSafety, InfiniteTimeRegionSafety
export AbstractionProblem, system, specification

# Reachability
abstract type AbstractRegionReachability <: Property end

"""
    FiniteTimeRegionReachability

A struct representing a finite-time reachability property.
"""
struct FiniteTimeRegionReachability{S<:LazySet,T<:Integer} <: AbstractRegionReachability
    reach_set::S
    time_horizon::T
end

IntervalMDP.isfinitetime(::FiniteTimeRegionReachability) = true
IntervalMDP.time_horizon(prop::FiniteTimeRegionReachability) = prop.time_horizon

"""
    reach

Return the reach region of a reachability or reach-avoid property.
"""
reach(prop::FiniteTimeRegionReachability) = prop.reach_set

"""
    dim

Return the dimension of the reach and avoid regions of a regional property.
"""
dim(prop::FiniteTimeRegionReachability) = LazySets.dim(reach(prop))

function transform(prop::FiniteTimeRegionReachability, transformation::LinearTransformation)
    reach_set = concretize(transformation.T * prop.reach_set)
    return FiniteTimeRegionReachability(reach_set, prop.time_horizon)
end

"""
    InfiniteTimeRegionReachability

A struct representing a infinite-time reachability property.
"""
struct InfiniteTimeRegionReachability{S<:LazySet,T<:Real} <: AbstractRegionReachability
    reach_set::S
    convergence_eps::T
end

IntervalMDP.isfinitetime(::InfiniteTimeRegionReachability) = false
IntervalMDP.convergence_eps(prop::InfiniteTimeRegionReachability) = prop.convergence_eps
reach(prop::InfiniteTimeRegionReachability) = prop.reach_set
dim(prop::InfiniteTimeRegionReachability) = LazySets.dim(reach(prop))

function transform(
    prop::InfiniteTimeRegionReachability,
    transformation::LinearTransformation,
)
    reach_set = concretize(transformation.T * prop.reach_set)
    return InfiniteTimeRegionReachability(reach_set, prop.convergence_eps)
end

"""
    ExactTimeRegionReachability

A struct representing a exact-time reachability property.
"""
struct ExactTimeRegionReachability{S<:LazySet,T<:Integer} <: AbstractRegionReachability
    reach_set::S
    time_horizon::T
end

IntervalMDP.isfinitetime(::ExactTimeRegionReachability) = true
IntervalMDP.time_horizon(prop::ExactTimeRegionReachability) = prop.time_horizon
reach(prop::ExactTimeRegionReachability) = prop.reach_set
dim(prop::ExactTimeRegionReachability) = LazySets.dim(reach(prop))

function transform(prop::ExactTimeRegionReachability, transformation::LinearTransformation)
    reach_set = concretize(transformation.T * prop.reach_set)
    return ExactTimeRegionReachability(reach_set, prop.convergence_eps)
end

## Reach-avoid
abstract type AbstractRegionReachAvoid <: Property end

"""
    FiniteTimeRegionReachAvoid

A struct representing a finite-time reach-avoid property.
"""
struct FiniteTimeRegionReachAvoid{S<:LazySet,R<:LazySet,T<:Integer} <:
       AbstractRegionReachAvoid
    reach_set::S
    avoid_set::R
    time_horizon::T
end

IntervalMDP.isfinitetime(::FiniteTimeRegionReachAvoid) = true
IntervalMDP.time_horizon(prop::FiniteTimeRegionReachAvoid) = prop.time_horizon
reach(prop::FiniteTimeRegionReachAvoid) = prop.reach_set

"""
    avoid

Return the avoid region of a reach-avoid or safety property.
"""
avoid(prop::FiniteTimeRegionReachAvoid) = prop.avoid_set
dim(prop::FiniteTimeRegionReachAvoid) = LazySets.dim(reach(prop))

function transform(prop::FiniteTimeRegionReachAvoid, transformation::LinearTransformation)
    reach_set = concretize(transformation.T * prop.reach_set)
    avoid_set = concretize(transformation.T * prop.avoid_set)
    return FiniteTimeRegionReachAvoid(reach_set, avoid_set, prop.time_horizon)
end

"""
    InfiniteTimeRegionReachAvoid

A struct representing a infinite-time reach-avoid property.
"""
struct InfiniteTimeRegionReachAvoid{S<:LazySet,R<:LazySet,T<:Real} <:
       AbstractRegionReachAvoid
    reach_set::S
    avoid_set::R
    convergence_eps::T
end

IntervalMDP.isfinitetime(::InfiniteTimeRegionReachAvoid) = false
IntervalMDP.convergence_eps(prop::InfiniteTimeRegionReachAvoid) = prop.convergence_eps
reach(prop::InfiniteTimeRegionReachAvoid) = prop.reach_set
avoid(prop::InfiniteTimeRegionReachAvoid) = prop.avoid_set
dim(prop::InfiniteTimeRegionReachAvoid) = LazySets.dim(reach(prop))

function transform(prop::InfiniteTimeRegionReachAvoid, transformation::LinearTransformation)
    reach_set = concretize(transformation.T * prop.reach_set)
    avoid_set = concretize(transformation.T * prop.avoid_set)
    return InfiniteTimeRegionReachAvoid(reach_set, avoid_set, prop.convergence_eps)
end

"""
    ExactTimeRegionReachAvoid

A struct representing a Exact-time reach-avoid property.
"""
struct ExactTimeRegionReachAvoid{S<:LazySet,R<:LazySet,T<:Integer} <:
       AbstractRegionReachAvoid
    reach_set::S
    avoid_set::R
    time_horizon::T
end

IntervalMDP.isfinitetime(::ExactTimeRegionReachAvoid) = true
IntervalMDP.time_horizon(prop::ExactTimeRegionReachAvoid) = prop.time_horizon
reach(prop::ExactTimeRegionReachAvoid) = prop.reach_set
avoid(prop::ExactTimeRegionReachAvoid) = prop.avoid_set
dim(prop::ExactTimeRegionReachAvoid) = LazySets.dim(reach(prop))

function transform(prop::ExactTimeRegionReachAvoid, transformation::LinearTransformation)
    reach_set = concretize(transformation.T * prop.reach_set)
    avoid_set = concretize(transformation.T * prop.avoid_set)
    return ExactTimeRegionReachAvoid(reach_set, avoid_set, prop.convergence_eps)
end

## Safety
abstract type AbstractRegionSafety <: Property end

"""
    FiniteTimeRegionSafety

A struct representing a finite-time safety property.
"""
struct FiniteTimeRegionSafety{S<:LazySet,T<:Integer} <: AbstractRegionSafety
    avoid_set::S
    time_horizon::T
end

IntervalMDP.isfinitetime(::FiniteTimeRegionSafety) = true
IntervalMDP.time_horizon(prop::FiniteTimeRegionSafety) = prop.time_horizon
avoid(prop::FiniteTimeRegionSafety) = prop.avoid_set
dim(prop::FiniteTimeRegionSafety) = LazySets.dim(avoid(prop))

function transform(prop::FiniteTimeRegionSafety, transformation::LinearTransformation)
    avoid_set = concretize(transformation.T * prop.avoid_set)
    return FiniteTimeRegionSafety(avoid_set, prop.time_horizon)
end

"""
    InfiniteTimeRegionSafety

A struct representing a infinite-time safety property.
"""
struct InfiniteTimeRegionSafety{S<:LazySet,T<:Real} <: AbstractRegionSafety
    avoid_set::S
    convergence_eps::T
end

IntervalMDP.isfinitetime(::InfiniteTimeRegionSafety) = false
IntervalMDP.convergence_eps(prop::InfiniteTimeRegionSafety) = prop.convergence_eps
avoid(prop::InfiniteTimeRegionSafety) = prop.avoid_set
dim(prop::InfiniteTimeRegionSafety) = LazySets.dim(avoid(prop))

function transform(prop::InfiniteTimeRegionSafety, transformation::LinearTransformation)
    avoid_set = concretize(transformation.T * prop.avoid_set)
    return InfiniteTimeRegionSafety(avoid_set, prop.convergence_eps)
end

## Problem

"""
    AbstractionProblem

A struct of a system and a specification to be used in the abstraction process.
"""
struct AbstractionProblem{S<:System,P<:Specification}
    system::S
    specification::P
end

"""
    system

Return the system of an abstraction problem.
"""
system(prob::AbstractionProblem) = prob.system

"""
    specification

Return the specification of an abstraction problem.
"""
specification(prob::AbstractionProblem) = prob.specification

"""
    decouple

Decoupled the noise in the system dynamics of an `AbstractionProblem` if possible,
and transform the specification accordingly. 

If the system dynamics cannot be decoupled, an error is thrown.

Returns a tuple of the decoupled `AbstractionProblem` and the transformation used.
"""
function decouple(prob::AbstractionProblem)
    sys = system(prob)
    return _decouple(prob, decouplingmode(sys))
end

function _decouple(prob::AbstractionProblem, ::CannotDecouple)
    throw(ArgumentError("The system dynamics cannot be decoupled."))
end

function _decouple(prob::AbstractionProblem, ::LinearTransformationRequired)
    sys = system(prob)
    T, sys = decouple(sys)
    spec = specification(prob)

    # Transform the specification
    spec = transform(spec, T)

    return AbstractionProblem(sys, spec), T
end

function _decouple(prob::AbstractionProblem, ::IsDecoupled)
    # No transformation needed
    return prob, LinearTransformation(I, I)
end

function transform(spec::Specification, transformation::LinearTransformation) 
    prop = transform(system_property(spec), transformation)
    spec = Specification(prop, satisfaction_mode(spec), strategy_mode(spec))
    return spec
end
