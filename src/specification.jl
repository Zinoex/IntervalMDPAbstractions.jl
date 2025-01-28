export AbstractRegionReachability,
    FiniteTimeRegionReachability, InfiniteTimeRegionReachability, reach, dim
export AbstractRegionReachAvoid,
    FiniteTimeRegionReachAvoid, InfiniteTimeRegionReachAvoid, avoid
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
