export AbstractRegionReachability, FiniteTimeRegionReachability, InfiniteTimeRegionReachability, reach, dim
export AbstractRegionReachAvoid, FiniteTimeRegionReachAvoid, InfiniteTimeRegionReachAvoid, avoid
export AbstractRegionSafety, FiniteTimeRegionSafety, InfiniteTimeRegionSafety
export AbstractionProblem, system, specification

# Reachability
abstract type AbstractRegionReachability <: Property end
struct FiniteTimeRegionReachability{S <: LazySet, T <: Integer} <: AbstractRegionReachability
    reach_set::S
    time_horizon::T
end

IntervalMDP.isfinitetime(::FiniteTimeRegionReachability) = true
IntervalMDP.time_horizon(prop::FiniteTimeRegionReachability) = prop.time_horizon
reach(prop::FiniteTimeRegionReachability) = prop.reach_set
dim(prop::FiniteTimeRegionReachability) = LazySets.dim(reach(prop))

struct InfiniteTimeRegionReachability{S <: LazySet, T <: Real} <: AbstractRegionReachability
    reach_set::S
    convergence_eps::T
end

IntervalMDP.isfinitetime(::InfiniteTimeRegionReachability) = false
IntervalMDP.convergence_eps(prop::InfiniteTimeRegionReachability) = prop.convergence_eps
reach(prop::InfiniteTimeRegionReachability) = prop.reach_set
dim(prop::InfiniteTimeRegionReachability) = LazySets.dim(reach(prop))

## Reach-avoid
abstract type AbstractRegionReachAvoid <: Property end
struct FiniteTimeRegionReachAvoid{S <: LazySet, R <: LazySet, T <: Integer} <: AbstractRegionReachAvoid
    reach_set::S
    avoid_set::R
    time_horizon::T
end

IntervalMDP.isfinitetime(::FiniteTimeRegionReachAvoid) = true
IntervalMDP.time_horizon(prop::FiniteTimeRegionReachAvoid) = prop.time_horizon
reach(prop::FiniteTimeRegionReachAvoid) = prop.reach_set
avoid(prop::FiniteTimeRegionReachAvoid) = prop.avoid_set
dim(prop::FiniteTimeRegionReachAvoid) = LazySets.dim(reach(prop))

struct InfiniteTimeRegionReachAvoid{S <: LazySet, R <: LazySet, T <: Real} <: AbstractRegionReachAvoid
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
struct FiniteTimeRegionSafety{S <: LazySet, T <: Integer} <: AbstractRegionSafety
    avoid_set::S
    time_horizon::T
end

IntervalMDP.isfinitetime(::FiniteTimeRegionSafety) = true
IntervalMDP.time_horizon(prop::FiniteTimeRegionSafety) = prop.time_horizon
avoid(prop::FiniteTimeRegionSafety) = prop.avoid_set
dim(prop::FiniteTimeRegionSafety) = LazySets.dim(avoid(prop))

struct InfiniteTimeRegionSafety{S <: LazySet, T <: Real} <: AbstractRegionSafety
    avoid_set::S
    convergence_eps::T
end

IntervalMDP.isfinitetime(::InfiniteTimeRegionSafety) = false
IntervalMDP.convergence_eps(prop::InfiniteTimeRegionSafety) = prop.convergence_eps
avoid(prop::InfiniteTimeRegionSafety) = prop.avoid_set
dim(prop::InfiniteTimeRegionSafety) = LazySets.dim(avoid(prop))

## Problem
struct AbstractionProblem{S <: System, P <: Specification}
    system::S
    specification::P
end
system(prob::AbstractionProblem) = prob.system
specification(prob::AbstractionProblem) = prob.specification