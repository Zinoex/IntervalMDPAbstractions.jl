export TargetAbstractionModel, IMDPTarget, SparseIMDPTarget, OrthogonalIMDPTarget, SparseOrthogonalIMDPTarget

"""
    TargetAbstractionModel

Type hierarchy to represent the target abstraction model.
"""
abstract type TargetAbstractionModel end

########
# IMDP #
########
abstract type AbstractIMDPTarget <: TargetAbstractionModel end

"""
    IMDPTarget

The traditional target for IMDP abstractions. This is a standard IMDP where each region in
the abstraction corresponds to a state in the IMDP with one additional state for transitions
to outside the partitioned region.
"""
struct IMDPTarget <: AbstractIMDPTarget end

includetransition(target::IMDPTarget, prob) = true

"""
    SparseIMDPTarget

Similar to [`IMDPTarget`](@ref), but uses a sparse matrix to represent the transition probabilities.
"""
Base.@kwdef struct SparseIMDPTarget <: AbstractIMDPTarget 
    sparsity_threshold = 1e-9
end

includetransition(target::SparseIMDPTarget, prob) = target.sparsity_threshold < prob


###################
# Orthogonal IMDP #
###################
abstract type AbstractOrthogonalIMDPTarget <: TargetAbstractionModel end

"""
    OrthogonalIMDPTarget

A target for IMDP abstractions where, for each source state, the transition probabilities
are orthogonally decomposed. One state is appended along each axis to capture the transitions
to outside the partitioned region. 

Benefits compared to `IMDPTarget` include less memory usage, faster computation of the
transition probabilities and value iteration, and tighter uncertainty set (see [1] for a proof).

[1] TODO: Add reference to paper.
"""
struct OrthogonalIMDPTarget <: AbstractOrthogonalIMDPTarget end

includetransition(target::OrthogonalIMDPTarget, prob) = true

"""
    SparseOrthogonalIMDPTarget

Similar to [`OrthogonalIMDPTarget`](@ref), but uses a sparse matrix to represent the transition probabilities.
"""
Base.@kwdef struct SparseOrthogonalIMDPTarget <: AbstractOrthogonalIMDPTarget 
    sparsity_threshold = 1e-9
end

includetransition(target::SparseOrthogonalIMDPTarget, prob) = target.sparsity_threshold < prob