export TargetAbstractionModel, DirectIMDP, SparseDirectIMDP, DecoupledIMDP

"""
    TargetAbstractionModel

Type hierarchy to represent the target abstraction model.
"""
abstract type TargetAbstractionModel end

"""
    DirectIMDP

The traditional target for IMDP abstractions. This is a standard IMDP where each region in
the abstraction corresponds to a state in the IMDP with one additional state for transitions
to outside the partitioned region.
"""
struct DirectIMDP <: TargetAbstractionModel end

includetransition(target::DirectIMDP, prob) = true

"""
    SparseDirectIMDP

Similar to [`DirectIMDP`](@ref), but uses a sparse matrix to represent the transition probabilities.
"""
Base.@kwdef struct SparseDirectIMDP <: TargetAbstractionModel 
    sparsity_threshold = 1e-9
end

includetransition(target::SparseDirectIMDP, prob) = target.sparsity_threshold < prob

"""
    DecoupledIMDP

A target for IMDP abstractions where, for each source state, the transition probabilities
are orthogonally decomposed. One state is appended along each axis to capture the transitions
to outside the partitioned region. 

Benefits compared to `DirectIMDP` include less memory usage, faster computation of the
transition probabilities and value iteration, and tighter uncertainty set (see [1] for a proof).

[1] TODO: Add reference to paper.
"""
struct DecoupledIMDP end