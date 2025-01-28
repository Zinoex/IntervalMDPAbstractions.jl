export UncertainAffineRegion, UncertainPWAAdditiveNoiseDynamics

"""
    UncertainAffineRegion

A struct representing an uncertain affine transition, valid over a region.
That is, ``A(\\alpha) x + C(\\alpha)`` where ``A(\\alpha) = \\alpha \\underline{A} + (1 - \\alpha) \\overline{A}``
and ``C(\\alpha) = \\alpha \\underline{C} + (1 - \\alpha) \\overline{C}`` for ``\\alpha \\in [0, 1]``, valid over a region ``X``.

### Fields
- `region::LazySet{Float64}`: The region over which the affine transition is valid.
- `Alower::AbstractMatrix{Float64}`: The linear lower bound matrix.
- `Clower::AbstractVector{Float64}`: The constant lower bound vector.
- `Aupper::AbstractMatrix{Float64}`: The linear upper bound matrix.
- `Cupper::AbstractVector{Float64}`: The constant upper bound vector.

"""
struct UncertainAffineRegion{T,VT<:AbstractVector{T},MT<:AbstractMatrix{T},S<:LazySet{T}}
    region::S

    Alower::MT
    Clower::VT

    Aupper::MT
    Cupper::VT

    function UncertainAffineRegion(
        region::S,
        Alower::MT,
        Clower::VT,
        Aupper::MT,
        Cupper::VT,
    ) where {T,VT<:AbstractVector{T},MT<:AbstractMatrix{T},S<:LazySet{T}}
        if size(Alower) != size(Aupper)
            throw(DimensionMismatch("The size of Alower and Aupper must be the same"))
        end

        if size(Clower, 1) != size(Clower, 1)
            throw(DimensionMismatch("The size of Clower and Cupper must be the same"))
        end

        n = LazySets.dim(region)

        if size(Alower, 2) != n
            throw(
                DimensionMismatch(
                    "The number of columns in Alower must be equal to the dimensionality of the region",
                ),
            )
        end

        new{T,VT,MT,S}(region, Alower, Clower, Aupper, Cupper)
    end
end
overapproximate(transformation::UncertainAffineRegion, input::LazySet) = ConvexHull(
    transformation.Alower * input + transformation.Clower,
    transformation.Aupper * input + transformation.Cupper,
)
region(transformation::UncertainAffineRegion) = transformation.region
inputdim(transformation::UncertainAffineRegion) = size(transformation.Alower, 2)
outputdim(transformation::UncertainAffineRegion) = size(transformation.Alower, 1)

"""
    UncertainPWAAdditiveNoiseDynamics

A struct representing uncertain PWA dynamics with additive noise.
That is, ``x_{k+1} = A_{iu}(\\alpha) x_k + C_{iu}(\\alpha) + w_k``, where ``x_k \\in X_i`` is the state, 
``A_{iu}(\\alpha) = \\alpha \\underline{A}_{iu} + (1 - \\alpha) \\overline{A}_{iu}`` and 
``C_{iu}(\\alpha) = \\alpha \\underline{C}_{iu} + (1 - \\alpha) \\overline{C}_{iu}`` with ``\\alpha \\in [0, 1]`` is the dynamics
for region ``X_i`` under control action ``u``, and ``w_k \\sim p_w`` is the additive noise where ``p_w`` is multivariate probability distribution.

### Fields
- `dimstate::Int`: The dimension of the state space.
- `dynregions::Vector{Vector{<:UncertainAffineRegion}`: A list (action) of lists (regions) of [`UncertainAffineRegion`](@ref) to represent the uncertain PWA dynamics.
- `w::AdditiveNoiseStructure`: The additive noise.

"""
struct UncertainPWAAdditiveNoiseDynamics{
    TU<:UncertainAffineRegion,
    TW<:AdditiveNoiseStructure,
} <: AdditiveNoiseDynamics
    dimstate::Int
    dynregions::Vector{Vector{TU}}
    w::TW

    function UncertainPWAAdditiveNoiseDynamics(
        dimstate,
        dynregions::Vector{Vector{TU}},
        w::TW,
    ) where {TU<:UncertainAffineRegion,TW<:AdditiveNoiseStructure}
        if dim(w) != dimstate
            throw(
                DimensionMismatch(
                    "The dimension of the noise must be the same as the dimension of the state",
                ),
            )
        end

        for action in dynregions
            for dynregion in action
                if inputdim(dynregion) != dimstate
                    throw(
                        DimensionMismatch(
                            "The dimension of the dynamics must be the same as the dimension of the input plus the dimension of the state",
                        ),
                    )
                end

                if outputdim(dynregion) != dimstate
                    throw(
                        DimensionMismatch(
                            "The dimension of the dynamics must be the same as the dimension of the noise",
                        ),
                    )
                end
            end
        end

        return new{TU,TW}(dimstate, dynregions, w)
    end
end
dimstate(dyn::UncertainPWAAdditiveNoiseDynamics) = dyn.dimstate
diminput(dyn::UncertainPWAAdditiveNoiseDynamics) = 1
noise(dyn::UncertainPWAAdditiveNoiseDynamics) = dyn.w
function nominal(dyn::UncertainPWAAdditiveNoiseDynamics, X::LazySet, a::Integer)
    # Subtract epsilon from set to avoid numerical issues
    eps_ball = BallInf(zeros(LazySets.dim(X)), 1e-6)
    Xquery = minkowski_difference(X, eps_ball)

    for dynregion in dyn.dynregions[a]
        if issubset(Xquery, region(dynregion))
            return overapproximate(dynregion, X)
        end
    end

    throw(ArgumentError("The state is not in the domain of the dynamics"))
end
function nominal(dyn::UncertainPWAAdditiveNoiseDynamics, x::AbstractVector, a::Integer)
    for dynregion in dyn.dynregions[a]
        if x ∈ region(dynregion)
            return overapproximate(dynregion, Singleton(x))
        end
    end

    throw(ArgumentError("The state is not in the domain of the dynamics"))
end
prepare_nominal(::UncertainPWAAdditiveNoiseDynamics, input_abstraction) = nothing
