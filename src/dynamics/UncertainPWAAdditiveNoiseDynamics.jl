export UncertainAffineRegion, UncertainPWAAdditiveNoiseDynamics

"""
    UncertainAffineRegion

A struct representing an uncertain affine transition, valid over a region.
I.e. `A(\\alpha) x + C(\\alpha)` for `\\alpha in [0, 1]`, valid over a region `X`.

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
I.e. `x_{k+1} = A_i(\\alpha) x_k + B_i(\\alpha) u_k + C_i(\\alpha) + w_k`, where `x_k \\in X_i` `w_k ~ p_w` and `p_w` is multivariate probability distribution.

### Fields
- `dimstate::Int`: The dimension of the state space.
- `dyn::Vector{Vector{<:UncertainAffineRegion}`: A list (action) of lists (regions) of UncertainAffineRegions to represent an uncertain PWA dynamics.
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
            return overapproximate(dynregion, Singleton(X))
        end
    end

    throw(ArgumentError("The state is not in the domain of the dynamics"))
end
prepare_nominal(::UncertainPWAAdditiveNoiseDynamics, input_abstraction) = nothing
