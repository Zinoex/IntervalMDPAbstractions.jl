
export AffineAdditiveNoiseDynamics

"""
    AffineAdditiveNoiseDynamics

A struct representing dynamics with additive noise.
That is, ``x_{k+1} = A x_k + B u_k + C + w_k``, where ``w_k ~ p_w`` and ``p_w`` is multivariate probability distribution.

### Fields
- `A::AbstractMatrix{Float64}`: The linear state matrix.
- `B::AbstractMatrix{Float64}`: The control input matrix.
- `C::AbstractMatrix{Float64}`: The constant drift vector.
- `w::AdditiveNoiseStructure`: The additive noise.

### Examples

```julia
A = [1.0 0.1; 0.0 1.0]
B = [0.0; 1.0]

w_stddev = [0.1, 0.1]
w = AdditiveDiagonalGaussianNoise(w_stddev)

# If no C is provided, it is assumed to be zero.
dyn = AffineAdditiveNoiseDynamics(A, B, w)
```

"""
struct AffineAdditiveNoiseDynamics{
    TA <: AbstractMatrix{Float64},
    TB <: AbstractMatrix{Float64},
    TC <: AbstractVector{Float64},
    TW <: AdditiveNoiseStructure
} <: AdditiveNoiseDynamics
    A::TA
    B::TB
    C::TC
    w::TW

    function AffineAdditiveNoiseDynamics(
            A::TA,
            B::TB,
            C::TC,
            w::TW
    ) where {
            TA <: AbstractMatrix{Float64},
            TB <: AbstractMatrix{Float64},
            TC <: AbstractVector{Float64},
            TW <: AdditiveNoiseStructure
    }
        n = LinearAlgebra.checksquare(A)

        if n != size(B, 1)
            throw(ArgumentError("The number of rows of B must match the state dimension"))
        end

        if n != size(C, 1)
            throw(ArgumentError("The length of C must match the dynamics dimension"))
        end

        if n != dim(w)
            throw(
                ArgumentError("The dimensionality of w must match the dynamics dimension"),
            )
        end

        return new{TA, TB, TC, TW}(A, B, C, w)
    end
end

function AffineAdditiveNoiseDynamics(
        A::AbstractMatrix{Float64},
        B::AbstractMatrix{Float64},
        w::AdditiveNoiseStructure
)
    C = zeros(eltype(A), size(A, 1))
    return AffineAdditiveNoiseDynamics(A, B, C, w)
end

function nominal(dyn::AffineAdditiveNoiseDynamics, X::LazySet{Float64}, U::LazySet{Float64})
    dyn.A * X + dyn.B * U + Singleton(dyn.C)
end
function nominal(
        dyn::AffineAdditiveNoiseDynamics,
        X::AbstractVector{Float64},
        U::AbstractVector{Float64}
)
    dyn.A * X + dyn.B * U + dyn.C
end
noise(dyn::AffineAdditiveNoiseDynamics) = dyn.w
dimstate(dyn::AffineAdditiveNoiseDynamics) = size(dyn.A, 1)
diminput(dyn::AffineAdditiveNoiseDynamics) = size(dyn.B, 2)
prepare_nominal(::AffineAdditiveNoiseDynamics, input_abstraction) = nothing

function transform(
        dyn::AffineAdditiveNoiseDynamics,
        transformation::LinearTransformation,
        w::AdditiveNoiseStructure  # Noise is already transformed
)
    # Transform the dynamics
    A = transformation.T * dyn.A * transformation.Tinv
    B = transformation.T * dyn.B
    C = transformation.T * dyn.C

    return AffineAdditiveNoiseDynamics(A, B, C, w)
end
