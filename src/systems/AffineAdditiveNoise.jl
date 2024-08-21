
export AffineAdditiveNoise

"""
    AffineAdditiveNoise

A struct representing an system with additive Gaussian noise.
I.e. `x_{k+1} = A x_k + B u_k + w_k`, where `w_k ~ N(0, diag(w_stddev))`.

### Fields
- `A::AbstractMatrix{Float64}`: The state transition matrix.
- `B::AbstractMatrix{Float64}`: The control input matrix.
- `w::AdditiveNoiseStructure`: The additive noise.

### Examples

```julia

A = [1.0 0.1; 0.0 1.0]
B = [0.0; 1.0]

w_stddev = [0.1, 0.1]
w = AdditiveDiagonalGaussianNoise(w_stddev)

sys = AffineAdditiveNoise(A, B, w)
```

"""
struct AffineAdditiveNoise{TA<:AbstractMatrix{Float64}, TB<:AbstractMatrix{Float64}, TW<:AdditiveNoiseStructure}
    A::TA
    B::TB
    w::TW

    function AffineAdditiveNoise(A::TA, B::TB, w::TW) where {TA<:AbstractMatrix{Float64}, TB<:AbstractMatrix{Float64}, TW<:AdditiveNoiseStructure}
        n = LinearAlgebra.checksquare(A)
        
        if n != size(B, 1)
            throw(ArgumentError("The number of rows of B must match the system dimension"))
        end

        if n != dim(w)
            throw(ArgumentError("The dimensionality of w must match the system dimension"))
        end
        
        return new{TA, TB, TW}(A, B, w)
    end
end

nominaldynamics(sys::AffineAdditiveNoise, X::LazySet, U::LazySet) = sys.A * X + sys.B * U
noise(sys::AffineAdditiveNoise) = sys.w
dimstate(sys::AffineAdditiveNoise) = size(sys.A, 1)
diminput(sys::AffineAdditiveNoise) = size(sys.B, 2)