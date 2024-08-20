
export AffineAdditiveGaussian

"""
    AffineAdditiveGaussian

A struct representing an system with additive Gaussian noise.
I.e. `x_{k+1} = A x_k + B u_k + w_k`, where `w_k ~ N(0, diag(w_stddev))`.

### Fields
- `A::Matrix{Float64}`: The state transition matrix.
- `B::Matrix{Float64}`: The control input matrix.
- `w_stddev::Vector{Float64}`: The standard deviation of the Gaussian noise.

### Examples

```julia

A = [1.0 0.1; 0.0 1.0]
B = [0.0; 1.0]

w_stddev = [0.1, 0.1]

sys = AffineAdditiveGaussian(A, B, w_stddev)
```

"""
struct AffineAdditiveGaussian{TA<:AbstractMatrix{Float64}, TB<:AbstractMatrix{Float64}, TW<:AbstractVector{Float64}}
    A::TA
    B::TB
    w_stddev::TW

    function AffineAdditiveGaussian(A::TA, B::TB, w_stddev::TW) where {TA<:AbstractMatrix{Float64}, TB<:AbstractMatrix{Float64}, TW<:AbstractVector{Float64}}
        n = LinearAlgebra.checksquare(A)
        
        if n != size(B, 1)
            throw(ArgumentError("The number of rows of B must match the system dimension"))
        end

        if n != length(w_stddev)
            throw(ArgumentError("The length of w_stddev must match the system dimension"))
        end
        
        return new{TA, TB, TW}(A, B, w_stddev)
    end
end

nominal_dynamics(sys::AffineAdditiveGaussian, X::LazySet, U::LazySet) = sys.A * X + sys.B * U