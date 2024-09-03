
export LinearAdditiveNoiseDynamics

"""
    LinearAdditiveNoiseDynamics

A struct representing dynamics with additive Gaussian noise.
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

dyn = LinearAdditiveNoiseDynamics(A, B, w)
```

"""
struct LinearAdditiveNoiseDynamics{TA<:AbstractMatrix{Float64}, TB<:AbstractMatrix{Float64}, TW<:AdditiveNoiseStructure} <: AdditiveNoiseDynamics
    A::TA
    B::TB
    w::TW

    function LinearAdditiveNoiseDynamics(A::TA, B::TB, w::TW) where {TA<:AbstractMatrix{Float64}, TB<:AbstractMatrix{Float64}, TW<:AdditiveNoiseStructure}
        n = LinearAlgebra.checksquare(A)
        
        if n != size(B, 1)
            throw(ArgumentError("The number of rows of B must match the state dimension"))
        end

        if n != dim(w)
            throw(ArgumentError("The dimensionality of w must match the dynamics dimension"))
        end
        
        return new{TA, TB, TW}(A, B, w)
    end
end

nominal(dyn::LinearAdditiveNoiseDynamics, X::LazySet, U::LazySet) = dyn.A * X + dyn.B * U
noise(dyn::LinearAdditiveNoiseDynamics) = dyn.w
dimstate(dyn::LinearAdditiveNoiseDynamics) = size(dyn.A, 1)
diminput(dyn::LinearAdditiveNoiseDynamics) = size(dyn.B, 2)