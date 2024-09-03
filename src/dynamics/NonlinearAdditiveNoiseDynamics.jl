
export NonlinearAdditiveNoiseDynamics

"""
    NonlinearAdditiveNoiseDynamics

A struct representing dynamics with additive Gaussian noise.
I.e. `x_{k+1} = A x_k + B u_k + w_k`, where `w_k ~ N(0, diag(w_stddev))`.

### Fields
- `A::AbstractMatrix{Float64}`: The state transition matrix.
- `B::AbstractMatrix{Float64}`: The control input matrix.
- `w::AdditiveNoiseStructure`: The additive noise.

### Examples

```julia

# Stochastic Van der Pol Oscillator with additive uniform noise, but no inputs.
τ = 0.1
f(x, u) = [x[1] + x[2] * τ; x[2] + (-x[1] + (1 - x[1])^2 * x[2]) * τ]

w_stddev = [0.1, 0.1]
w = AdditiveDiagonalUniformNoise(w_stddev)

dyn = NonlinearAdditiveNoiseDynamics(A, B, w)
```

"""
struct NonlinearAdditiveNoiseDynamics{F<:Function, TW<:AdditiveNoiseStructure} <: AdditiveNoiseDynamics
    f::F
    nstate::Int
    ninput::Int
    w::TW

    function NonlinearAdditiveNoiseDynamics(f::F, nstate, ninput, w::TW) where {F<:Function, TW<:AdditiveNoiseStructure}
        if nstate != dim(w)
            throw(ArgumentError("The dimensionality of w must match the state dimension"))
        end
        
        return new{F, TW}(f, nstate, ninput, w)
    end
end

nominal(dyn::NonlinearAdditiveNoiseDynamics, X::LazySet, U::LazySet) = dyn.f(X, U)
noise(dyn::NonlinearAdditiveNoiseDynamics) = dyn.w
dimstate(dyn::NonlinearAdditiveNoiseDynamics) = dyn.nstate
diminput(dyn::NonlinearAdditiveNoiseDynamics) = dyn.ninput