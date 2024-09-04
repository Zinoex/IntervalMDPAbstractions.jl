
export NonlinearAdditiveNoiseDynamics

"""
    NonlinearAdditiveNoiseDynamics

A struct representing dynamics with additive noise.
I.e. `x_{k+1} = f(x_k, u_k) + w_k`, where `w_k ~ p_w` and `p_w` is multivariate probability distribution.

!!! note
    The nominal dynamics of this class are _assumed_ to be infinitely differentiable, i.e. 
    the Taylor expansion of the dynamics function `f` is well-defined. This is because to over-approximate
    the one-step reachable set, we rely on Taylor models, which are Taylor expansions + a remainder term.
    If you are dealing wit a non-differentiable dynamics function, consider using `PiecewiseNonlinearAdditiveNoiseDynamics` instead.
    The one-step reachable set of `PiecewiseNonlinearAdditiveNoiseDynamics` is over-approximated using
    Linear Bound Propagation.

### Fields
- `f::Function`: A function taking `x::Vector` and `u::Vector` as input and returns a `Vector` of the dynamics output.
- `nstate::Int`: The state dimension.
- `ninput::Int`: The input dimension.
- `w::AdditiveNoiseStructure`: The additive noise.

### Examples

```julia

# Stochastic Van der Pol Oscillator with additive uniform noise, but no inputs.
τ = 0.1
f(x, u) = [x[1] + x[2] * τ, x[2] + (-x[1] + (1 - x[1])^2 * x[2]) * τ]

w_stddev = [0.1, 0.1]
w = AdditiveDiagonalUniformNoise(w_stddev)

dyn = NonlinearAdditiveNoiseDynamics(f, 2, 0, w)
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