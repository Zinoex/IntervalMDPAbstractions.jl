
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

!!! warning
    The `nominal_dynamics` is _not_ thread-safe. This is because the TaylorSeries.jl package modifies its global state.

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
struct NonlinearAdditiveNoiseDynamics{F<:Function,TW<:AdditiveNoiseStructure} <:
       AdditiveNoiseDynamics
    f::F
    nstate::Int
    ninput::Int
    w::TW

    function NonlinearAdditiveNoiseDynamics(
        f::F,
        nstate,
        ninput,
        w::TW,
    ) where {F<:Function,TW<:AdditiveNoiseStructure}
        if nstate != dim(w)
            throw(ArgumentError("The dimensionality of w must match the state dimension"))
        end

        return new{F,TW}(f, nstate, ninput, w)
    end
end

function nominal(
    dyn::NonlinearAdditiveNoiseDynamics,
    X::Hyperrectangle{Float64},
    U::Hyperrectangle{Float64},
)
    # Use the Taylor model to over-approximate the reachable set
    order = 1

    x0 = center(X)
    u0 = center(U)
    z0 = [x0; u0]
    dom = IntervalBox([low(X); low(U)], [high(X); high(U)])

    # TaylorSeries.jl modifieds the global state - eww...
    # Therefore, we prepare the global state before entering the threaded section.
    # set_variables(Float64, "z"; order=order, numvars=dimstate(dyn) + diminput(dyn))

    z = [
        TaylorModelN(i, order, IntervalBox(z0), dom) for i = 1:dimstate(dyn)+diminput(dyn)
    ]
    x, u = (z-z0)[1:dimstate(dyn)], z[dimstate(dyn)+1:end]

    # Perform the Taylor expansion
    y = dyn.f(x, u)

    # Extract the linear and constant terms + the remainder
    C = [constant_term(y[i]) for i = 1:dimstate(dyn)]
    Clow = inf.(C)
    Cupper = sup.(C)

    AB = [
        linear_polynomial(y[i])[1][j] for i = 1:dimstate(dyn),
        j = 1:dimstate(dyn)+diminput(dyn)
    ]

    A = AB[:, 1:dimstate(dyn)]
    Alow = inf.(A)
    Aupper = sup.(A)

    B = AB[:, dimstate(dyn)+1:end]
    Blow = inf.(B)
    Bupper = sup.(B)

    D = [remainder(y[i]) for i = 1:dimstate(dyn)]
    Dlower = inf.(D)
    Dupper = sup.(D)

    Y1 = Alow * Translation(X, -x0) + Blow * Translation(U, -u0) + Clow + Dlower
    Y2 = Aupper * Translation(X, -x0) + Bupper * Translation(U, -u0) + Cupper + Dupper

    Yconv = ConvexHull(Y1, Y2)

    return Yconv
end

nominal(
    dyn::NonlinearAdditiveNoiseDynamics,
    X::Hyperrectangle{Float64},
    U::Singleton{Float64},
) = nominal(dyn, X, element(U))

function nominal(
    dyn::NonlinearAdditiveNoiseDynamics,
    X::Hyperrectangle{Float64},
    u::AbstractVector{Float64},
)
    # Use the Taylor model to over-approximate the reachable set

    x0 = center(X)
    dom = IntervalBox(low(X), high(X))

    # TaylorSeries.jl modifieds the global state - eww...
    # It also means that this function is not thread-safe!!

    # We set 10 as the maximum order of the Taylor expansion
    # set_variables(Float64, "x"; order=10, numvars=dimstate(dyn))

    order = 1
    x = [TaylorModelN(i, order, IntervalBox(x0), dom) for i = 1:dimstate(dyn)]

    # Perform the Taylor expansion
    y = dyn.f(x, u)

    # Extract the linear and constant terms + the remainder
    C = [constant_term(y[i]) for i = 1:dimstate(dyn)]
    Clow = inf.(C)
    Cupper = sup.(C)

    A = [linear_polynomial(y[i])[1][j] for i = 1:dimstate(dyn), j = 1:dimstate(dyn)]
    Alow = inf.(A)
    Aupper = sup.(A)

    D = [remainder(y[i]) for i = 1:dimstate(dyn)]
    Dlower = inf.(D)
    Dupper = sup.(D)

    Y1 = Alow * Translation(X, -x0) + Clow + Dlower
    Y2 = Aupper * Translation(X, -x0) + Cupper + Dupper

    Yconv = ConvexHull(Y1, Y2)

    return Yconv
end

function nominal(
    dyn::NonlinearAdditiveNoiseDynamics,
    X::Singleton{Float64},
    U::Singleton{Float64},
)
    x = element(X)
    u = element(U)

    y = dyn.f(x, u)

    return Singleton(y)
end


nominal(
    dyn::NonlinearAdditiveNoiseDynamics,
    x::AbstractVector{Float64},
    u::AbstractVector{Float64},
) = dyn.f(x, u)

noise(dyn::NonlinearAdditiveNoiseDynamics) = dyn.w
dimstate(dyn::NonlinearAdditiveNoiseDynamics) = dyn.nstate
diminput(dyn::NonlinearAdditiveNoiseDynamics) = dyn.ninput

function prepare_nominal(dyn::NonlinearAdditiveNoiseDynamics, input_abstraction)
    n = dimstate(dyn)
    if issetbased(input_abstraction)
        m = diminput(dyn)
        n += m
    end

    # Set the Taylor model variables
    set_variables(Float64, "z"; order = 2, numvars = n)

    return nothing
end
