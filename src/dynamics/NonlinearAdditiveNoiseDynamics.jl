
export NonlinearAdditiveNoiseDynamics

"""
    NonlinearAdditiveNoiseDynamics

A struct representing continuous non-linear dynamics with additive noise.
That is, ``x_{k+1} = f(x_k, u_k) + w_k``, where ``f(\\cdot, u_k)`` is continuously differentiable function for each ``u_k \\in U`` and
``w_k \\sim p_w`` and ``p_w`` is multivariate probability distribution.

!!! note
    The nominal dynamics of this class are _assumed_ to be infinitely differentiable, i.e. 
    the Taylor expansion of the dynamics function `f` is well-defined. This is because to over-approximate
    the one-step reachable set, we rely on Taylor models, which are Taylor expansions + a remainder term.
    If you are dealing wit a non-differentiable dynamics function, consider using [`UncertainPWAAdditiveNoiseDynamics`](@ref) instead.
    To obtain an `UncertainPWAAdditiveNoiseDynamics`, you can partitoned the state space and use Linear Bound Propagation
    with each region (see [bound_propagation](https://github.com/Zinoex/bound_propagation)).

!!! warning
    Before calling [`nominal`](@ref) with a `LazySet` as input, you must call [`prepare_nominal`](@ref). 
    This is because the `TaylorSeries.jl` package modifies its global state. If you are using multi-threading,
    [`prepare_nominal`](@ref) must be called before entering the threaded section.

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
w = AdditiveCentralUniformNoise(w_stddev)

dyn = NonlinearAdditiveNoiseDynamics(f, 2, 0, w)
```

"""
struct NonlinearAdditiveNoiseDynamics{F <: Function, TW <: AdditiveNoiseStructure} <:
       AdditiveNoiseDynamics
    f::F
    nstate::Int
    ninput::Int
    w::TW

    function NonlinearAdditiveNoiseDynamics(
            f::F,
            nstate,
            ninput,
            w::TW
    ) where {F <: Function, TW <: AdditiveNoiseStructure}
        if nstate != dim(w)
            throw(ArgumentError("The dimensionality of w must match the state dimension"))
        end

        return new{F, TW}(f, nstate, ninput, w)
    end
end

function nominal(
        dyn::NonlinearAdditiveNoiseDynamics,
        X::Hyperrectangle{Float64},
        U::Hyperrectangle{Float64}
)
    # Use the Taylor model to over-approximate the reachable set
    order = 1

    Z = CartesianProduct(X, U)
    z0 = center(Z)
    dom = IntervalBox(low(Z), high(Z))

    # TaylorSeries.jl modifieds the global state - eww...
    # Therefore, we prepare the global state before entering the threaded section.
    # set_variables(Float64, "z"; order=order, numvars=dimstate(dyn) + diminput(dyn))

    z = [TaylorModelN(i, order, IntervalBox(z0), dom) for i in 1:(dimstate(dyn) + diminput(dyn))]
    x, u = z[1:dimstate(dyn)], z[(dimstate(dyn) + 1):end]

    # Perform the Taylor expansion
    y = dyn.f(x, u)

    # Extract the linear and constant terms + the remainder
    C = [yi[0][1] for yi in y]
    Clower = inf.(C)
    Cupper = sup.(C)

    AB = transpose([yi[1][:] for yi in y])
    AB = reduce(vcat, AB)

    ABlower = inf.(AB)
    ABupper = sup.(AB)

    D = remainder.(y)
    Dlower = inf.(D)
    Dupper = sup.(D)

    Y1 = AffineMap(ABlower, Translation(Z, -z0), Clower)
    Y2 = AffineMap(ABupper, Translation(Z, -z0), Cupper)

    Yconv = ConvexHull(Y1, Y2) + Hyperrectangle(; low=Dlower, high=Dupper)

    return Yconv
end

function nominal(
        dyn::NonlinearAdditiveNoiseDynamics,
        X::Hyperrectangle{Float64},
        U::Singleton{Float64}
)
    nominal(dyn, X, element(U))
end

function nominal(
        dyn::NonlinearAdditiveNoiseDynamics,
        X::Hyperrectangle{Float64},
        u::AbstractVector{Float64}
)
    # Use the Taylor model to over-approximate the reachable set

    x0 = center(X)
    dom = IntervalBox(low(X), high(X))

    # TaylorSeries.jl modifieds the global state - eww...
    # It also means that this function is not thread-safe!!

    # We set 10 as the maximum order of the Taylor expansion
    # set_variables(Float64, "x"; order=10, numvars=dimstate(dyn))

    order = 1
    x = [TaylorModelN(i, order, IntervalBox(x0), dom) for i in 1:dimstate(dyn)]

    # Perform the Taylor expansion
    y = dyn.f(x, u)

    # Extract the linear and constant terms + the remainder
    C = [yi[0][1] for yi in y]
    Clower = inf.(C)
    Cupper = sup.(C)

    A = transpose([yi[1][:] for yi in y])
    A = reduce(vcat, A)
    Alower = inf.(A)
    Aupper = sup.(A)

    D = remainder.(y)
    Dlower = inf.(D)
    Dupper = sup.(D)

    Y1 = AffineMap(Alower, Translation(X, -x0), Clower)
    Y2 = AffineMap(Aupper, Translation(X, -x0), Cupper)

    Yconv = ConvexHull(Y1, Y2) + Hyperrectangle(; low=Dlower, high=Dupper)

    return Yconv
end

function nominal(
        dyn::NonlinearAdditiveNoiseDynamics,
        X::Singleton{Float64},
        U::Singleton{Float64}
)
    x = element(X)
    u = element(U)

    y = dyn.f(x, u)

    return Singleton(y)
end

nominal(
dyn::NonlinearAdditiveNoiseDynamics,
x::AbstractVector{Float64},
u::AbstractVector{Float64}
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
    set_variables(Float64, "z"; order=2, numvars=n)

    return nothing
end

function transform(
        dyn::NonlinearAdditiveNoiseDynamics,
        transformation::LinearTransformation,
        w::AdditiveNoiseStructure  # Noise is already transformed
)
    # Transform the dynamics
    function f(z, u)
        x = transformation.Tinv * z
        y = dyn.f(x, u)
        return transformation.T * y
    end

    return NonlinearAdditiveNoiseDynamics(f, dimstate(dyn), diminput(dyn), w)
end
