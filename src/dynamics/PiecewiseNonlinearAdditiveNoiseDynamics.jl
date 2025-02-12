export PiecewiseNonlinearAdditiveNoiseDynamics, NonlinearDynamicsRegion

"""
    NonlinearDynamicsRegion

A struct representing a non-linear dynamics, valid over a region.
"""
struct NonlinearDynamicsRegion{F<:Function, S<:LazySet}
    f::F
    region::S
end

function (dyn::NonlinearDynamicsRegion)(X::Hyperrectangle{Float64}, U::Hyperrectangle{Float64})
    # Use the Taylor model to over-approximate the reachable set
    active_region = intersection(X, dyn.region)
    if isempty(active_region)
        throw(ArgumentError("The input region X does not intersect with the valid region of the dynamics"))
    end
    active_region_box = box_approximation(active_region)

    x0 = center(active_region_box)
    u0 = center(U)
    z0 = [x0; u0]
    dom = IntervalBox([low(active_region_box); low(U)], [high(active_region_box); high(U)])

    # TaylorSeries.jl modifieds the global state - eww...
    # Therefore, we prepare the global state before entering the threaded section.
    # set_variables(Float64, "z"; order=order, numvars=LazySets.dim(X) + LazySets.dim(U))

    order = 1
    z = [TaylorModelN(i, order, IntervalBox(z0), dom) for i = 1:LazySets.dim(X)+LazySets.dim(U)]
    x, u = (z-z0)[1:LazySets.dim(X)], z[LazySets.dim(X)+1:end]

    # Perform the Taylor expansion
    y = dyn.f(x, u)

    # Extract the linear and constant terms + the remainder
    C = [constant_term(y[i]) for i = 1:LazySets.dim(X)]
    Clow = inf.(C)
    Cupper = sup.(C)

    AB = [
        linear_polynomial(y[i])[1][j] for i = 1:LazySets.dim(X),
        j = 1:LazySets.dim(X)+LazySets.dim(U)
    ]

    A = AB[:, 1:LazySets.dim(X)]
    Alow = inf.(A)
    Aupper = sup.(A)

    B = AB[:, LazySets.dim(X)+1:end]
    Blow = inf.(B)
    Bupper = sup.(B)

    D = [remainder(y[i]) for i = 1:LazySets.dim(X)]
    Dlower = inf.(D)
    Dupper = sup.(D)

    Y1 = Alow * Translation(active_region, -x0) + Blow * Translation(U, -u0) + Clow + Dlower
    Y2 = Aupper * Translation(active_region, -x0) + Bupper * Translation(U, -u0) + Cupper + Dupper

    Yconv = ConvexHull(Y1, Y2)

    return Yconv
end


function (dyn::NonlinearDynamicsRegion)(X::Hyperrectangle{Float64}, U::Singleton{Float64})
    return dyn(X, element(U))
end

function (dyn::NonlinearDynamicsRegion)(X::Hyperrectangle{Float64}, u::AbstractVector{Float64})
    # Use the Taylor model to over-approximate the reachable set

    active_region = intersection(X, dyn.region)
    if isempty(active_region)
        throw(ArgumentError("The input region X does not intersect with the valid region of the dynamics"))
    end
    active_region_box = box_approximation(active_region)

    x0 = center(active_region_box)
    dom = IntervalBox(low(active_region_box), high(active_region_box))

    # TaylorSeries.jl modifieds the global state - eww...
    # It also means that this function is not thread-safe!!

    # We set 10 as the maximum order of the Taylor expansion
    # set_variables(Float64, "x"; order=10, numvars=LazySets.dim(X))

    order = 1
    x = [TaylorModelN(i, order, IntervalBox(x0), dom) for i = 1:LazySets.dim(X)]

    # Perform the Taylor expansion
    y = dyn.f(x, u)

    # Extract the linear and constant terms + the remainder
    C = [constant_term(y[i]) for i = 1:LazySets.dim(X)]
    Clow = inf.(C)
    Cupper = sup.(C)

    A = [linear_polynomial(y[i])[1][j] for i = 1:LazySets.dim(X), j = 1:LazySets.dim(X)]
    Alow = inf.(A)
    Aupper = sup.(A)

    D = [remainder(y[i]) for i = 1:LazySets.dim(X)]
    Dlower = inf.(D)
    Dupper = sup.(D)

    Y1 = Alow * Translation(active_region, -x0) + Clow + Dlower
    Y2 = Aupper * Translation(active_region, -x0) + Cupper + Dupper

    Yconv = ConvexHull(Y1, Y2)

    return Yconv
end

function (dyn::NonlinearDynamicsRegion)(X::Singleton{Float64}, U::Singleton{Float64})
    x = element(X)
    u = element(U)

    y = dyn.f(x, u)

    return Singleton(y)
end

function (dyn::NonlinearDynamicsRegion)(x::AbstractVector{Float64}, u::AbstractVector{Float64})
    if x ∉ dyn.region
        throw(ArgumentError("The input x does not belong to the valid region of the dynamics"))
    end

    return dyn.f(x, u)
end

"""
    PiecewiseNonlinearAdditiveNoiseDynamics

A struct representing non-linear dynamics with additive noise.
That is, ``x_{k+1} = f(x_k, u_k) + w_k``, where ``f(\\cdot, u_k)`` is piecewise continuously differentiable function for each ``u_k \\in U`` and
``w_k \\sim p_w`` and ``p_w`` is multivariate probability distribution.

!!! note
    The nominal dynamics of this class are _assumed_ to be piecewise infinitely differentiable, i.e. 
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
- `regions::Vector{<:NonlinearDynamicsRegion}`: A list of [`NonlinearDynamicsRegion`](@ref) to represent the piecewise dynamics.
- `nstate::Int`: The state dimension.
- `ninput::Int`: The input dimension.
- `w::AdditiveNoiseStructure`: The additive noise.

### Examples

```julia

τ = 0.1

region1 = Hyperrectangle(low=[-1.0, -1.0], high=[0.0, 1.0])
f(x, u) = [x[1] + x[2] * τ, x[2] + (-x[1] + (1 - x[1])^2 * x[2]) * τ]
dyn_reg1 = NonlinearDynamicsRegion(f, region1)

region2 = Hyperrectangle(low=[0.0, -1.0], high=[1.0, 1.0])
g(x, u) = [x[1] + x[2] * τ, x[2] + (-x[2] + (1 - x[2])^2 * x[1]) * τ]
dyn_reg2 = NonlinearDynamicsRegion(g, region2)

w_stddev = [0.1, 0.1]
w = AdditiveDiagonalGaussianNoise(w_stddev)

dyn = PiecewiseNonlinearAdditiveNoiseDynamics([dyn_reg1, dyn_reg2], 2, 0, w)
```

"""
struct PiecewiseNonlinearAdditiveNoiseDynamics{TW<:AdditiveNoiseStructure} <:
       AdditiveNoiseDynamics
    regions::Vector{<:NonlinearDynamicsRegion}
    nstate::Int
    ninput::Int
    w::TW

    function PiecewiseNonlinearAdditiveNoiseDynamics(
        regions::Vector{<:NonlinearDynamicsRegion},
        nstate,
        ninput,
        w::TW,
    ) where {TW<:AdditiveNoiseStructure}
        if nstate != dim(w)
            throw(ArgumentError("The dimensionality of w must match the state dimension"))
        end

        return new{TW}(regions, nstate, ninput, w)
    end
end

function nominal(
    dyn::PiecewiseNonlinearAdditiveNoiseDynamics,
    X::Hyperrectangle{Float64},
    u,
)
    reachable_set = EmptySet(dimstate(dyn))

    for region in dyn.regions
        if !iszeromeasure(region.region, X)
            reachable_set = ConvexHull(reachable_set, region(X, u))
        end
    end

    return reachable_set
end


function nominal(
    dyn::PiecewiseNonlinearAdditiveNoiseDynamics,
    x::AbstractVector{Float64},
    u,
) 
    for region in dyn.regions
        if x ∈ region.region
            return region(x, u)
        end
    end
end

noise(dyn::PiecewiseNonlinearAdditiveNoiseDynamics) = dyn.w
dimstate(dyn::PiecewiseNonlinearAdditiveNoiseDynamics) = dyn.nstate
diminput(dyn::PiecewiseNonlinearAdditiveNoiseDynamics) = dyn.ninput

function prepare_nominal(dyn::PiecewiseNonlinearAdditiveNoiseDynamics, input_abstraction)
    n = dimstate(dyn)
    if issetbased(input_abstraction)
        m = diminput(dyn)
        n += m
    end

    # Set the Taylor model variables
    set_variables(Float64, "z"; order = 2, numvars = n)

    return nothing
end