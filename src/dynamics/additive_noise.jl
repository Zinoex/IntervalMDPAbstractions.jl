export AdditiveNoiseDynamics, nominal, prepare_nominal, noise

"""
    AdditiveNoiseDynamics

Dynamics with additive noise, i.e. ``x_{k+1} = f(x_k, u_k) + w_k`` with some i.i.d. noise structure ``w_k``.
"""
abstract type AdditiveNoiseDynamics <: DiscreteTimeStochasticDynamics end

"""
    nominal

Compute the reachable set under the nominal dynamics of the dynamics `dyn` over the state region `X` and control `u`.
The nominal dynamics are given by ``\\hat{x}_{k+1} = f(x_k, u_k)``, which implies that nominal dynamics are only well-defined for additive noise dynamics.
Since for some non-linear dynamics, no analytical formula exists for the one-step reachable set under the nominal dynamics,
the function `nominal` only guarantees that the returned set is a superset of the one-step true reachable set, i.e. an over-approximation.

Note that for [`NonlinearAdditiveNoiseDynamics`](@ref), you must first call [`prepare_nominal`](@ref) before calling `nominal`.
If the method is called with a single state `x::Vector{<:Real}`, it is not necessary to call `prepare_nominal` first.

"""
function nominal end

"""
    prepare_nominal

The need for this method is a result of using `TaylorModels.jl` for the over-approximation of the reachable set under non-linear nominal dynamics.
This package relies on global variables to store the variables of the Taylor models, which can be problematic when using multi-threading.
Furthermore, when setting the variables, it invalidates existing Taylor models. Therefore, before entering the loop of [`abstraction`](@ref) to compute
the transition probability bounds for all regions, we call this method to set up the global state appropriately.

This method is a no-op for dynamics that are not [`NonlinearAdditiveNoiseDynamics`](@ref).

Eventually, we hope to remove the need for this method by using a more thread-safe library for Taylor models, e.g. akin to `MultivariatePolynomials.jl`.
"""
function prepare_nominal end

"""
    noise

For additive dynamics ``x_{k+1} = f(x_k, u_k) + w_k``, return ``w_k`` as a struct. See [`AdditiveNoiseStructure`](@ref) for implementations.
"""
function noise end

#### Noise structures
export AdditiveNoiseStructure, AdditiveDiagonalGaussianNoise, AdditiveCentralUniformNoise

"""
    AdditiveNoiseStructure

Structure to represent the noise of additive noise dynamics. See [`AdditiveDiagonalGaussianNoise`](@ref) and [`AdditiveCentralUniformNoise`](@ref) for concrete types.
"""
abstract type AdditiveNoiseStructure end

"""
    AdditiveDiagonalGaussianNoise

Additive diagonal Gaussian noise structure with zero mean, i.e. ``w_k \\sim \\mathcal{N}(0, \\mathrm{diag}(\\sigma))``.
Zero mean is without loss of generality, since the mean can be absorbed into the nominal dynamics.
"""
struct AdditiveDiagonalGaussianNoise <: AdditiveNoiseStructure
    w_stddev::Vector{Float64}

    function AdditiveDiagonalGaussianNoise(w_stddev::Vector{Float64})
        if any(w_stddev .< 0.0)
            throw(ArgumentError("Standard deviation must be non-negative"))
        end
        return new(w_stddev)
    end
end
stddev(w::AdditiveDiagonalGaussianNoise) = w.w_stddev
stddev(w::AdditiveDiagonalGaussianNoise, i) = w.w_stddev[i]
dim(w::AdditiveDiagonalGaussianNoise) = length(w.w_stddev)
candecouple(w::AdditiveDiagonalGaussianNoise) = true

function transition_prob_bounds(Y, Z::Hyperrectangle, w::AdditiveDiagonalGaussianNoise)
    # Use the box approximation for the transition probability bounds, as 
    # that makes the computation of the bounds more efficient (altought slightly more conservative).

    Y = box_approximation(Y)

    pl = 1.0
    pu = 1.0

    for i = 1:LazySets.dim(Y)
        axis_pl, axis_pu = axis_transition_prob_bounds(Y, Z, w, i)
        pl *= axis_pl
        pu *= axis_pu
    end

    return pl, pu
end

function axis_transition_prob_bounds(
    Y::Hyperrectangle,
    Z::Hyperrectangle,
    w::AdditiveDiagonalGaussianNoise,
    axis::Int,
)
    z = Interval(low(Z, axis), high(Z, axis))

    return axis_transition_prob_bounds(Y, z, w, axis)
end

function axis_transition_prob_bounds(
    Y::Hyperrectangle,
    z::Interval,
    w::AdditiveDiagonalGaussianNoise,
    axis::Int,
)
    y = Interval(low(Y, axis), high(Y, axis))
    σ = stddev(w, axis)

    return axis_transition_prob_bounds(y, z, w, σ)
end

function axis_transition_prob_bounds(
    y::Interval,
    z::Interval,
    w::AdditiveDiagonalGaussianNoise,
    σ::Real,
)
    # Compute the transition probability bounds for each dimension
    cy, cz = center(y, 1), center(z, 1)

    min_point = ifelse(cz ≥ cy, low(y, 1), high(y, 1))
    pl = gaussian_transition(min_point, low(z, 1), high(z, 1), σ)

    max_point = min(high(y, 1), max(cz, low(y, 1)))
    pu = gaussian_transition(max_point, low(z, 1), high(z, 1), σ)

    # Just in case the numerical computation is slightly off
    return max(pl, 0.0), min(pu, 1.0)
end

# Use two parameter erf function for higher numerical precision
function gaussian_transition(v, l, h, σ)
    a = (v - l) * invsqrt2 / σ
    b = (v - h) * invsqrt2 / σ
    return 0.5 * erf(b, a)
end

"""
    AdditiveCentralUniformNoise

Additive diagonal uniform noise structure, i.e. ``w_k \\sim \\mathcal{U}(-r, r)``. 
This is without loss of generality, since the mean can be absorbed into the nominal dynamics.
That is, ``w_k \\sim \\mathcal{U}(a, b)`` is equivalent to ``c + w_k`` where ``w_k \\sim \\mathcal{U}(-r, r)`` with ``c = (a + b) / 2``
and ``r = (b - a) / 2``, such that ``c`` can be absorbed into the nominal dynamics.
"""
struct AdditiveCentralUniformNoise <: AdditiveNoiseStructure
    r::Vector{Float64}

    function AdditiveCentralUniformNoise(r::Vector{Float64})
        if any(r .< 0.0)
            throw(ArgumentError("Half-width must be non-negative"))
        end
        return new(r)
    end
end
dim(w::AdditiveCentralUniformNoise) = length(w.r)
candecouple(w::AdditiveCentralUniformNoise) = true

function transition_prob_bounds(
    Y,
    Z::Hyperrectangle,
    w::AdditiveCentralUniformNoise,
)
    # Use the box approximation for the transition probability bounds, as 
    # that makes the computation of the bounds more efficient (altought slightly more conservative).

    Y = box_approximation(Y)

    pl = 1.0
    pu = 1.0

    for i = 1:LazySets.dim(Y)
        axis_pl, axis_pu = axis_transition_prob_bounds(Y, Z, w, i)
        pl *= axis_pl
        pu *= axis_pu
    end

    return pl, pu
end

function axis_transition_prob_bounds(
    Y::Hyperrectangle,
    Z::Hyperrectangle,
    w::AdditiveCentralUniformNoise,
    axis::Int,
)
    z = Interval(extrema(Z, axis)...)

    return axis_transition_prob_bounds(Y, z, w, axis)
end

function axis_transition_prob_bounds(
    Y::Hyperrectangle,
    z::Interval,
    w::AdditiveCentralUniformNoise,
    axis::Int,
)
    y = Interval(extrema(Y, axis)...)
    r = w.r[axis]

    return axis_transition_prob_bounds(y, z, w, r)
end

function axis_transition_prob_bounds(
    y::Interval,
    z::Interval,
    w::AdditiveCentralUniformNoise,
    r::Real,
)
    # Compute the transition probability bounds for each dimension
    cy, cz = center(y, 1), center(z, 1)

    min_point = ifelse(cz ≥ cy, low(y, 1), high(y, 1))
    pl = uniform_transition(min_point, low(z, 1), high(z, 1), r)

    max_point = min(high(y, 1), max(cz, low(y, 1)))
    pu = uniform_transition(max_point, low(z, 1), high(z, 1), r)

    # Just in case the numerical computation is slightly off
    return max(pl, 0.0), min(pu, 1.0)
end

function uniform_transition(v, l, h, r)
    a, b = v - r, v + r

    if a ≥ h || b ≤ l
        return 0.0
    end

    l = max(l, a)
    h = min(h, b)

    return (h - l) / (b - a)
end
