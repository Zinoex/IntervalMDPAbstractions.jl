export AdditiveNoiseDynamics, nominal, noise

"""
    AdditiveNoiseDynamics

Dynamics with additive noise, i.e. `x_{k+1} = f(x_k, u_k) + w_k`.
"""
abstract type AdditiveNoiseDynamics <: DiscreteTimeStochasticDynamics end
dimnoise(dyn::AdditiveNoiseDynamics) = dimstate(dyn)

"""
nominal(dyn::AdditiveNoiseDynamics, X::LazySet, U::LazySet)

Compute the reachable set under the nominal dynamics of the dynamics `dyn` over the state region `X` and control region `U`.
The nominal dynamics are given by `x_{k+1} = f(x_k, u_k)`, i.e. only well-defined for dynamicss with additive noise.
Since for some non-linear dynamics, no analytical formula exists for the one-step reachable set under the nominal dynamics,
the function `nominal` only guarantees that the returned set is a superset of the one-step true reachable set, i.e. an over-approximation.
"""
function nominal end
function noise end

#### Noise structures
export AdditiveNoiseStructure, AdditiveDiagonalGaussianNoise

"""
    AdditiveNoiseStructure

Structure to represent the noise of additive noise dynamics.
"""
abstract type AdditiveNoiseStructure end

"""
    AdditiveDiagonalGaussianNoise

Additive diagonal Gaussian noise structure with zero mean, i.e. `w_k ~ N(0, diag(w_stddev))`.
Zero mean is without loss of generality, since the mean can be absorbed into the nominal dynamics.
"""
struct AdditiveDiagonalGaussianNoise <: AdditiveNoiseStructure
    w_stddev::Vector{Float64}
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

    for (ly, hy, lz, hz, σ) in zip(low(Y), high(Y), low(Z), high(Z), stddev(w))
        axis_pl, axis_pu = axis_transition_prob_bounds(Interval(ly, hy), Interval(lz, hz), w, σ)
        pl *= axis_pl
        pu *= axis_pu
    end

    return pl, pu
end

function axis_transition_prob_bounds(Y::Hyperrectangle, Z::Hyperrectangle, w::AdditiveDiagonalGaussianNoise, axis::Int)
    z = Interval(extrema(Z, axis)...)

    return axis_transition_prob_bounds(Y, z, w, axis)
end

function axis_transition_prob_bounds(Y::Hyperrectangle, z::Interval, w::AdditiveDiagonalGaussianNoise, axis::Int)
    y = Interval(extrema(Y, axis)...)
    σ = stddev(w, axis)

    return axis_transition_prob_bounds(y, z, w, σ)
end

function axis_transition_prob_bounds(y::Interval, z::Interval, w::AdditiveDiagonalGaussianNoise, σ::Real)
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