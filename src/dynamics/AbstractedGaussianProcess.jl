export AbstractedGaussianProcessRegion, AbstractedGaussianProcess

"""
    AbstractedGaussianProcessRegion

A struct representing an bounds on the mean and stddev of a Gaussian process over a region.
I.e. `\\underline{\\mu}_{s} \\leq \\mu(x) \\leq \\overline{\\mu}_{s}` and 
`\\underline{\\Sigma}_{s,ll} \\leq \\Sigma(x)_{ll} \\leq \\overline{\\Sigma}_{s,ll}`
for all `x \\in s` and each axis `l`.

### Fields
- `region::LazySet{Float64}`: The region over which the affine transition is valid.
- `mean_lower::AbstractVector{Float64}`: The linear lower bound vector.
- `mean_upper::AbstractVector{Float64}`: The constant lower bound vector.
- `stddev_lower::AbstractVector{Float64}`: The linear upper bound vector.
- `stddev_upper::AbstractVector{Float64}`: The constant upper bound vector.

"""
struct AbstractedGaussianProcessRegion{T,VT<:AbstractVector{T},S<:LazySet{T}}
    region::S

    mean_lower::VT
    mean_upper::VT

    stddev_lower::VT
    stddev_upper::VT

    function AbstractedGaussianProcessRegion(
        region::S,
        mean_lower::VT,
        mean_upper::VT,
        stddev_lower::VT,
        stddev_upper::VT,
    ) where {T,VT<:AbstractVector{T},S<:LazySet{T}}
        n = LazySets.dim(region)

        if size(mean_lower, 1) != n
            throw(
                DimensionMismatch(
                    "The number of rows in mean_lower must be equal to the dimensionality of the region",
                ),
            )
        end

        if size(mean_upper, 1) != n
            throw(
                DimensionMismatch(
                    "The number of rows in mean_upper must be equal to the dimensionality of the region",
                ),
            )
        end

        if size(stddev_lower, 1) != n
            throw(
                DimensionMismatch(
                    "The number of rows in stddev_lower must be equal to the dimensionality of the region",
                ),
            )
        end

        if size(stddev_upper, 1) != n
            throw(
                DimensionMismatch(
                    "The number of rows in stddev_upper must be equal to the dimensionality of the region",
                ),
            )
        end

        new{T,VT,S}(region, mean_lower, mean_upper, stddev_lower, stddev_upper)
    end
end
region(abstracted_region::AbstractedGaussianProcessRegion) = abstracted_region.region
outputdim(abstracted_region::AbstractedGaussianProcessRegion) =
    size(abstracted_region.mean_lower, 1)
mean_lower(abstracted_region::AbstractedGaussianProcessRegion) =
    abstracted_region.mean_lower
mean_lower(abstracted_region::AbstractedGaussianProcessRegion, i) =
    abstracted_region.mean_lower[i]
mean_upper(abstracted_region::AbstractedGaussianProcessRegion) =
    abstracted_region.mean_upper
mean_upper(abstracted_region::AbstractedGaussianProcessRegion, i) =
    abstracted_region.mean_upper[i]
mean_center(abstracted_region::AbstractedGaussianProcessRegion) =
    0.5 * (abstracted_region.mean_lower + abstracted_region.mean_upper)
mean_center(abstracted_region::AbstractedGaussianProcessRegion, i) =
    0.5 * (abstracted_region.mean_lower[i] + abstracted_region.mean_upper[i])
stddev_lower(abstracted_region::AbstractedGaussianProcessRegion) =
    abstracted_region.stddev_lower
stddev_lower(abstracted_region::AbstractedGaussianProcessRegion, i) =
    abstracted_region.stddev_lower[i]
stddev_upper(abstracted_region::AbstractedGaussianProcessRegion) =
    abstracted_region.stddev_upper
stddev_upper(abstracted_region::AbstractedGaussianProcessRegion, i) =
    abstracted_region.stddev_upper[i]

function transition_prob_bounds(
    gp_bounds::AbstractedGaussianProcessRegion,
    Z::Hyperrectangle,
)
    pl = 1.0
    pu = 1.0

    for i = 1:outputdim(gp_bounds)
        axis_pl, axis_pu = axis_transition_prob_bounds(gp_bounds, Z, i)
        pl *= axis_pl
        pu *= axis_pu
    end

    return pl, pu
end

function axis_transition_prob_bounds(
    gp_bounds::AbstractedGaussianProcessRegion,
    Z::Hyperrectangle,
    axis::Int,
)
    z = Interval(low(Z, axis), high(Z, axis))

    return axis_transition_prob_bounds(gp_bounds, z, axis)
end

function axis_transition_prob_bounds(
    gp_bounds::AbstractedGaussianProcessRegion,
    z::Interval,
    axis::Int,
)
    # Compute the transition probability bounds for each dimension
    cμ, cz = mean_center(gp_bounds, axis), center(z, 1)

    min_point = ifelse(cz ≥ cμ, mean_lower(gp_bounds, axis), mean_upper(gp_bounds, axis))
    pl = min(
        gaussian_transition(
            min_point,
            low(z, 1),
            high(z, 1),
            stddev_lower(gp_bounds, axis),
        ),
        gaussian_transition(
            min_point,
            low(z, 1),
            high(z, 1),
            stddev_upper(gp_bounds, axis),
        ),
    )

    max_point = min(mean_upper(gp_bounds, axis), max(cz, mean_lower(gp_bounds, axis)))
    pu = max(
        gaussian_transition(
            max_point,
            low(z, 1),
            high(z, 1),
            stddev_lower(gp_bounds, axis),
        ),
        gaussian_transition(
            max_point,
            low(z, 1),
            high(z, 1),
            stddev_upper(gp_bounds, axis),
        ),
    )

    # Just in case the numerical computation is slightly off
    return max(pl, 0.0), min(pu, 1.0)
end

"""
    AbstractedGaussianProcess

A struct representing a Gaussian process over a partitioned space and for each region, bounds are computed on the mean and stddev.

### Fields
- `dyn::Vector{<:AbstractedGaussianProcessRegion}`: A list (action) of lists (regions) of AbstractedGaussianProcessRegions.

"""
struct AbstractedGaussianProcess{TU<:AbstractedGaussianProcessRegion} <:
       DiscreteTimeStochasticDynamics
    dynregions::Vector{Vector{TU}}

    function AbstractedGaussianProcess(
        dynregions::Vector{Vector{TU}},
    ) where {TU<:AbstractedGaussianProcessRegion}
        if isempty(dynregions)
            throw(ArgumentError("The list of regions cannot be empty"))
        end

        dimstate = outputdim(first(first(dynregions)))
        diminput = inputdim(first(first(dynregions)))

        for action in dynregions
            for dynregion in action
                if outputdim(dynregion) != dimstate
                    throw(
                        DimensionMismatch(
                            "The dimension of the GP output must be the same for all regions",
                        ),
                    )
                end

                if inputdim(dynregion) != diminput
                    throw(
                        DimensionMismatch(
                            "The number of modes in the GP must be the same for all regions",
                        ),
                    )
                end
            end
        end

        return new{TU}(dynregions)
    end
end
dimstate(dyn::AbstractedGaussianProcess) = outputdim(first(first(dyn.dynregions)))
diminput(dyn::AbstractedGaussianProcess) = 1

function bounds(dyn::AbstractedGaussianProcess, X::LazySet, input::Int)
    # Subtract epsilon from set to avoid numerical issues
    eps_ball = BallInf(zeros(LazySets.dim(X)), 1e-6)
    Xquery = minkowski_difference(X, eps_ball)

    for dynregion in dyn.dynregions[input]
        if issubset(Xquery, region(dynregion))
            return dynregion
        end
    end

    throw(ArgumentError("The state is not in the domain of the dynamics"))
end
