function iszeromeasure(X::AbstractHyperrectangle, Y::AbstractHyperrectangle)
    return any(high(Y, i) ≤ low(X, i) || high(X, i) ≤ low(Y, i) for i = 1:LazySets.dim(X))
end

iszeromeasure(X::Complement, Y::LazySet) = issubset(Y, X.X)

iszeromeasure(X::EmptySet, Y::LazySet) = true
iszeromeasure(X::LazySet, Y::EmptySet) = true
iszeromeasure(X::EmptySet, Y::EmptySet) = true

iszeromeasure(X::AbstractPolyhedron, Y::AbstractPolyhedron) = _iszeromeasure(Y, X)

function iszeromeasure(
    C::CartesianProduct{N,<:LazySet,<:Universe},
    Z::AbstractHyperrectangle,
) where {N}
    X = first(C)
    Zp = LazySets.project(Z, 1:LazySets.dim(X))
    return iszeromeasure(X, Zp)
end

function iszeromeasure(
    Z::AbstractHyperrectangle,
    C::CartesianProduct{N,<:LazySet,<:Universe},
) where {N}
    X = first(C)
    Zp = LazySets.project(Z, 1:LazySets.dim(X))
    return iszeromeasure(X, Zp)
end

function iszeromeasure(
    C::CartesianProduct{N,<:Universe,<:LazySet},
    Z::AbstractHyperrectangle,
) where {N}
    Y = second(C)
    Zp = LazySets.project(Z, (LazySets.dim(first(C))+1):LazySets.dim(C))
    return iszeromeasure(Y, Zp)
end

function iszeromeasure(
    Z::AbstractHyperrectangle,
    C::CartesianProduct{N,<:Universe,<:LazySet},
) where {N}
    Y = second(C)
    Zp = LazySets.project(Z, (LazySets.dim(first(C))+1):LazySets.dim(C))
    return iszeromeasure(Y, Zp)
end

function iszeromeasure(X::LazySet, Y::LazySet)
    if ispolyhedral(X) && ispolyhedral(Y)
        return _iszeromeasure(X, Y)
    end

    error("iszeromeasure not implemented for $(typeof(X)) and $(typeof(Y))")
end

function _iszeromeasure(X::LazySet, Y::LazySet)
    if isdisjoint(X, Y)  # Short-circuit if there exists some set specific disjointness test.
        return true
    end

    n = LazySets.dim(X)
    if LazySets.dim(Y) != n
        throw(ArgumentError("Dimension mismatch: $(LazySets.dim(X)) != $(LazySets.dim(Y))"))
    end

    model = Model(HiGHS.Optimizer)
    set_string_names_on_creation(model, false)
    set_silent(model)

    @variable(model, x[1:n])
    @variable(model, x₀, upper_bound=1.0)

    @objective(model, Max, 1.0 * x₀)

    # Intuitively, if x₀ > 0 then the intersection is full-dimensional.
    H, h = tosimplehrep(X)
    @constraint(model, H * x .+ x₀ .≤ h)
    H, h = tosimplehrep(Y)
    @constraint(model, H * x .+ x₀ .≤ h)

    optimize!(model)

    iszeromeasure = objective_value(model) ≤ 0
    return iszeromeasure
end
