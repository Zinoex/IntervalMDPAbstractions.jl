function iszeromeasure(X::AbstractHyperrectangle, Y::AbstractHyperrectangle)
    return any(high(Y, i) ≤ low(X, i) || high(X, i) ≤ low(Y, i) for i = 1:LazySets.dim(X))
end

iszeromeasure(X::EmptySet, Y::LazySet) = true
iszeromeasure(X::LazySet, Y::EmptySet) = true
iszeromeasure(X::EmptySet, Y::EmptySet) = true
