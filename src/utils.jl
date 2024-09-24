
function efficient_hcat(X::Vector{Vector{Tv}}) where {Tv}
    return reduce(hcat, X)
end

function efficient_hcat(X::Vector{SparseVector{Tv, Ti}}) where {Tv, Ti}
    # check sizes
    n = length(X)
    m = length(X[1])
    tnnz = nnz(X[1])
    for j = 2:n
        length(X[j]) == m ||
            throw(DimensionMismatch("Inconsistent column lengths."))
        tnnz += nnz(X[j])
    end

    # construction
    colptr = Vector{Ti}(undef, n+1)
    nzrow = Vector{Ti}(undef, tnnz)
    nzval = Vector{Tv}(undef, tnnz)
    roff = 1
    @inbounds for j = 1:n
        xj = X[j]
        xnzind = SparseArrays.nonzeroinds(xj)
        xnzval = nonzeros(xj)
        colptr[j] = roff
        copyto!(nzrow, roff, xnzind)
        copyto!(nzval, roff, xnzval)
        roff += length(xnzind)
    end
    colptr[n+1] = roff
    return SparseMatrixCSC{Tv,Ti}(m, n, colptr, nzrow, nzval)
end

function iszeromeasure(X::AbstractHyperrectangle, Y::AbstractHyperrectangle)
    return any(high(Y, i) ≤ low(X, i) || high(X, i) ≤ low(Y, i) for i in 1:LazySets.dim(X))
end

iszeromeasure(X::EmptySet, Y::LazySet) = true
iszeromeasure(X::LazySet, Y::EmptySet) = true
iszeromeasure(X::EmptySet, Y::EmptySet) = true