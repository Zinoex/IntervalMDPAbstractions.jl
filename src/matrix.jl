
struct AtomicSparseMatrixCOO{Tv,Ti<:Integer} <: AbstractMatrix{Tv}
    m::Int                 # Number of rows
    n::Int                 # Number of columns
    rows::Vector{Ti}       # Row indices of non-zero values
    cols::Vector{Ti}       # Column indices of non-zero values
    values::Vector{Tv}     # Non-zero values
    lock::ReentrantLock    # Lock for atomic operations

    function AtomicSparseMatrixCOO(
        m::Integer,
        n::Integer,
        rows::Vector{Ti},
        cols::Vector{Ti},
        values::Vector{Tv},
    ) where {Tv,Ti}
        if length(values) != length(rows) || length(values) != length(cols)
            throw(ArgumentError("values, rows, and cols must have the same length"))
        end
        return new{Tv,Ti}(Int(m), Int(n), rows, cols, values, ReentrantLock())
    end
end

AtomicSparseMatrixCOO{Tv,Ti}(::UndefInitializer, m::Integer, n::Integer) where {Tv,Ti} =
    AtomicSparseMatrixCOO(m, n, Ti[], Ti[], Tv[])
AtomicSparseMatrixCOO{Tv}(::UndefInitializer, m::Integer, n::Integer) where {Tv} =
    AtomicSparseMatrixCOO{Tv,Int32}(UndefInitializer(), m, n)
AtomicSparseMatrixCOO(::UndefInitializer, m::Integer, n::Integer) =
    AtomicSparseMatrixCOO{Float64}(UndefInitializer(), m, n)

Base.ndims(::Type{<:AtomicSparseMatrixCOO}) = 2
Base.size(A::AtomicSparseMatrixCOO) = (A.m, A.n)
Base.size(A::AtomicSparseMatrixCOO, i::Integer) = i == 1 ? A.m : A.n

function Base.setindex!(
    A::AtomicSparseMatrixCOO{Tv,Ti},
    v,
    i::Integer,
    j::Integer,
) where {Tv,Ti}
    @boundscheck checkbounds(A, i, j)

    v = convert(Tv, v)
    i = convert(Ti, i)
    j = convert(Ti, j)

    lock(A.lock) do
        push!(A.values, v)
        push!(A.rows, i)
        push!(A.cols, j)
    end
end
