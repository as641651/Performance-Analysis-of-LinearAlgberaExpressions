using LinearAlgebra.BLAS
using LinearAlgebra

"""
    algorithm4(ml0::Array{Float64,2}, ml1::Array{Float64,2}, ml2::Array{Float64,1})

Compute
b = ((X^T M^-1 X)^-1 X^T M^-1 y).

Requires at least Julia v1.0.

# Arguments
- `ml0::Array{Float64,2}`: Matrix X of size 2500 x 500 with property FullRank.
- `ml1::Array{Float64,2}`: Matrix M of size 2500 x 2500 with property SPD.
- `ml2::Array{Float64,1}`: Vector y of size 2500.
"""                    
function algorithm4(ml0::Array{Float64,2}, ml1::Array{Float64,2}, ml2::Array{Float64,1})
    # cost: 9.01e+09 FLOPs
    # X: ml0, full, M: ml1, full, y: ml2, full
    # (L7 L7^T) = M
    LAPACK.potrf!('L', ml1)

    # X: ml0, full, y: ml2, full, L7: ml1, lower_triangular
    # tmp68 = (L7^-1 y)
    trsv!('L', 'N', 'N', ml1, ml2)

    # X: ml0, full, L7: ml1, lower_triangular, tmp68: ml2, full
    # tmp12 = (L7^-1 X)
    trsm!('L', 'L', 'N', 'N', 1.0, ml1, ml0)

    # tmp68: ml2, full, tmp12: ml0, full
    ml3 = Array{Float64}(undef, 500)
    # tmp21 = (tmp12^T tmp68)
    gemv!('T', 1.0, ml0, ml2, 0.0, ml3)

    # tmp12: ml0, full, tmp21: ml3, full
    ml4 = Array{Float64}(undef, 500, 500)
    # tmp14 = (tmp12^T tmp12)
    syrk!('L', 'T', 1.0, ml0, 0.0, ml4)

    # tmp21: ml3, full, tmp14: ml4, symmetric_lower_triangular
    # (L15 L15^T) = tmp14
    LAPACK.potrf!('L', ml4)

    # tmp21: ml3, full, L15: ml4, lower_triangular
    # tmp23 = (L15^-1 tmp21)
    trsv!('L', 'N', 'N', ml4, ml3)

    # L15: ml4, lower_triangular, tmp23: ml3, full
    # tmp24 = (L15^-T tmp23)
    trsv!('L', 'T', 'N', ml4, ml3)

    # tmp24: ml3, full
    # b = tmp24
    return (ml3)
end