using LinearAlgebra.BLAS
using LinearAlgebra

"""
    algorithm78(ml0::Array{Float64,2}, ml1::Array{Float64,2}, ml2::Array{Float64,1})

Compute
b = ((X^T M^-1 X)^-1 X^T M^-1 y).

Requires at least Julia v1.0.

# Arguments
- `ml0::Array{Float64,2}`: Matrix X of size 2500 x 500 with property FullRank.
- `ml1::Array{Float64,2}`: Matrix M of size 2500 x 2500 with property SPD.
- `ml2::Array{Float64,1}`: Vector y of size 2500.
"""                    
function algorithm78(ml0::Array{Float64,2}, ml1::Array{Float64,2}, ml2::Array{Float64,1})
    # cost: 1.07e+10 FLOPs
    # X: ml0, full, M: ml1, full, y: ml2, full
    # (L7 L7^T) = M
    LAPACK.potrf!('L', ml1)

    # X: ml0, full, y: ml2, full, L7: ml1, lower_triangular
    # tmp12 = (L7^-1 X)
    trsm!('L', 'L', 'N', 'N', 1.0, ml1, ml0)

    # y: ml2, full, L7: ml1, lower_triangular, tmp12: ml0, full
    # tmp68 = (L7^-1 y)
    trsv!('L', 'N', 'N', ml1, ml2)

    # tmp12: ml0, full, tmp68: ml2, full
    ml3 = Array{Float64}(undef, 500)
    # tmp21 = (tmp12^T tmp68)
    gemv!('T', 1.0, ml0, ml2, 0.0, ml3)

    # tmp12: ml0, full, tmp21: ml3, full
    # (Q73 R74) = tmp12
    ml0 = qr!(ml0)

    # tmp21: ml3, full, Q73: ml0, QRfact_Q, R74: ml0, QRfact_R
    ml4 = ml0.R
    # tmp84 = (R74^-T tmp21)
    trsv!('U', 'T', 'N', ml4, ml3)

    # Q73: ml0, QRfact_Q, R74: ml4, full, tmp84: ml3, full
    # tmp85 = (R74^-1 tmp84)
    trsv!('U', 'N', 'N', ml4, ml3)

    # Q73: ml0, QRfact_Q, tmp85: ml3, full
    # b = tmp85
    return (ml3)
end