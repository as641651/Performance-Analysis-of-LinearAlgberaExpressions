using LinearAlgebra.BLAS
using LinearAlgebra

"""
    algorithm75(ml0::Array{Float64,2}, ml1::Array{Float64,2}, ml2::Array{Float64,1})

Compute
b = ((X^T M^-1 X)^-1 X^T M^-1 y).

Requires at least Julia v1.0.

# Arguments
- `ml0::Array{Float64,2}`: Matrix X of size 2500 x 500 with property FullRank.
- `ml1::Array{Float64,2}`: Matrix M of size 2500 x 2500 with property SPD.
- `ml2::Array{Float64,1}`: Vector y of size 2500.
"""                    
function algorithm75(ml0::Array{Float64,2}, ml1::Array{Float64,2}, ml2::Array{Float64,1})
    # cost: 1.07e+10 FLOPs
    # X: ml0, full, M: ml1, full, y: ml2, full
    # (L7 L7^T) = M
    LAPACK.potrf!('L', ml1)

    # X: ml0, full, y: ml2, full, L7: ml1, lower_triangular
    # tmp12 = (L7^-1 X)
    trsm!('L', 'L', 'N', 'N', 1.0, ml1, ml0)

    # y: ml2, full, L7: ml1, lower_triangular, tmp12: ml0, full
    # (Q73 R74) = tmp12
    ml0 = qr!(ml0)

    # y: ml2, full, L7: ml1, lower_triangular, Q73: ml0, QRfact_Q, R74: ml0, QRfact_R
    # tmp68 = (L7^-1 y)
    trsv!('L', 'N', 'N', ml1, ml2)

    # Q73: ml0, QRfact_Q, R74: ml0, QRfact_R, tmp68: ml2, full
    ml3 = Array(ml0.Q)
    ml4 = Array{Float64}(undef, 500)
    # tmp78 = (Q73^T tmp68)
    gemv!('T', 1.0, ml3, ml2, 0.0, ml4)

    # R74: ml0, QRfact_R, tmp78: ml4, full
    ml5 = ml0.R
    # tmp79 = (R74^-1 tmp78)
    trsv!('U', 'N', 'N', ml5, ml4)

    # tmp79: ml4, full
    # b = tmp79
    return (ml4)
end