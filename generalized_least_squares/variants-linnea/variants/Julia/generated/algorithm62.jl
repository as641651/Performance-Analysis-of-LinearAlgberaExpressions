using LinearAlgebra.BLAS
using LinearAlgebra

"""
    algorithm62(ml0::Array{Float64,2}, ml1::Array{Float64,2}, ml2::Array{Float64,1})

Compute
b = ((X^T M^-1 X)^-1 X^T M^-1 y).

Requires at least Julia v1.0.

# Arguments
- `ml0::Array{Float64,2}`: Matrix X of size 2500 x 500 with property FullRank.
- `ml1::Array{Float64,2}`: Matrix M of size 2500 x 2500 with property SPD.
- `ml2::Array{Float64,1}`: Vector y of size 2500.
"""                    
function algorithm62(ml0::Array{Float64,2}, ml1::Array{Float64,2}, ml2::Array{Float64,1})
    # cost: 1.06e+10 FLOPs
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
    ml3 = Array{Float64}(undef, 500, 500)
    # tmp14 = (tmp12^T tmp12)
    syrk!('L', 'T', 1.0, ml0, 0.0, ml3)

    # tmp12: ml0, full, tmp68: ml2, full, tmp14: ml3, symmetric_lower_triangular
    for i = 1:500-1;
        view(ml3, i, i+1:500)[:] = view(ml3, i+1:500, i);
    end;
    # (Q16 R17) = tmp14
    ml3 = qr!(ml3)

    # tmp12: ml0, full, tmp68: ml2, full, Q16: ml3, QRfact_Q, R17: ml3, QRfact_R
    ml4 = Array(ml3.Q)
    ml5 = Array{Float64}(undef, 500, 2500)
    # tmp86 = (Q16^T tmp12^T)
    gemm!('T', 'T', 1.0, ml4, ml0, 0.0, ml5)

    # tmp68: ml2, full, R17: ml3, QRfact_R, tmp86: ml5, full
    ml6 = Array{Float64}(undef, 500)
    # tmp25 = (tmp86 tmp68)
    gemv!('N', 1.0, ml5, ml2, 0.0, ml6)

    # R17: ml3, QRfact_R, tmp25: ml6, full
    ml7 = ml3.R
    # tmp24 = (R17^-1 tmp25)
    trsv!('U', 'N', 'N', ml7, ml6)

    # tmp24: ml6, full
    # b = tmp24
    return (ml6)
end