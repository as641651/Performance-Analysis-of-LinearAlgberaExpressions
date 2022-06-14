using LinearAlgebra.BLAS
using LinearAlgebra

"""
    algorithm72(ml0::Array{Float64,2}, ml1::Array{Float64,2}, ml2::Array{Float64,1})

Compute
b = ((X^T M^-1 X)^-1 X^T M^-1 y).

Requires at least Julia v1.0.

# Arguments
- `ml0::Array{Float64,2}`: Matrix X of size 2500 x 500 with property FullRank.
- `ml1::Array{Float64,2}`: Matrix M of size 2500 x 2500 with property SPD.
- `ml2::Array{Float64,1}`: Vector y of size 2500.
"""                    
function algorithm72(ml0::Array{Float64,2}, ml1::Array{Float64,2}, ml2::Array{Float64,1})
    # cost: 1.06e+10 FLOPs
    # X: ml0, full, M: ml1, full, y: ml2, full
    # (L7 L7^T) = M
    LAPACK.potrf!('L', ml1)

    # X: ml0, full, y: ml2, full, L7: ml1, lower_triangular
    # tmp12 = (L7^-1 X)
    trsm!('L', 'L', 'N', 'N', 1.0, ml1, ml0)

    # y: ml2, full, L7: ml1, lower_triangular, tmp12: ml0, full
    ml3 = Array{Float64}(undef, 500, 500)
    # tmp14 = (tmp12^T tmp12)
    syrk!('L', 'T', 1.0, ml0, 0.0, ml3)

    # y: ml2, full, L7: ml1, lower_triangular, tmp12: ml0, full, tmp14: ml3, symmetric_lower_triangular
    # tmp68 = (L7^-1 y)
    trsv!('L', 'N', 'N', ml1, ml2)

    # tmp12: ml0, full, tmp14: ml3, symmetric_lower_triangular, tmp68: ml2, full
    ml4 = Array{Float64}(undef, 500)
    # (Z18 W19 Z18^T) = tmp14
    ml4, ml3 = LAPACK.syev!('V', 'L', ml3)

    # tmp12: ml0, full, tmp68: ml2, full, W19: ml4, diagonal_vector, Z18: ml3, full
    ml5 = Array{Float64}(undef, 500, 2500)
    # tmp477 = (Z18^T tmp12^T)
    gemm!('T', 'T', 1.0, ml3, ml0, 0.0, ml5)

    # tmp68: ml2, full, W19: ml4, diagonal_vector, Z18: ml3, full, tmp477: ml5, full
    # tmp478 = (W19^-1 tmp477)
    ml5 ./= ml4;

    # tmp68: ml2, full, Z18: ml3, full, tmp478: ml5, full
    ml6 = Array{Float64}(undef, 500)
    # tmp269 = (tmp478 tmp68)
    gemv!('N', 1.0, ml5, ml2, 0.0, ml6)

    # Z18: ml3, full, tmp269: ml6, full
    ml7 = Array{Float64}(undef, 500)
    # tmp24 = (Z18 tmp269)
    gemv!('N', 1.0, ml3, ml6, 0.0, ml7)

    # tmp24: ml7, full
    # b = tmp24
    return (ml7)
end