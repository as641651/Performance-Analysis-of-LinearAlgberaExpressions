using LinearAlgebra.BLAS
using LinearAlgebra

"""
    algorithm5(ml0::Array{Float64,2}, ml1::Array{Float64,2}, ml2::Array{Float64,2}, ml3::Array{Float64,2})

Compute
Y = (A B C D).

Requires at least Julia v1.0.

# Arguments
- `ml0::Array{Float64,2}`: Matrix A of size 10 x 100 with property FullRank.
- `ml1::Array{Float64,2}`: Matrix B of size 100 x 80 with property FullRank.
- `ml2::Array{Float64,2}`: Matrix C of size 80 x 150 with property FullRank.
- `ml3::Array{Float64,2}`: Matrix D of size 150 x 120 with property FullRank.
"""                    
function algorithm5(ml0::Array{Float64,2}, ml1::Array{Float64,2}, ml2::Array{Float64,2}, ml3::Array{Float64,2})
    # cost: 6.24e+06 FLOPs
    # A: ml0, full, B: ml1, full, C: ml2, full, D: ml3, full
    ml4 = Array{Float64}(undef, 100, 150)
    # tmp2 = (B C)
    gemm!('N', 'N', 1.0, ml1, ml2, 0.0, ml4)

    # A: ml0, full, D: ml3, full, tmp2: ml4, full
    ml5 = Array{Float64}(undef, 100, 120)
    # tmp5 = (tmp2 D)
    gemm!('N', 'N', 1.0, ml4, ml3, 0.0, ml5)

    # A: ml0, full, tmp5: ml5, full
    ml6 = Array{Float64}(undef, 10, 120)
    # tmp6 = (A tmp5)
    gemm!('N', 'N', 1.0, ml0, ml5, 0.0, ml6)

    # tmp6: ml6, full
    # Y = tmp6
    return (ml6)
end