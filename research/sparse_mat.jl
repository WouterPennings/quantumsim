using SparseArrays
using BenchmarkTools
using CUDA

function dense_to_coo(A::Matrix{Float32})
    rowIdx = Int[]     # Stores row indices
    colIdx = Int[]     # Stores column indices
    values = Float32[] # Stores nonzero values

    rows, cols = size(A)

    for i in 1:rows
        for j in 1:cols
            if A[i, j] != 0.0  # Store only nonzero elements
                push!(rowIdx, i)
                push!(colIdx, j)
                push!(values, A[i, j])
            end
        end
    end

    return rowIdx, colIdx, values
end

function coo_spmv(rowIdx::Vector{Int}, colIdx::Vector{Int}, values::Vector{Float32}, v::Vector{Float32})
    out = zeros(Float32, length(v)) 

    nnz = length(values) 

    for i in 1:nnz
        out[rowIdx[i]] += values[i] * v[colIdx[i]]
    end

    return out
end

# function gpu_coo_spmv(rowIdx, colIdx, values, v, out)
#     idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x

#     if idx <= length(values)
#         out[rowIdx[idx]] += values[idx] * v[colIdx[idx]]
#     end

#     return
# end

# function run_gpu_coo_cpmv(rowIdx::Vector{Int}, colIdx::Vector{Int}, values::Vector{Float32}, v::Vector{Float32})::Vector{Float32}
#     threads = 256
#     blocks = Int(round(length(values)/threads))
#     # blocks = 1  
#     rowIdx_d = CuArray(rowIdx)
#     colIdx_d = CuArray(colIdx)
#     values_d = CuArray(values)
#     v_d = CuArray(v)
#     out_d = CUDA.zeros(Float32, length(v))

#     CUDA.@sync begin
#         @cuda threads=threads blocks=blocks gpu_coo_spmv(rowIdx_d, colIdx_d, values_d, v_d, out_d)
#     end

#     return Array(out_d)
# end

N = 2^15

m = sprand(Float32, N, N, 0.0005)
v = rand(Float32, N)
d = Array(m)
rowIdx, colIdx, values = dense_to_coo(Array(m))

display(@benchmark d*v)
display(@benchmark m*v)
display(@benchmark coo_spmv(rowIdx, colIdx, values, v))
