using LinearAlgebra
using Test
using CUDA
using BenchmarkTools

function cpu_kron(a, b) 
    s_a = size(a)
    s_b = size(b)

    height = s_a[1]*s_b[1]
    width  = s_a[2]*s_b[2]
    out::Matrix{Int} = zeros(height, width)

    for x = 1:width
        for y in 1:height
            out[y, x] = b[((y-1) % s_b[1])+1, ((x-1) % s_b[2])+1] * a[Int(ceil(y/s_b[1])), Int(ceil(x/s_b[2]))]
        end
    end

    return out
end

function GPU_kron(a, b, out, s_b)
    x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    y = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    out[y, x] = b[((y-1) % s_b[1])+1, ((x-1) % s_b[2])+1] * a[Int(ceil(y/s_b[1])), Int(ceil(x/s_b[2]))]

    return
end

function run_GPU_kron(a, b)
    s_a = size(a)
    s_b = size(b)
    height = s_a[1]*s_b[1]
    width  = s_a[2]*s_b[2]

    d_a = CuArray(a)
    d_b = CuArray(b)
    d_s_b = CuArray([s_b[1], s_b[2]])
    d_out = CUDA.zeros(Int32, height, width)

    CUDA.@sync begin
        @cuda threads=(16, 16) blocks=(512, 512) GPU_kron(d_a, d_b, d_out, d_s_b)
    end

    return Array(d_out)
end

a = rand(1:10, 4096, 4096)
b = Matrix(1 * I(2))

display(a)
display(b)

# c = kron(a, b)
# c2 = cpu_kron(a, b)
# c3 = run_GPU_kron(a, b)

for _ in 1:10
    display(CUDA.@profile trace=true run_GPU_kron(a, b))
end

display(@benchmark kron(a, b))
display(@benchmark cpu_kron(a, b))

# display(c)
# display(c2)
# display(c3)

@test all(c .== c2)
@test all(c .== c3)