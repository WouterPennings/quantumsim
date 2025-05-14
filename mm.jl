using LinearAlgebra

n = 10

a = randn(Int, n, n)
b = randn(Int, n, n)
c = randn(Int, n, n)
d = randn(Int, n, n)
e = randn(Int, n, n)

m = randn(Int, n, n)

res1 = a * b * c * d * e
res1 = m * res1

res2 = m * a * b * c * d * e

println(res1)
println(res2)

println(res1-res2)