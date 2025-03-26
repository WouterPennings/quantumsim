import numpy as np
import scipy.sparse as sc
from timeit import timeit 

N = 2**1
m = sc.rand(N, N, density=0.0005)

print(m.size)
print(m.toarray())
print(timeit(m.tocsc, number=10))