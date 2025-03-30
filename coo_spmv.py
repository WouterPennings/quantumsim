import ctypes
import numpy as np 
import scipy.sparse as sparse

# Load shared library
coo_lib = ctypes.CDLL("./util.dll")  # Use "coo_spmv.dll" on Windows

# Define the function signature
coo_lib.coo_spmv_row.argtypes = [
    ctypes.POINTER(ctypes.c_int),   # rowIdx
    ctypes.POINTER(ctypes.c_int),   # colIdx
    ctypes.POINTER(ctypes.c_double), # values
    ctypes.POINTER(ctypes.c_double), # v
    ctypes.POINTER(ctypes.c_double), # out
    ctypes.c_int,  # nnz (number of non-zero elements)
    ctypes.c_int   # n (size of output vector)
]

def coo_spmv_c(rowIdx: np.ndarray, colIdx: np.ndarray, values: np.ndarray, v: np.ndarray) -> np.ndarray:
    assert rowIdx.dtype == np.int32, "rowIdx must be of dtype np.int32"
    assert colIdx.dtype == np.int32, "colIdx must be of dtype np.int32"
    assert values.dtype == np.float64, "values must be of dtype np.float64"
    assert v.dtype == np.float64, "v must be of dtype np.float64"

    nnz = len(values)  # Number of non-zero elements
    n = len(v)         # Size of output vector

    # Allocate output array (zero-initialized)
    out = np.zeros(n, dtype=np.float64)

    # Call the C function
    coo_lib.coo_spmv_row(
        rowIdx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        colIdx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        nnz,
        n
    )

    out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    out_array = np.ctypeslib.as_array(out_ptr, shape=(1, n))

    return out_array

m = np.eye(5, dtype=np.float64)
m[1, 2] = 4.3
m[2, 4] = 6.9

s = sparse.coo_matrix(m)

# v = np.array([2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float64)
v = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)

print(m)
print(v)

out = coo_spmv_c(s.row, s.col, s.data, v)
print(out)